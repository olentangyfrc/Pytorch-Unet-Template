# Training script

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from datamodule import CHANNELS, DataModule
from model import SegModelLightning
import torch
import numpy as np

##################################################
## Training Parameters.                         ##
## Can be changed with command line args too    ##
##################################################

# Where labelme_data is. Should be "data/" unless you move that folder to another drive
DATA_DIR = 'data/'

# Which GPU IDs to use. Set to None to use CPU
GPUS = [0]
if not torch.cuda.is_available():
    GPUS = None

# List the classes from your dataset that you want to train on
CLASSES = [
    "left",
    "right"
]

# Whether to use 16 bit precision. Speeds up training on newer GPUs.
# Can only be 16 if using a GPU
PRECISION = 16 if GPUS is not None else 32

# Whether to benchmark. Speeds up training when samples are same size
# They are in this case because they are being tiled
# Only works on GPU
BENCHMARK = True if GPUS is not None else False

# How many samples to look at at a time.
# Large batch size will cause memory errors
# If you get an out of memory, divide this number by 2
BATCH_SIZE = 8

# Number of data loader workers. More workers uses more RAM but may be faster.
NUM_WORKERS = 4


def normalize(img):
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    img = img[::-1] * 60 + 60
    img = np.clip(img, 0, 255)
    return img

def labels_to_rgb(labels):
    img = np.zeros((3, *labels.shape[1:]), 'uint8')
    num_channels_displayed = min(3, labels.shape[0])
    img[:num_channels_displayed] = labels[:num_channels_displayed]
    return img * 255

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # Add command line args
    parser = SegModelLightning.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # Set command line arg defaults
    # Any command line args you dont want to specify every time list here
    parser.set_defaults(
        gpus=GPUS,
        precision=PRECISION,
        benchmark=BENCHMARK
    )

    args = parser.parse_args()

    # Initialize data module and make sure npy files are generated
    datamodule = DataModule(
        preprocessing_augmentation=SegModelLightning.get_preprocessing(),
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        classes=CLASSES
    )
    datamodule.prepare_data()

    # Initialize model from command line args
    dict_args = vars(args)
    model = SegModelLightning(
        in_ch=CHANNELS,
        num_classes=len(CLASSES),
        percent_positive=datamodule.positive_pixel_percentage,
        **dict_args
    )

    # Init logger and add example samples
    tb_logger = pl_loggers.TensorBoardLogger('logs/')

    # Plot samples
    datamodule.setup()
    training_sample = next(iter(datamodule.train_dataloader()))
    validation_sample = next(iter(datamodule.val_dataloader()))
    tb_logger.experiment.add_image(
        "Training input", img_tensor=normalize(training_sample[0][0, :3])/255)
    tb_logger.experiment.add_image(
        "Training label", img_tensor=labels_to_rgb(training_sample[1][0]))
    tb_logger.experiment.add_image(
        "Validation input", img_tensor=normalize(validation_sample[0][0, :3])/255)
    tb_logger.experiment.add_image(
        "Validation label", img_tensor=labels_to_rgb(validation_sample[1][0]))

    # Stop early if val_iou hasn't improved
    # Will wait patience epochs before stopping
    early_stop_callback = EarlyStopping(
        monitor='val_iou',
        min_delta=0.00,
        patience=200,
        verbose=False,
        mode='max'
    )

    # Save the best val_iou
    checkpoint_callback = ModelCheckpoint(
        monitor='val_iou',
        filename='{epoch:02d}-{val_iou:.2f}',
        mode='max',
    )

    # Init trainer from command line args
    trainer: Trainer = Trainer.from_argparse_args(
        args,
        max_epochs=10000,
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    # Begin training
    trainer.fit(model, datamodule)

    # Log best val_iou for this set of hyperparameters
    trainer.logger.log_metrics({'hp_metric': early_stop_callback.best_score})
