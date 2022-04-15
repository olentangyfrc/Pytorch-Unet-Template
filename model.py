# Pytorch lightning model for segmentation

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import segmentation_models_pytorch as smp
import albumentations as A
import torch
from model_prod import SegModel
from math import sqrt



class SegModelLightning(LightningModule):
    def __init__(self, in_ch, num_classes, encoder, lr, percent_positive, initial_weights='imagenet', **kwargs):
        super(SegModelLightning, self).__init__()

        # Initialize our low level model
        self.model_prod = SegModel(in_ch, num_classes, encoder, initial_weights)

        # Log our hyperparameters to view in tensorboard
        self.save_hyperparameters()

        # Initialize our loss function and validation metric
        self.bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1/sqrt(percent_positive)))
        self.iou_metric = smp.utils.metrics.IoU(threshold=0.9)

        self.learning_rate = lr
        self.num_classes = num_classes
        self.channels = in_ch

   
    def training_step(self, batch, batch_idx):
        """ Function that computes the loss for every training step """
        inputs, labels = batch

        # Conpute region that needs to be ignored
        ignore_mask = labels == 255

        # Get raw logits
        outputs = self.model_prod.forward(inputs, sigmoid=False)

        # Apply ignore mask by hard coding those pixels with 0 error
        outputs[ignore_mask] = -1000 # False detection is a very small logit
        labels[ignore_mask] = 0

        # Compute BCE loss
        loss = self.bce_loss(outputs, labels)

        # Log loss
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch

        # Conpute region that needs to be ignored
        ignore_mask = labels == 255

        # Masks out uncertain pixels. No loss will be computed for them
        outputs = self(inputs)

        # Apply ignore mask by hard coding those pixels with 0 error
        outputs[ignore_mask] = 0
        labels[ignore_mask] = 0

        # Compute iou for our samples
        iou = self.iou_metric(outputs, labels)

        # Log our IOU
        self.log('val_iou', iou, prog_bar=True)

    def forward(self, x) -> torch.Tensor:
        return self.model_prod.forward(x)

    def configure_optimizers(self):
        return torch.optim.Adam([dict(params=self.model_prod.parameters(), lr=self.learning_rate)])

    @staticmethod
    def get_preprocessing():
        """Generates preprocessing transform

        Returns:
            A.BasicTransform: The preprocessing transform
        """

        _transform = [
            A.Lambda(image=SegModel.normalize_image),
            A.Lambda(image=SegModel.fix_shape, mask=SegModel.fix_shape),
        ]

        return A.Compose(_transform)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        # All command line arguments for the model
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument(
            '--encoder', type=str, default='mobilenet_v2', choices=smp.encoders.encoders.keys())
        parser.add_argument('--lr', type=float, default=0.0001)
        return parent_parser
