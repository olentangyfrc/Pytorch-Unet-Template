# Module that handles data samples while training and validating

from data_loading import load_img, load_labelme_data
from typing import Callable, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import albumentations as A
import numpy as np
import os
from torch.utils.data import Dataset as BaseDataset
import glob
import hashlib
from tqdm import tqdm


# Probability of being assigned to each dataset
DATASET_SPLIT = {
    "train": 0.85,
    "val": 0.15
}

# Seed for dataset assignment
DATASET_SEED = 8766

# How much to scale the image while training
SCALE = 0.25

# Whether the input is grayscale
CHANNELS = 3


class Dataset(BaseDataset):
    """Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        augmentation (Amentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (Amentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
        shuffle (bool): whether to shuffle the list of samples

    """

    def __init__(
            self,
            image_dir,
            mask_dir,
            augmentation=None,
            preprocessing=None,
            shuffle=True,

    ):
        self.ids = glob.glob(image_dir + "/*.npy")
        self.image_fps = [os.path.join(image_dir, os.path.basename(image_id))
                          for image_id in self.ids]
        self.mask_fps = [os.path.join(mask_dir, os.path.basename(image_id))
                         for image_id in self.ids]

        self.samples = list(zip(self.image_fps, self.mask_fps))

        if shuffle:
            np.random.shuffle(self.samples)

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        image_fp, mask_fp = self.samples[i]
        image = np.load(image_fp, mmap_mode='r')
        mask = np.load(mask_fp, mmap_mode='r')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image.astype('float32'), mask.astype('uint8')

    def __len__(self):
        return len(self.samples)


class DataModule(LightningDataModule):
    """
    LightningDataModule for segmentation dataset. Loads the data, 
    assigns datasets, and applies augmentation

    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        classes: list = [],
        preprocessing_augmentation: Optional[Callable[[
            int], A.BasicTransform]] = None,
        validation_augmentation: Optional[A.BasicTransform] = None,
        persistent_workers: Optional[bool] = None
    ):
        super().__init__()

        self.data_dir = data_dir
        self.training_dir = data_dir + "training_data/"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.preprocessing_augmentation = preprocessing_augmentation
        self.validation_augmentation = validation_augmentation
        self.classes = classes
        self.persistent_workers = persistent_workers
        if self.persistent_workers is None:
            self.persistent_workers = num_workers > 0
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Converts the inputs into smaller, easier to manage numpy files"""

        json_paths = glob.glob(
            self.data_dir + "labelme_data/**/*.json", recursive=True)
        print("Verifying and updating training data files...")

        for json_path in tqdm(json_paths):

            img_name = json_path.replace(self.data_dir + "labelme_data", "").replace(
                ".json", "").replace("/", " ").replace("\\", " ").strip()

            # Read labels
            output = load_labelme_data(json_path, SCALE, self.classes)

            # Load image
            img = load_img(json_path, SCALE, CHANNELS)

            # Get which dataset this sample should be in
            seed = (int(hashlib.sha1(img_name.encode("utf-8")
                                     ).hexdigest(), 16) + DATASET_SEED) % (1 << 32)
            rng = np.random.default_rng(seed)
            dataset_selection_num = rng.random()
            for dataset, proportion in DATASET_SPLIT.items():
                if dataset_selection_num < proportion:
                    selected_dataset_name = dataset
                    break
                dataset_selection_num -= proportion

            # Create paths
            input_path = os.path.join(
                self.training_dir + selected_dataset_name)
            output_path = input_path + 'annot'
            os.makedirs(input_path, exist_ok=True)
            os.makedirs(output_path, exist_ok=True)

            # Save npy files
            np_name = img_name + ".npy"

            if len(output.shape) == 2:
                output = output[..., np.newaxis]
            with open(os.path.join(output_path, np_name), 'wb') as f:
                np.save(f, output)

            with open(os.path.join(input_path, np_name), 'wb') as f:
                np.save(f, img)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        self.data_train = Dataset(
            self.training_dir + "train",
            self.training_dir + "trainannot",
            self.get_training_augmentation(),
            self.preprocessing_augmentation
        )
        self.data_val = Dataset(
            self.training_dir + "val",
            self.training_dir + "valannot",
            self.get_validation_augmentation(),
            self.preprocessing_augmentation
        )
        self.data_test = Dataset(
            self.training_dir + "test",
            self.training_dir + "testannot",
            self.get_validation_augmentation(),
            self.preprocessing_augmentation
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )

    # The following is the list of augmentations applied to training samples.
    # Look up the Albumentations API to view available augmentations.
    def get_training_augmentation(self):
        train_transform = [

            A.HorizontalFlip(p=0.5),

            A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=10, shift_limit=0.2, p=1),

            A.GaussNoise(p=0.2),
            A.Perspective(p=0.5),

            A.OneOf(
                [
                    A.CLAHE(p=1),
                    A.RandomBrightnessContrast(p=1, contrast_limit=0),
                    A.RandomGamma(p=1),
                ],
                p=0.9,
            ),

            A.OneOf(
                [
                    A.Sharpen(p=1),
                    A.Blur(blur_limit=3, p=1),
                    A.MotionBlur(blur_limit=3, p=1),
                ],
                p=0.9,
            ),

            A.OneOf(
                [
                    A.RandomBrightnessContrast(p=1, brightness_limit=0),
                    A.HueSaturationValue(p=1),
                ],
                p=0.9,
            ),
        ]
        return A.Compose(train_transform)

    def get_validation_augmentation(self):
        if self.validation_augmentation is not None:
            return self.validation_augmentation
        return A.Compose([])
