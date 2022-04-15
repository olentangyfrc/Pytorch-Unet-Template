# Module that handles data samples while training and validating

from data_loading import load_img, load_labelme_data
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
from torch.utils.data import Dataset as BaseDataset
import glob
from tqdm import tqdm
import re

def flip_left_right(image, **kwargs):
    image = np.flip(image,1)
    return image

def flip_channels(image, **kwargs):
    image = np.flip(image,1)
    image = np.flip(image,2)
    return image

# Names of the videos to be used as validation
VALIDATION_VIDEOS = [
    "ringlightpictures",
    "hallway"
]

# How much to scale the image while training. Useful for speeding up training and detection
SCALE = 1

# How many channels the data is. 3 for RGB, 1 for grayscale
CHANNELS = 3


class Dataset(BaseDataset):
    """Dataset that loads data for training. Read images, apply augmentation and preprocessing transformations.

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
        # Initialize lists of all samples
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
        """ Called to get each batch before each step """

        # Read data
        image_fp, mask_fp = self.samples[i]
        image = np.load(image_fp, mmap_mode='r')
        mask = np.load(mask_fp, mmap_mode='r')

        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

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
        preprocessing_augmentation = None,
        validation_augmentation = None,
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
        self.prepared = False
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.loader_train = None
        self.loader_val = None
        self.loader_test = None

    def prepare_data(self):
        """ Converts the labelme data to numpy training files """

        if not self.prepared:
            # Get a list of all labeled examples
            json_paths = glob.glob(
                self.data_dir + "labelme_data/**/*.json", recursive=True)
            print("Verifying and updating training data files...")

            # Get names of the different videos
            frames = {}
            for json_path in json_paths:
                basename = os.path.basename(json_path)

                frame_num = int(re.findall(r'\d+', basename)[-1])
                video_name = basename.replace("%d.json" % frame_num, "")

                if video_name not in frames:
                    frames[video_name] = {}
                frames[video_name][frame_num] = json_path

            # Prepare each video
            positive_pixel_percentages = []
            for video_name, video_frames in tqdm(frames.items()):
                selected_dataset_name = "train"
                for val_video_name in VALIDATION_VIDEOS:
                    if val_video_name in video_name:
                        selected_dataset_name = "val"

                # Create paths
                input_path = os.path.join(
                    self.training_dir + selected_dataset_name)
                output_path = input_path + 'annot'
                os.makedirs(input_path, exist_ok=True)
                os.makedirs(output_path, exist_ok=True)

                # For each frame in that video
                for frame_num, json_path in tqdm(video_frames.items()):

                    img_name = json_path.replace(self.data_dir + "labelme_data", "").replace(
                        ".json", "").replace("/", " ").replace("\\", " ").strip()

                    # Read labels
                    output = load_labelme_data(json_path, SCALE, self.classes)
                    positive_pixel_percentages.append(np.mean(output))

                    # Load image
                    img = load_img(json_path, SCALE, CHANNELS)

                    # Save npy files
                    np_name = img_name + ".npy"

                    if len(output.shape) == 2:
                        output = output[..., np.newaxis]
                    with open(os.path.join(output_path, np_name), 'wb') as f:
                        np.save(f, output)

                    with open(os.path.join(input_path, np_name), 'wb') as f:
                        np.save(f, img)

            self.positive_pixel_percentage = np.mean(positive_pixel_percentages)
            self.prepared = True

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
        if self.loader_train is None:
            self.loader_train = DataLoader(
                dataset=self.data_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                shuffle=True,
            )
        return self.loader_train

    def val_dataloader(self):
        if self.loader_val is None:
            self.loader_val = DataLoader(
                dataset=self.data_val,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                shuffle=False,
            )
        return self.loader_val

    def test_dataloader(self):
        if self.loader_test is None:
            self.loader_test = DataLoader(
                dataset=self.data_test,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                shuffle=False,
            )
        return self.loader_test

    # The following is the list of augmentations applied to training samples.
    # Look up the Albumentations API to view available augmentations.
    def get_training_augmentation(self):
        import albumentations as A
        train_transform = [
            A.Resize(480, 640),
            A.Lambda(image=flip_left_right, mask=flip_channels, p=0.5),

            A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=10, shift_limit=0.2, border_mode=cv2.BORDER_CONSTANT, mask_value=(0,0), value=(0,0,0), p=1),

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
        import albumentations as A
        if self.validation_augmentation is not None:
            return self.validation_augmentation
        return A.Compose([A.Resize(480, 640)])
