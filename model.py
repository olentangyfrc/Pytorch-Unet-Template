# Pytorch lightning model for segmentation

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import segmentation_models_pytorch as smp
import albumentations as A
import torch
import numpy as np


def to_tensor(x, **kwargs):
    if len(x.shape) == 2:
        x = x[..., np.newaxis]
    return x.transpose(2, 0, 1).astype('float32')


def make_divisible(x, **kwargs):
    if len(x.shape) == 2:
        x = x[..., np.newaxis]
    w_pad = (32 - x.shape[1]) % 32
    h_pad = (32 - x.shape[0]) % 32
    return np.pad(x, ((0, h_pad), (0, w_pad), (0, 0)))


class SegModel(LightningModule):
    def __init__(self, in_ch, num_classes, encoder, lr, **kwargs):
        super(SegModel, self).__init__()

        self.model = smp.FPN(
            in_channels=in_ch,
            encoder_name=encoder,
            encoder_weights='imagenet',
            classes=num_classes,
            activation='sigmoid'
        )

        self.save_hyperparameters()

        self.criterion = smp.utils.losses.DiceLoss()

        self.learning_rate = lr
        self.num_classes = num_classes
        self.channels = in_ch

    def forward(self, x) -> torch.Tensor:
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        outputs = self.model(inputs)

        # Compute loss and iou for each class and average them
        loss = torch.zeros(1, device=inputs.device)
        for i in range(self.num_classes):
            loss += self.criterion(outputs[:, i], labels[:, i])

        # Log IOU and loss
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch

        # Masks out uncertain pixels. No loss will be computed for them
        outputs = self.model(inputs)

        # Compute iou for each class and average
        intersections = torch.zeros(self.num_classes, device=inputs.device)
        unions = torch.zeros(self.num_classes, device=inputs.device)
        for i in range(self.num_classes):
            pred = outputs[:, i] > 0.5
            intersection = torch.sum(pred * labels[:, i])
            intersections[i] += intersection
            unions[i] += torch.sum(pred) + \
                torch.sum(labels[:, i]) - intersection

        return {'intersections': intersections, 'unions': unions}

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        intersections = torch.zeros(self.num_classes, device=self.device)
        unions = torch.zeros(self.num_classes, device=self.device)
        for pred in outputs:
            intersections += pred['intersections']
            unions += pred['unions']

        iou = torch.zeros(1, device=self.device)
        for c in range(self.num_classes):
            iou += intersections[c] / unions[c] / self.num_classes

        self.log('val_iou', iou.item())

    def configure_optimizers(self):
        return torch.optim.Adam([dict(params=self.model.parameters(), lr=self.learning_rate)])

    def get_preprocessing(self):
        """Generates preprocessing transform

        Returns:
            A.BasicTransform: The preprocessing transform
        """

        mean = std = 60
        if self.channels == 3:
            mean = [60, 60, 60]
            std = [60, 60, 60]

        _transform = [
            A.Lambda(image=make_divisible, mask=make_divisible),
            A.Normalize(mean=mean, std=std, max_pixel_value=1),
            A.Lambda(image=to_tensor, mask=to_tensor),
        ]

        return A.Compose(_transform)

    @staticmethod
    def add_model_specific_args(parent_parser):
        # All command line arguments for the model
        parser = parent_parser.add_argument_group("LitSegModel")
        parser.add_argument(
            '--encoder', type=str, default='mobilenet_v2', choices=smp.encoders.encoders.keys())
        parser.add_argument('--lr', type=float, default=0.00005)
        return parent_parser
