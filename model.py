# Pytorch lightning model for segmentation

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import segmentation_models_pytorch as smp
import albumentations as A
import torch
from model_prod import SegModel




class SegModelLightning(LightningModule):
    def __init__(self, in_ch, num_classes, encoder, lr, initial_weights='imagenet', **kwargs):
        super(SegModelLightning, self).__init__()

        self.model_prod = SegModel(in_ch, num_classes, encoder, initial_weights)

        self.save_hyperparameters()

        self.criterion = smp.utils.losses.DiceLoss()

        self.learning_rate = lr
        self.num_classes = num_classes
        self.channels = in_ch

    def forward(self, x) -> torch.Tensor:
        return self.model_prod.forward(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        outputs = self.model_prod(inputs)

        # Compute loss and iou for each class and average them
        loss = self.criterion(outputs, labels)

        # Log IOU and loss
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch

        # Masks out uncertain pixels. No loss will be computed for them
        outputs = self.model_prod(inputs)

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
        return torch.optim.Adam([dict(params=self.model_prod.parameters(), lr=self.learning_rate)])

    def get_preprocessing(self):
        """Generates preprocessing transform

        Returns:
            A.BasicTransform: The preprocessing transform
        """

        _transform = [
            A.Lambda(image=self.model_prod.normalize_image),
            A.Lambda(image=self.model_prod.fix_shape, mask=self.model_prod.fix_shape),
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
