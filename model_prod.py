import torch
from torch.nn import Module
import segmentation_models_pytorch as smp
import numpy as np


class SegModel(Module):
    def __init__(self, in_ch, num_classes, encoder, initial_weights='imagenet', **kwargs):
        super(SegModel, self).__init__()

        self.model = smp.FPN(
            in_channels=in_ch,
            encoder_name=encoder,
            encoder_weights=initial_weights,
            classes=num_classes
        )

        self.sigmoid = torch.nn.Sigmoid()

        self.channels = in_ch

    def forward(self, x, sigmoid=True) -> torch.Tensor:
        logits = self.model.forward(x)
        if sigmoid:
            return self.sigmoid(logits)
        return logits


    @staticmethod
    def normalize_image(image, **kwargs):
        mean = std = 60
        if image.shape[2] == 3:
            mean = [60, 60, 60]
            std = [60, 60, 60]

        return (image - mean) / std

    @staticmethod
    def fix_shape(x, **kwargs):
        if len(x.shape) == 2:
            x = x[..., np.newaxis]
        w_pad = (32 - x.shape[1]) % 32
        h_pad = (32 - x.shape[0]) % 32
        padded = np.pad(x, ((0, h_pad), (0, w_pad), (0, 0)), constant_values=-1)

        return padded.transpose(2, 0, 1).astype('float32')

    @staticmethod
    def load_from_checkpoint(path, map_location=None, initial_weights=None, **kwargs):
        checkpoint = torch.load(path, map_location=map_location)
        hyper_parameters = checkpoint["hyper_parameters"]
        hyper_parameters["initial_weights"] = initial_weights
 
        # if you want to restore any hyperparameters, you can pass them too
        model = SegModel(**hyper_parameters)

        model_weights = checkpoint["state_dict"]

        # update keys by dropping `model_prod.`
        for key in list(model_weights):
            if "model_prod." in key:
                model_weights[key.replace("model_prod.", "")] = model_weights.pop(key)
            else:
                del model_weights[key]


        model.load_state_dict(model_weights)

        return model
