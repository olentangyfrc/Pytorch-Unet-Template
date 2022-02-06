try:
    from .model import SegModel
except:
    from model import SegModel
from glob import glob
import numpy as np
import torch
import os


class Predictor():

    def __init__(self, weights="best", device=None):
        if device is None:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
            else:
                device = 'cpu'

        self.device = device
        file_path = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = glob(file_path + "/weights/%s.ckpt" % weights)[0]

        self.model: SegModel = SegModel.load_from_checkpoint(checkpoint_path, initial_weights=None).eval().to(device)

    def predict(self, img):
        preprocessed = self.model.get_preprocessing()(image=img)['image']

        with torch.no_grad():
            with torch.cuda.amp.autocast_mode.autocast(enabled=self.device != "cpu"):
                input_tensor = torch.from_numpy(
                    preprocessed[np.newaxis, :]).to(self.device)

                prediction = self.model.forward(input_tensor)

                return prediction.detach().moveaxis(1, -1).cpu().numpy()[0]
