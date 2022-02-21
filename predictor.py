try:
    from .model_prod import SegModel
except:
    from model_prod import SegModel
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

        self.device = torch.device(device)
        file_path = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = glob(file_path + "/weights/%s.ckpt" % weights)[0]

        self.model: SegModel = SegModel.load_from_checkpoint(checkpoint_path, initial_weights=None).eval().to(device)

    def predict(self, img):
        normalized = self.model.normalize_image(img)
        input_img = self.model.fix_shape(normalized)

        with torch.no_grad():
            with torch.cuda.amp.autocast_mode.autocast(enabled=self.device != "cpu"):
                input_tensor = torch.from_numpy(
                    input_img[np.newaxis, :]).to(self.device)

                prediction = self.model.forward(input_tensor)

                return prediction.detach().moveaxis(1, -1).cpu().numpy()[0]

if __name__ == "__main__":
    import cv2

    cap = cv2.VideoCapture(0)
    predictor = Predictor("test")

    while(True):
        ret, frame = cap.read()

        if ret:
            predictor.predict(frame)

            cv2.imshow(" ", frame)
            cv2.waitKey(1)

