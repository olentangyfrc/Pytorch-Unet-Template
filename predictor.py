""" All the code to handle the use of a model to predict samples """

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

        # Determine which device to use
        if device is None:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
            else:
                device = 'cpu'

        self.device = torch.device(device)
        file_path = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = glob(file_path + "/weights/%s.ckpt" % weights)[0]

        # Load model
        self.model: SegModel = SegModel.load_from_checkpoint(checkpoint_path, initial_weights=None).eval().to(device)

    def predict(self, img: np.ndarray, conf_thresh=None) -> np.ndarray:
        assert img.shape == (480,640,3), "Image is the wrong size. Expected a 640 x 480 image size"

        # Make image match the input used at training time
        normalized = self.model.normalize_image(img)
        input_img = self.model.fix_shape(normalized)

        with torch.no_grad():
            with torch.cuda.amp.autocast_mode.autocast(enabled=self.device != "cpu"):
                # Load image into tensor and send to device
                input_tensor = torch.from_numpy(
                    input_img[np.newaxis, :]).to(self.device)

                # Run the model and convert the result into a numpy array
                prediction = self.model(input_tensor).detach().moveaxis(1, -1).cpu().numpy()[0]
                
                # Apply confidence threshold if provided
                if conf_thresh:
                    prediction = prediction > conf_thresh

                return prediction

if __name__ == "__main__":
    import cv2

    cap = cv2.VideoCapture(0)
    predictor = Predictor("test")
    cv2.namedWindow(" ", cv2.WINDOW_NORMAL)

    while(True):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))

        if ret:
            prediction = predictor.predict(frame, 0.9)

            annotated = frame.astype(float)
            annotated[:,:,1:] += prediction * 255
            annotated = np.clip(annotated, 0, 255)

            cv2.imshow(" ", annotated.astype('uint8'))
            cv2.waitKey(1)

