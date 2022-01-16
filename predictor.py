from model import SegModel
from glob import glob
import numpy as np
import torch
import cv2
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

        self.model: SegModel = SegModel.load_from_checkpoint(checkpoint_path).eval().to(device)

    def predict(self, img):
        preprocessed = self.model.get_preprocessing()(image=img)['image']

        with torch.no_grad():
            with torch.cuda.amp.autocast_mode.autocast(enabled=self.device != "cpu"):
                input_tensor = torch.from_numpy(
                    preprocessed[np.newaxis, :]).to(self.device)

                prediction = self.model.forward(input_tensor)

                return prediction.detach().moveaxis(1, -1).cpu().numpy()[0]


if __name__ == "__main__":
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture("C:\\Users\\blain\\Downloads\\programming_room.mov")
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    predictor = Predictor(weights="best")

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            frame = cv2.resize(frame, None, fx=0.25, fy=0.25)

            predicted = predictor.predict(frame)

            annotated = frame.astype(float)
            annotated[:, :, 0] -= predicted[:annotated.shape[0],
                                            :annotated.shape[1], 0] * 255
            annotated[:, :, 1] -= predicted[:annotated.shape[0],
                                            :annotated.shape[1], 0] * 255
            annotated[:, :, 2] -= predicted[:annotated.shape[0],
                                            :annotated.shape[1], 0] * 255
            annotated = np.clip(annotated, 0, 255)

            # Display the resulting frame
            cv2.imshow('Frame', annotated.astype('uint8'))

            # Press Q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
