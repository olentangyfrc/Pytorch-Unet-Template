import sys
sys.path.append(".")

from predictor import Predictor
import numpy as np
from tqdm import tqdm
from glob import glob
import os
import cv2

predictor = Predictor(weights="test")
ious = {}
os.makedirs("results", exist_ok=True)

for annot_file in tqdm(glob("data/training_data/*annot/*.npy")):
    basename = os.path.basename(annot_file)
    annot_folder = os.path.dirname(annot_file)
    input_folder = annot_folder[:-5]

    input = np.load(os.path.join(input_folder, basename))
    label = np.load(annot_file)

    predicted = predictor.predict(input) > 0.5

    image = input.astype(float)
    image[:,:,1:] += predicted * 255
    image = np.clip(image, 0, 255)

    cv2.imwrite("results/" + basename + ".png", image.astype('uint8'))

