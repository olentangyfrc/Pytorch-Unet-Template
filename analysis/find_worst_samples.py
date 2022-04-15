# This script will look through all samples and find the ones with the worst performance

import sys
sys.path.append(".")

from predictor import Predictor
import numpy as np
from tqdm import tqdm
from glob import glob
import os
import cv2

CONF_THRESH = 0.9

predictor = Predictor(weights="test")
ious = {}

for annot_file in tqdm(glob("data/training_data/*annot/*.npy")):
    basename = os.path.basename(annot_file)
    annot_folder = os.path.dirname(annot_file)
    input_folder = annot_folder[:-5]

    # Load in our training data and resize it
    input = np.load(os.path.join(input_folder, basename))
    label = np.load(annot_file)
    input = cv2.resize(input, (640, 480))
    label = cv2.resize(label, (640, 480))

    # Conpute region that needs to be ignored
    ignore_mask = label > 100

    # Run our model
    predicted = predictor.predict(input, CONF_THRESH)

    # Apply ignore mask by hard coding those pixels with 0 error
    predicted[ignore_mask] = False
    label[ignore_mask] = 0

    # Compute performance
    label = label > 0.5
    intersection = np.sum(predicted * label)
    union = np.sum(predicted) + np.sum(label) - intersection
    ious[basename] = (intersection + 0.01) / (union + 0.01)

# Print our results
for name, iou in sorted(ious.items(), key=lambda x:x[1]):
    print("%s: %f" % (name, iou))