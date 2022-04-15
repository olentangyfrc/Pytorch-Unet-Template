# Predicts every sample and labels mistakes.
# Green is a true positive: The pixel is classified correctly
# Red is a false positive: The model thought something was there but the label says there is nothing
# Blue is a false negative: The model thought nothing was there but the label says there is something

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
os.makedirs("results", exist_ok=True)

for annot_file in tqdm(glob("data/training_data/*annot/*.npy")):
    basename = os.path.basename(annot_file)
    annot_folder = os.path.dirname(annot_file)
    input_folder = annot_folder[:-5]

    # Load in our training data and resize it
    input = np.load(os.path.join(input_folder, basename))
    label = np.load(annot_file)
    input = cv2.resize(input, (640, 480))
    label = cv2.resize(label, (640, 480))

    # Run our model
    predicted = predictor.predict(input, CONF_THRESH)

    # Compute our mistakes while keeping the ignores in mind
    annotated = input.astype(float)
    tp = np.logical_and(predicted, label > 0) # If True and label is True or ignore
    fp = np.logical_and(predicted, label == 0) # If True and label is false
    fn = np.logical_and(~predicted, label == 1)     # If False and label is true

    # Draw these mistakes
    annotated[:,:,0] += np.sum(fn, -1) * 255
    annotated[:,:,1] += np.sum(tp, -1) * 255
    annotated[:,:,2] += np.sum(fp, -1) * 255
    annotated = np.clip(annotated, 0, 255)

    cv2.imwrite("results/" + basename + ".png", annotated.astype('uint8'))

