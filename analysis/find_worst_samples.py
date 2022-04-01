import sys
from typing import OrderedDict
sys.path.append(".")

from predictor import Predictor
import numpy as np
from tqdm import tqdm
from glob import glob
import os

predictor = Predictor(weights="test")
ious = {}

for annot_file in tqdm(glob("data/training_data/*annot/*.npy")):
    basename = os.path.basename(annot_file)
    annot_folder = os.path.dirname(annot_file)
    input_folder = annot_folder[:-5]

    input = np.load(os.path.join(input_folder, basename))
    label = np.load(annot_file)

    predicted = predictor.predict(input) > 0.5

    intersection = np.sum(predicted * label)
    union = np.sum(predicted) + np.sum(label) - intersection

    ious[basename] = (intersection + 0.01) / (union + 0.01)


for name, iou in sorted(ious.items(), key=lambda x:x[1]):
    print("%s: %f" % (name, iou))