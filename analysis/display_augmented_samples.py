# This file is use to display the augmented training samples

import sys
sys.path.append(".")

from datamodule import DataModule
import cv2
import numpy as np


# Where labelme_data is. Should be "data/" unless you move that folder to another drive
DATA_DIR = 'data/'

# List the classes from your dataset that you want to train on
CLASSES = [
    "left",
    "right"
]

if __name__ == "__main__":
    
    # Initialize data module and make sure npy files are generated
    datamodule = DataModule(
        data_dir=DATA_DIR,
        pin_memory=True,
        classes=CLASSES
    )
    datamodule.prepare_data()
    datamodule.setup()

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Annotated", cv2.WINDOW_NORMAL)

    # For each batch
    for batch in datamodule.train_dataloader():

        # For each sample in batch
        images, labels = batch
        for i in range(len(labels)):
            # Get image and label as numpy arrays
            image = images[i].numpy().astype('uint8')
            label = labels[i].numpy()

            # Generate an annotated image with labels
            annotated = image.copy()
            annotated[np.sum(label, axis=2) > 0] = 0
            annotated[:,:,1:] += labels[i].numpy() * 255

            # Display image
            cv2.imshow("Image", image)
            cv2.imshow("Annotated", annotated)
            cv2.waitKey()


