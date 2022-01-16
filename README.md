# PyTorch U-Net Pipeline

A pipeline for labeling data and training a U-net

## Pre-Requirements

Install python. Make sure you check the "Add to PATH" checkbox during install for ease of use.

## Installation

1. Install python packages
    ```sh
    python -m pip install -r requirements.txt
    ```
2. That's it!


## Usage

1. Run start_labelme.bat and label your samples. The __ignore__ class is used to exclude a region from training.
2. Change the parameters at the top of datamodule.py. Mainly CHANNELS
3. Change the augmentations in datamodule.py if you desire.
4. List the classes you want to train on in train.py. You can start with one to test it.
5. View the tensorboard dashboard to monitor training.
6. Select your best model, copy its checkpoint from logs into weights, and use predictor.py to predict for your project.