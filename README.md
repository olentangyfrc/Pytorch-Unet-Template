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

1. Follow the `README.md` in the `data/` folder to prepare the data.
2. List the classes you want to train in `train.py`.
3. Set the `CHANNELS` in `datamodule.py` to color or grayscale.
4. Change the augmentations in `datamodule.py` if you desire.
5. Run `train.py` to start training. Will take a fair bit of time and will stop when it determines it is done.
6. View the TensorBoard dashboard to monitor training. Press F1 and type `TensorBoard` in VSCode.
7. Select your best model, copy its checkpoint from `logs/` into `weights/`, and use `predictor.py` to predict for your project.
8. If you need a smaller model, reference [this GitHub page](https://github.com/qubvel/segmentation_models.pytorch#encoders-) for smaller encoders and list your choice in the bottom of `model.py`.