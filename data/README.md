# Data Labelling

The folder for data labeling.

## Installation

Everything should be good to go for Windows. If you are on mac, you need to install LabelMe from [this URL](https://github.com/wkentaro/labelme/releases) or through `pip install labelme`.

## Usage

1. Copy all of you images into the `data/labelme_data/` folder. Do not create any sub-directories within that folder.
2. List all the classes you want to identify in `labels.txt`
3. Double click `start_labelme.bat`. Should work on mac, let me know if it doesn't.
4. Click the label on the right to select the class you want 
5. Right click on the image and select `Create Circle` or whatever shape you want to make.
6. Everything auto-saves. Check the bottom right corner to see how many images you have left.
7. Once all the data is labeled, you can begin to train.

