# Script that contains all code relating to loading data

import numpy as np
import json
import cv2
import glob


# Load in an image
def load_img(path: str, scale, channels):
    with open(path) as f:
        data = json.load(f)

    img_files = glob.glob(path.replace("json", "*"))
    img_path = [
        file_path for file_path in img_files if "json" not in file_path][0]
    h = int(data['imageHeight'] * scale)
    w = int(data['imageWidth'] * scale)
    if channels == 1:
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    elif channels == 3:
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (w, h))

    return image


def load_labelme_data(path, scale, classes):
    with open(path) as f:
        data = json.load(f)

        h, w = int(data["imageHeight"] * scale), int(data["imageWidth"] * scale)

        output = np.zeros((len(classes), h, w), 'uint8')

        def draw_shape(img, shape, class_indicies, color):
            """
            Draw a labelme shape on an image.
            """
            points = np.array(np.array(shape['points']) * scale, int)
            for class_idx in class_indicies:
                if shape['shape_type'] == 'rectangle':
                    img[class_idx] = cv2.rectangle(
                        img[class_idx], points[0], points[1], color, -1)
                elif shape['shape_type'] == 'circle':
                    radius = int(np.linalg.norm(points[1] - points[0]))
                    img[class_idx] = cv2.circle(
                        img[class_idx], points[0], radius, color, -1)
                elif shape['shape_type'] == 'polygon':
                    img[class_idx] = cv2.fillPoly(
                        img[class_idx], [points], color)
                elif shape['shape_type'] == 'line':
                    img[class_idx] = cv2.line(
                        img[class_idx], points[0], points[1], color, int((h + w) / 300))
                else:
                    raise(Exception("Unknown shape type"))

        # Now do everything else
        for shape in data['shapes']:
            class_indicies = []
            for idx, class_name in enumerate(classes):
                if (class_name.lower()+'/') in (shape['label'].lower()+'/'):
                    class_indicies.append(idx)

            draw_shape(output, shape, class_indicies, 1)

        return np.moveaxis(output, 0, -1)
