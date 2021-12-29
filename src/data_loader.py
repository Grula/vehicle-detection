import csv

import cv2
import numpy as np

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

def load_data(paths, undersamples = [], set = 'train'):
    """For given list of paths to csv files, load images, labels and bboxes.

    Args:
        paths ([type]): [description]
        undersamples (list, optional): [description]. Defaults to [].

    Returns:
        [type]: [description]
    """
    data = []
    classes = []
    bboxes =  []
    i = 0
    for csv_path in paths:
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader, None)
            for row in csv_reader:
                
                # Check 
                # Skip validation set
                if 'valid' in filename:
                    continue

                filename, label, startX, startY, w, h = row
                
                # check if lavel is in undersample list
                # if label in undersamples:
                #     continue

                endX = float(startX) + float(w)
                endY = float(startY) + float(h)

                image = cv2.imread(filename)
                (h,w)=image.shape[:2]
                
                startX = float(startX) / w
                startY = float(startY) / h
                endX = float(endX) / w
                endY = float(endY) / h

                image = load_img(filename, target_size=(224, 224))
                image = img_to_array(image)
                
                data.append(image)
                classes.append(label)
                bboxes.append((startX, startY, endX, endY))
                # Just testing on my pc remove for colab
                if i > 10:
                    # break
                    pass
                i += 1
    data = np.array(data, dtype="float32") / 255.
    classes = np.array(classes)
    bboxes = np.array(bboxes, dtype="float32")


    return data, classes, bboxes