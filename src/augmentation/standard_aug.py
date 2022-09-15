import csv
import random

import cv2
import numpy as np

import os

# from tensorflow.keras.preprocessing.image import load_img
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.preprocessing.image import ImageDataGenerator


from augmentations import augment_image

def load_data(paths, undersamples = [], set = 'train'):
    """For given list of paths to csv files, load images, labels and bboxes.

    Args:
        paths ([type]): [description]
        undersamples (list, optional): [description]. Defaults to [].

    Returns:
        [type]: [description]
    """
    data = {}
    i = 0
    for csv_path in paths:
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader, None)
            for row in csv_reader:
                # Check 
                filename, label, startX, startY, w, h = row
                # Skip validation set
                if 'valid' in filename:
                    continue
                if label not in data:
                    data[label] = []
                
                data[label].append((filename, np.array([int(startX), int(startY), int(w), int(h)]), ))

    return data


data  = load_data(['data/data.csv'])


augmented_folder = 'data/augmented_standard/'
max_n = 0
for label in data.keys():
# Creaing folders with appropiate names
    if not os.path.exists(augmented_folder+str(label)):
        os.makedirs(augmented_folder+str(label))
    # calculate how many unique images for laber there is
    _n = len(data[label])
    if _n > max_n:
        max_n = _n


# Note: we need to create set number of images for each label
#       those numbers are based on highest amount of images that exist to keep 
#       balance of the set
create_max = max_n * 2
create_count = {}
for label in data.keys():
    create_count[label] = create_max - len(data[label])

print(create_count)

# Open csv for appending synthetic images
csv_fw = open('data/augmented_standard.csv','w')
csv_fw.write('im_path,label,x,y,w,h\n')

for label, im_bbox in data.items():
    # shuffle im_bbox
    print("creating synthetic images for label: ", label)
    random.shuffle(im_bbox)
    for i in range(create_count[label]):
        # Get random image from label
        filename, bbox = random.choice(im_bbox)
        # Load image
        img = cv2.imread(filename)
        # Augment image
        img, bbox = augment_image(img, bbox)

        # draw bbox on image
        # cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
        # Save image
        _tmp_path = os.path.join(augmented_folder, str(label), str(i)+'.jpg')
        cv2.imwrite(_tmp_path, img)
        # Write to csv
        csv_fw.write(f'{_tmp_path},{label},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n')

        if i % 200 == 0:
            print(f"finised {i} images")

