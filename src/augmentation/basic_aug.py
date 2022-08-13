import csv
import random

import cv2
import numpy as np

import os

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_data(paths, undersamples = [], set = 'train'):
    """For given list of paths to csv files, load images, labels and bboxes.

    Args:
        paths ([type]): [description]
        undersamples (list, optional): [description]. Defaults to [].

    Returns:
        [type]: [description]
    """
    data = []
    labels = []
    bboxes =  []
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

                
                # check if lavel is in undersample list
                # if label in undersamples:
                #     continue

                # endX = float(startX) + float(w)
                # endY = float(startY) + float(h)

                # image = cv2.imread(filename)
                # (h,w)=image.shape[:2]
                
                # startX = float(startX) / w
                # startY = float(startY) / h
                # endX = float(endX) / w
                # endY = float(endY) / h

                # image = load_img(filename)
                # image = img_to_array(image)
                
                data.append(filename)
                labels.append(label)
                bboxes.append((startX, startY, w, h))
                # Just testing on my pc remove for colab
                # if i > 10:
                #     break
                # i += 1
    data = np.array(data)
    labels = np.array(labels)
    bboxes = np.array(bboxes, dtype=np.float64)


    return data, labels, bboxes

# Generator parametes
datagen = ImageDataGenerator(rotation_range=5,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=-0.01,
                            brightness_range=[0.5,1.5],
                            fill_mode='nearest',
                            # fill_mode='wrap',
                            # fill_mode='reflect',
                            # horizontal_flip=True,
                            )

def add_noise(img,):
    img = img[...,::-1]/255.0
    noise =  np.random.normal(loc=0, scale=1, size=img.shape)
    img = np.clip((img + noise*0.1),0,1)*255.0
    return img

# Load  data then generate synthetic images applying basic augmentation + noise
# preserving bboxes postion on images ( will be considered as regulasitation)


# datas  = list(zip(load_data(['data/data.csv'])))
images, labels, bboxes  = load_data(['data/data.csv'])

# Creaing folders with appropiate names
augmented_folder = 'data/augmented/'
unique = np.unique(labels)
for i in unique:
    if not os.path.exists(augmented_folder+str(i)):
        os.makedirs(augmented_folder+str(i))

# Open csv for appending synthetic images
csv_data = open('data/augmented.csv','w')
csv_data.write('im_path,label,x,y,w,h\n')


# we need to create set number of images for each label
# those numbers are based on highest amount of images that exist to keep 
# balance of the set
count = {'motorbike' : 869,
         'car' : 937,
         'bus' : 1732,
         'truck' : 1729,}

# Create dictionary with labels being keys and values beeing imagse with bboxes

data_dict = {}
for i in range(len(labels)):
    if labels[i] not in data_dict:
        data_dict[labels[i]] = []
    data_dict[labels[i]].append((images[i], bboxes[i]))

for key in data_dict.keys():
    # select that one label and based on label create that many images
    img_count = count[key]
    img_idx = 0
    print("working on label: ", key)
    while img_count > 0:
        aug_path = augmented_folder+str(key)+'/'+str(img_idx)+'.png'
        
        # select random image from that label
        filename, bbox = random.choice(data_dict[key])

        # load image
        img = load_img(filename)
        img = img_to_array(img)
        w, h = img.shape[:2]

        # Image Augmentation (rotation, zoom, brightness)
        aug_img  = datagen.random_transform(img)
        
        # Add noise to image 25% chnace
        if np.random.rand() < 0.2:
            aug_img = add_noise(aug_img)
          
        # Flip the image 25% chance
        if np.random.rand() > 0.2:
            aug_img = cv2.flip(aug_img, 1)
            # Flip coordinates of bbox
            bbox[0] = w - bbox[0]            


        # Change colorspace 5% chance
        if np.random.rand() > 0.05:
            # Pick 2 numbers out of 0,1,2
            a, b = random.sample([0, 1, 2], 2)
            aug_img[[a,b]] = aug_img[[b,a]]

        # Add blur 15% chance
        if np.random.rand() < 0.2:
            dst = cv2.GaussianBlur(aug_img, (3,3), cv2.BORDER_DEFAULT)

            
        # cv2.rectangle(current_img, (startX, startY), (int(noisy_bbox[2]*w), int(noisy_bbox[3]*h)), (0, 255, 0), 2)
        cv2.imwrite(aug_path, aug_img)

        # Write to csv
        # csv_data.write(aug_path+','+str(key)+','+str(startX)+','+str(startY)+','+str(w)+','+str(h)+'\n')
        csv_data.write('{},{},{},{},{},{}\n'.format(aug_path, key, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
        # Updated index and count
        img_idx += 1
        img_count -= 1


csv_data.close()