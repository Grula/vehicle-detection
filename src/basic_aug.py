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
                labels.append(label)
                bboxes.append((startX, startY, endX, endY))
                # Just testing on my pc remove for colab
                # if i > 10:
                #     break
                # i += 1
    data = np.array(data, dtype=np.float64)# / 255.
    labels = np.array(labels)
    bboxes = np.array(bboxes, dtype=np.float64)


    return data, labels, bboxes

# Generator parametes
datagen = ImageDataGenerator(rotation_range=5,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=-0.01,
                            brightness_range=[1,2.5],
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
csv_data = open('data/data_augmented.csv','w')
csv_data.write('im_path,label,x,y,w,h\n')

# Content of csv
# filename, label, startX, startY, w, h = row
for data in zip(images, labels, bboxes):
    image, label, bbox = data

    # Create new images with image generator
    # To current bbox we will aply small amount of noise
    # Label stays the same

    # Get number of current images in folder to avoid overwriting
    count = int(len(os.listdir(augmented_folder+str(label))))
    
    for i in range(12):    
        aug_filename = augmented_folder+str(label)+'/'+str(label)+'_'+str(count+i)+'.jpg'
        
        # Image Augmentation (rotation, zoom, brightness)
        current_img  = datagen.random_transform(image)
        
        # Add noise to BBOXes
        noisy_bbox = bbox + np.random.normal(0, 0.02, bbox.shape)

        # Add noise to image 25% chnace
        if np.random.rand() > 0.75:
            current_img = add_noise(current_img)
          
        # Flip the image 25% chance
        if np.random.rand() > 0.75:
            current_img = cv2.flip(current_img, 1)
            # Flip coordinates
            noisy_bbox = 1 - noisy_bbox

        # Change colorspace 40% chance
        if np.random.rand() > 0.6:
            # Pick 2 numbers out of 0,1,2
            a, b = random.sample([0, 1, 2], 2)
            current_img[[a,b]] = current_img[[b,a]]

        # Add blur 50% chance
        if np.random.rand() > 0.5:
            dst = cv2.GaussianBlur(current_img, (3,3), cv2.BORDER_DEFAULT)

            
        # Change bbox by adding very small noise to it
        # Coorect bbox if its out of image bounds
        noisy_bbox[0] = min(1.,max(0., noisy_bbox[0]))
        noisy_bbox[1] = min(1.,max(0., noisy_bbox[1]))
        noisy_bbox[2] = min(1.,max(0., noisy_bbox[2]))
        noisy_bbox[3] = min(1.,max(0., noisy_bbox[3]))


        h, w = current_img.shape[:2]

        startX = int(noisy_bbox[0] * w)
        startY = int(noisy_bbox[1] * h)

        w = abs(startX - int(noisy_bbox[2] * w))
        h = abs(startY - int(noisy_bbox[3] * h))

        # cv2.rectangle(current_img, (startX, startY), (int(noisy_bbox[2]*w), int(noisy_bbox[3]*h)), (0, 255, 0), 2)
        cv2.imwrite(aug_filename, current_img)

        # Write to csv
        csv_data.write(aug_filename+','+str(label)+','+str(startX)+','+str(startY)+','+str(w)+','+str(h)+'\n')
    
    # Add noise to images
    for j in range(3):
        aug_filename = augmented_folder+str(label)+'/'+str(label)+'_'+str(count+i+j+1)+'.jpg'
        
        current_img = add_noise(image)

        # Change bbox by adding very small noise to it
        noisy_bbox = bbox + np.random.normal(0, 0.02, bbox.shape)
        # Coorect bbox if its out of image bounds
        noisy_bbox[0] = min(1.,max(0., noisy_bbox[0]))
        noisy_bbox[1] = min(1.,max(0., noisy_bbox[1]))
        noisy_bbox[2] = min(1.,max(0., noisy_bbox[2]))
        noisy_bbox[3] = min(1.,max(0., noisy_bbox[3]))

        h, w = current_img.shape[:2]

        startX = int(noisy_bbox[0] * w)
        startY = int(noisy_bbox[1] * h)

        w = abs(startX - int(noisy_bbox[2] * w))
        h = abs(startY - int(noisy_bbox[3] * h))

        # cv2.rectangle(current_img, (startX, startY), (int(noisy_bbox[2]*w), int(noisy_bbox[3]*h)), (0, 255, 0), 2)
        cv2.imwrite(aug_filename, current_img)

        csv_data.write(aug_filename+','+str(label)+','+str(startX)+','+str(startY)+','+str(w)+','+str(h)+'\n')










