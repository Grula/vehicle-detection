# Idea is 
# GEnerate image of bus/truck
# Display with cv2 and register mouse click
# Save those cooridnates as they wiull be our reference


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# add path to the system modules
from path import Path

import cv2
import numpy as np
import csv


from tensorflow.keras.preprocessing.image import ImageDataGenerator

def callback_select(event, x, y, flags, param):
        # Ctrl + left click
    if event == cv2.EVENT_LBUTTONUP and flags == (cv2.EVENT_FLAG_LBUTTON + cv2.EVENT_FLAG_CTRLKEY):
        # print("Ctrl + left click")
        if not param['click']:
            state_image = param['img'][-1].copy()
            param['click'] = True
            param['fields'].append({'coords':[(x, y)]})
            cv2.circle(state_image, (x, y), 3, (0, 0, 255), -1)
            param['img'].append(state_image)
            
        else:
            param['click'] = False
            param['fields'][-1]['coords'].append((x, y))
            cv2.circle(param['img'][-1], (x, y), 3, (0, 0, 255), -1)
            cv2.rectangle(param['img'][-1], param['fields'][-1]['coords'][0], (x, y), (0, 255, 0), 2)

            

    if param['click']:
        # If its true we will draw a rectangle on current mouse position
        # print("Awaintg second click")
        tmp_image = param['img'][-1].copy()
        cv2.rectangle(tmp_image, param['fields'][-1]['coords'][0], (x, y), (0, 255, 0), 2)
        cv2.imshow(param['window_name'], tmp_image)


    return None
   
def load_data(csv_path, target = "bus"):
    images = []
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        for row in csv_reader:
            filename, label, startX, startY, w, h = row
            if 'valid' in filename:
                continue
            if label != target : continue
            image = cv2.imread(filename)
            images.append(image)
    return images


image_generator = ImageDataGenerator(rotation_range=7,
                                    width_shift_range=0.2,
                                    height_shift_range=0.1,
                                    zoom_range=-0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest',
                                    # fill_mode='wrap',
                                    # fill_mode='reflect',
                                    )

def generate(images, data_folder, label, image_generate_size = 10):
    csv_out = []
    # get number of images in data_folder/label folder
    data_in_folder = len(os.listdir(os.path.join(data_folder, label)))
    for i in range(image_generate_size):
        j = i % len(images)
        
        image = images[j]
    # for imageNo, image in enumerate(images):
        # generate image with keras
        gen_image  = image_generator.random_transform(image)
        # Then send this image to display to capture coordinates
        param = {'img':[gen_image], 'click':False, 'fields': [], 'imshow': None, 'window_name' : 'generator' }
        
        cv2.namedWindow(param['window_name'], cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(param['window_name'], callback_select, param)

        while(True):
            cv2.imshow(param['window_name'], param['img'][-1])
            k = cv2.waitKeyEx(0) & 0xFF
            if k == ord('q'):
                # os._exit(0)
                pass
            if k == ord('e'):
                # Pop last gen_image and fields
                if len(param['img']) > 1:
                    param['img'].pop()
                    param['fields'].pop()
            if k == ord('n'):
                break

        path = data_folder / label / f"{label}_{data_in_folder+i}.jpg"
        startX = param['fields'][-1]['coords'][0][0]
        startY = param['fields'][-1]['coords'][0][1]
        w = param['fields'][-1]['coords'][1][0] - startX
        h = param['fields'][-1]['coords'][1][1] - startY

        csv_out.append([path, label, startX, startY, w, h])
        # Save gen_image on dat path
        cv2.imwrite(str(path), gen_image)


    return csv_out

        

# Bus and Truck
data_folder = Path("data/synthetic/")
if __name__ == '__main__':
    label = "truck"
    # Check if folder data_folder exists if not create it
    images = load_data('data/data.csv', label)
    if not (data_folder/label).exists():
        (data_folder/label).mkdir()
    print(len(images))
    out = generate(images,data_folder, label, 17)

    # Write csv file to synthetic
    with open(data_folder / 'data.csv', 'a') as csv_file:
        writer = csv.writer(csv_file)
        for row in out:
            writer.writerow(row)