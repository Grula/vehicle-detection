# TODO
# We want to open csv of real data 
# find the min and max of each column (x, y, w, h)
# based on that knoweledge create a new bboxes for augmented images in that range
# save the new bboxes in a new csv

# open data.csv

import os
import csv
import random

import cv2

f = open('data/data.csv', 'r')

csvfr = csv.reader(f, delimiter=',')
next(csvfr, None) # skip header

# values will be tuple (min, max)
true_mm = { 'bus'   : {'x': (None, None), 'y': (None, None), 'w': (None, None), 'h': (None, None)},
            'truck' : {'x': (None, None), 'y': (None, None), 'w': (None, None), 'h': (None, None)},
        }

height, width = None, None
for row in csvfr:
    # im_path,label,x,y,w,h
    filename, label, x, y, w, h = row
    if 'train' not in filename:
        continue
    if label not in ('bus', 'truck'):
        continue

    if height is None:
        print(filename)
        image = cv2.imread(filename)
        height, width = image.shape[:2]

    # save relative values between 0 and 1
    x = float(x) / width
    y = float(y) / height
    w = float(w) / width
    h = float(h) / height

    
    if true_mm[label]['x'][0] is None:
        true_mm[label]['x'] = (x, x)
    else:
        true_mm[label]['x'] = (min(true_mm[label]['x'][0], x), max(true_mm[label]['x'][1], x))

    if true_mm[label]['y'][0] is None:
        true_mm[label]['y'] = (y, y)
    else:
        true_mm[label]['y'] = (min(true_mm[label]['y'][0], y), max(true_mm[label]['y'][1], y))


    if true_mm[label]['w'][0] is None:
        true_mm[label]['w'] = (w, w)
    else:
        true_mm[label]['w'] = (min(true_mm[label]['w'][0], w), max(true_mm[label]['w'][1], w))

    if true_mm[label]['h'][0] is None:
        true_mm[label]['h'] = (h, h)
    else:
        true_mm[label]['h'] = (min(true_mm[label]['h'][0], h), max(true_mm[label]['h'][1], h))

print(true_mm)
# create new csv
f = open('data/data_augmented_bt.csv', 'w') # bt = bus and truck
f.write('im_path,label,x,y,w,h\n')


# traverse folder in data/augmented
subfolders = [ f.path for f in os.scandir('data/augmented_stylegan') if f.is_dir() ]

# traverse subfolders and get images_path
for subfolder in subfolders:
        images = [f for f in os.listdir(subfolder) if f.endswith('.png') or f.endswith('.jpg')]
        label = subfolder.split('/')[-1]
        coords = true_mm[label]

        # load image and get width and height
        image = cv2.imread(os.path.join(subfolder, images[0]))
        height, width = image.shape[:2]


        for image in images:
            true_path = os.path.join(subfolder, image)

            # bbox ranges
            # pick x randomly between coords[x] 
            x = random.uniform(coords['x'][0], coords['x'][1])
            # pick y randomly between coords[y]
            y = random.uniform(coords['y'][0], coords['y'][1])
            # pick w randomly between coords[w]
            w = random.uniform(coords['w'][0], coords['w'][1])
            # pick h randomly between coords[h]
            h = random.uniform(coords['h'][0], coords['h'][1])

            # transform to absolute values
            x = int(x * width)
            y = int(y * height)
            w = int(w * width)
            h = int(h * height)


            # load image and draw coords on it
            # img = cv2.imread(true_path)
            # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # # save image to different folder
            # new_path = os.path.join('data/draw', label, image)
            # cv2.imwrite(new_path, img)

            # write to csv
            f.write(f'{true_path},{label},{x},{y},{w},{h}\n')
            
f.close()
