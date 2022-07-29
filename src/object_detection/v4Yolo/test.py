import os
import argparse
import sys

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo4.model import yolo_eval, yolo4_body
from yolo4.utils import letterbox_image

from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer
import cv2

from decode_np import Decode


def get_class(classes_path):
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    anchors_path = os.path.expanduser(anchors_path)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def get_relative_path(path):
    return os.path.join(sys.path[0], path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--model_dir', default='model_data/', type=str)
    parser.add_argument('--weights_path', default='', type=str)
    parser.add_argument('--classes_path', default='', type=str)
    parser.add_argument('--image_dim', default=416, type=int)
    parser.add_argument('--images', default='', type=str, required=True)
    parser.add_argument('--save', default=0, type=bool)

    args = vars(parser.parse_args())

    model_dir = get_relative_path(args['model_dir'])
    print(model_dir)
    images_path = args['images']

    # Give an option to choose the model if args is empty
    # model_path = 'model_data/yolo4_weight.h5'
    weights = args['weights_path']
    if not weights:
        # list all weights files that end with .h5
        weights_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
        for i, weights in enumerate(weights_files):
            print(f'\t{i}: {weights}')
        while True:
            try:
                idx = int(input('Select model: '))
                weights = weights_files[idx]
                break
            except Exception as e:
                print('Invalid model index')
                continue
    weights_path = os.path.join(get_relative_path(model_dir), weights)
    # model_path = 'model_data/custom_trained_weights.h5'

    # Give an option to choose the model if args is empty
    # classes_path = 'model_data/custom_classes.txt'
    # classes_path = 'model_data/coco_classes.txt'
    classes = args['classes_path']
    if not classes:
        # list all classes files that have class in name
        classes_files = [f for f in os.listdir(model_dir) if 'class' in f]
        for i, cls in enumerate(classes_files):
            print(f'\t{i}: {cls}')
        while True:
            try:
                idx = int(input('Select class file: '))
                classes = classes_files[idx]
                break
            except Exception as e:
                print('Invalid model index')
                continue

    classes_path = os.path.join(model_dir, classes)

    # model_image_size = (416, 416)
    # model_image_size = (608, 608)
    model_image_size = (args['image_dim'], args['image_dim'])
    
    anchors_path = get_relative_path('model_data/yolo4_anchors.txt')

    class_names = get_class(classes_path)
    anchors = get_anchors(anchors_path)

    num_anchors = len(anchors)
    num_classes = len(class_names)


    # conf_thresh = 0.8
    # nms_thresh = 0.7
    conf_thresh = 0.2
    nms_thresh = 0.45

    yolo4_model = yolo4_body(Input(shape=model_image_size+(3,)), num_anchors//3, num_classes)

    model_path = os.path.expanduser(weights_path)
    print(model_path)
    assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

    yolo4_model.load_weights(model_path)
    _decode = Decode(conf_thresh, nms_thresh, model_image_size, yolo4_model, class_names)


    # create csv file
    f = open('data_augmented.csv', 'w')
    # im_path, label, x, y, w, h
    f.write('im_path,label,x,y,w,h\n')



    # detect images in test floder
    subfolders = [ f.path for f in os.scandir(images_path) if f.is_dir() ]
    print(subfolders)
    for subfolder in subfolders:
        images = [f for f in os.listdir(subfolder) if f.endswith('.png') or f.endswith('.jpg')]
        current_label = subfolder.split('/')[-1]
        destination_folder = get_relative_path(f'{subfolder}-PREDICTED')
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        for image_file in images:
            images_path = os.path.join(subfolder, image_file)
            image = cv2.imread(images_path)

            print("Detecing image: {}".format(images_path))
            image, boxes, scores, classes = _decode.detect_image(image, True)
            
            # find image with highest score if exists
           
            if args['save']:
                if scores is not None and len(scores) > 0:
                    max_score = 0
                    max_idx = 0
                    for i, box in enumerate(boxes):
                        if scores[i] > max_score:
                            max_score = scores[i]
                            max_idx = i

                    x0, y0, x1, y1 = boxes[max_idx]
                    x = max(0, np.floor(x0 + 0.5).astype(int))
                    y = max(0, np.floor(y0 + 0.5).astype(int))
                    right = min(image.shape[1], np.floor(x1 + 0.5).astype(int))
                    bottom = min(image.shape[0], np.floor(y1 + 0.5).astype(int))
                    w = right - x
                    h = bottom - y
                    # saving info to csv file 
                    f.write(f'{images_path},{current_label},{x},{y},{w},{h}\n')

                cv2.imwrite(f'{destination_folder}/{image_file}', image)
                print(f'Saved {image_file}')
    
    
    f.close()