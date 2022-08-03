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

# def get_relative_path(path):
#     return os.path.join(sys.path[0], path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--model_data', type=str , default='model_data/', help='path to model data')
    parser.add_argument('--weights_name', type=str , default='512_yolo4_weights.h5', help='name of model weights')
    parser.add_argument('--image_dim', default=512, type=int)
    parser.add_argument('--images', default='', type=str, required=True)
    
    # parser.add_argument('--model_dir', default='model_data/', type=str)
    # parser.add_argument('--weights_path', default='', type=str)
    # parser.add_argument('--classes_path', default='', type=str)
    parser.add_argument('--save', default=0, type=bool)

    args = vars(parser.parse_args())

    model_dir = args['model_data']
    print(model_dir)
    images_path = args['images']

    weights_path = os.path.join(args['model_data'],args['weights_name'])
    classes_path = os.path.join(args['model_data'], 'custom_classes.txt')
    anchors_path = os.path.join(args['model_data'], 'yolo4_anchors.txt')


    # model_image_size = (416, 416)
    # model_image_size = (608, 608)
    model_image_size = (args['image_dim'], args['image_dim'])
    
    class_names = get_class(classes_path)
    anchors = get_anchors(anchors_path)

    num_anchors = len(anchors)
    num_classes = len(class_names)

    # conf_thresh = 0.8
    # nms_thresh = 0.7
    conf_thresh = 0.1
    nms_thresh = 0.45

    yolo4_model = yolo4_body(Input(shape=model_image_size+(3,)), num_anchors//3, num_classes)

    model_path = os.path.expanduser(weights_path)
    print(model_path)
    assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

    yolo4_model.load_weights(model_path)
    _decode = Decode(conf_thresh, nms_thresh, model_image_size, yolo4_model, class_names)

    # create csv file
    f = open('data_prediction.csv', 'w')
    # im_path, label, x, y, w, h
    # f.write('im_path,label,x,y,w,h\n')
    f.write('detected_label, score\n')



    # detect images in test floder
    subfolders = [ f.path for f in os.scandir(images_path) if f.is_dir() ]
    for subfolder in subfolders:
        images = [f for f in os.listdir(subfolder) if f.endswith('.png') or f.endswith('.jpg')]
        current_label = subfolder.split('/')[-1]

        print(subfolder)
        if 'PREDICTED' in subfolder:
            continue

        destination_folder = os.path.join(images_path,(f'{current_label}-PREDICTED'))
        print("Destination folder ----> ", destination_folder)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        
        for image_file in images:
            image_path = os.path.join(subfolder, image_file)

            print("Detecing image: {}".format(image_path))
            image = cv2.imread(image_path)
            image, boxes, scores, classes = _decode.detect_image(image, True)
            if boxes is  None:
                f.write(f'{current_label}:{False},{0}\n')
                continue
            predicted_data = list(zip(boxes, scores, classes))
            predicted_data.sort(key=lambda x: x[1], reverse=True)
            
            # find image with highest score if exists
            detected = False
            max_score = 0
            for box, score, cl in predicted_data:
                # Check if class is in the list of classes to detect
                if class_names[cl] != current_label:
                    continue
                detected = True
                if max_score < score:
                    max_score = score
                # max_idx = 0
                # for i, box in enumerate(boxes):
                #     if scores[i] > max_score:
                #         max_score = scores[i]
                #         max_idx = i

                # x0, y0, x1, y1 = boxes[max_idx]
                x0, y0, x1, y1 = box
                x = max(0, np.floor(x0 + 0.5).astype(int))
                y = max(0, np.floor(y0 + 0.5).astype(int))
                right = min(image.shape[1], np.floor(x1 + 0.5).astype(int))
                bottom = min(image.shape[0], np.floor(y1 + 0.5).astype(int))
                w = right - x
                h = bottom - y
                # saving info to csv file 

                # f.write(f'{image_path},{current_label},{class_names[classes[i]]},{max_score},{x},{y},{w},{h}\n')
            
            f.write(f'{current_label}:{detected},{max_score}\n')

            if args['save']:
                cv2.imwrite(f'{destination_folder}/{image_file}', image)
                print(f'Saved {image_file}')
    
    
    f.close()