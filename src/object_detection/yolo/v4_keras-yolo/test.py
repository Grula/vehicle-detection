import os

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

if __name__ == '__main__':

    model_path = 'model_data/custom_trained_weights.h5'
    classes_path = 'model_data/custom_classes.txt'
    model_image_size = (416, 416)
    
    # model_path = 'model_data/yolo4_weight.h5'
    # classes_path = 'model_data/coco_classes.txt'
    # model_image_size = (608, 608)
    
    
    anchors_path = 'model_data/yolo4_anchors.txt'

    class_names = get_class(classes_path)
    anchors = get_anchors(anchors_path)

    num_anchors = len(anchors)
    num_classes = len(class_names)


    conf_thresh = 0.8
    nms_thresh = 0.7

    yolo4_model = yolo4_body(Input(shape=model_image_size+(3,)), num_anchors//3, num_classes)

    model_path = os.path.expanduser(model_path)
    assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

    yolo4_model.load_weights(model_path)

    _decode = Decode(conf_thresh, nms_thresh, model_image_size, yolo4_model, class_names)

    images_path = 'images'
    # detect images in test floder.
    for (root, dirs, files) in os.walk(images_path):
        if files:
            for f in files:
                path = os.path.join(root, f)
                image = cv2.imread(path)
                image, boxes, scores, classes = _decode.detect_image(image, True)
                print("Detecing image: {}".format(path))
                cv2.imwrite('pred_images/' + f, image)