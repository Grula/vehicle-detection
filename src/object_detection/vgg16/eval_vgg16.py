import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pickle

import numpy as np
import cv2

from sklearn.preprocessing import LabelBinarizer

from tensorflow import keras
from keras.utils import img_to_array

import argparse


def load_model(model_path):
    model = keras.models.load_model(model_path)
    return model

def eval_model(model, image):
    image = preproccess_image(image)
    pred = model.predict(image)
    # label, bbox predicitons
    return pred[0][0], pred[1][0]

def preproccess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image,axis=0)
    return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--model', type=str , default='model.h5', help='Name of model')
    parser.add_argument('--valid', default='', type=str, required=True , help='Path to validation data')
    args = vars(parser.parse_args())

    model = load_model(args['model'])


    # Open LabelBinarizer from pickle file
    # with open('lb.pickle', 'rb') as f:
    #     lb = pickle.load(f)

    validation_images = args['valid']
    label_ids = {0 : 'car', 1 : 'motorbike', 2 : 'bus', 3 : 'truck'}

    # find txt file in validation data
    for file in os.listdir(validation_images):
        if file.endswith('.txt'):
            valid_paths = os.path.join(validation_images, file)
            break
    
    # read txt file
    with open(valid_paths) as f:
        lines = f.readlines()

    # create csv file
    f = open(f'VGG16-data_prediction.csv', 'w')
    # write header
    # file_path,true_label, predicted_label, confidence, [true_bboxes], [predicted_bboxes] 
    f.write('file_path, true_label, predicted_label, confidence, true_bboxes, predicted_bboxes\n')

    for line in lines:
        path_to_image, coords = line.strip().split(' ')
        path_to_image = os.path.join(validation_images, path_to_image)
        
        # x, y, w, h, id
        _data = list(map(int, coords.split(',')) ) # last one is class id
        _data = coords.split(',') # last one is class id
        true_bbox, _id = _data[:-1], int(_data[-1])
        true_label = label_ids[_id]


        print("Detecing image: {}".format(path_to_image))
        # load image
        image = cv2.imread(path_to_image)

        # Predict image
        predicted_label, pred_bbox = eval_model(model, image)
        
        confidence = max(predicted_label)
        
        # find index of max value
        predicted_label = np.argmax(predicted_label)
        predicted_label = label_ids[predicted_label]

        # max value in predcted_label is confidence
        # predicted bbox
        x0, y0, x1, y1 = pred_bbox
        x0 = int(x0 * image.shape[1])
        y0 = int(y0 * image.shape[0])
        x1 = int(x1 * image.shape[1])
        y1 = int(y1 * image.shape[0])

        w = x1 - x0
        h = y1 - y0

        predicted_bbox = [str(x0), str(y0), str(w), str(h)]

        # write to csv file
        # f.write(f'{path_to_image},{true_label},{predicted_label},{pred_bbox},{true_bbox},{predicted_bbox}\n')

        f.write('{}, {}, {}, {}, {}, {}\n'.format(path_to_image, true_label, predicted_label, \
                 confidence, ' '.join(true_bbox), ' '.join(predicted_bbox))
                )

        
    f.close()

            


        

        








