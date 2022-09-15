import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np

import csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import pickle

import cv2

from tensorflow import keras
import PIL

from tensorflow.keras.applications.vgg16 import VGG16
# from keras.applications import VGG16
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping, TensorBoard

# from keras.utils import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
# from keras.utils import img_to_array

import argparse



def create_model(weights = None):
    # Load VGG16 model
    vgg=VGG16(weights=weights, include_top=False,input_tensor=Input(shape=(224,224,3)))
    flatten = vgg.output
    flatten = Flatten()(flatten)

    bboxhead = Dense(128,activation="relu")(flatten)
    bboxhead = Dense(64,activation="relu")(bboxhead)
    bboxhead = Dense(32,activation="relu")(bboxhead)
    bboxhead = Dense(4,activation="sigmoid", name = 'bounding_box')(bboxhead)


    classhead = Dense(128,activation="relu")(flatten)
    classhead = Dense(64,activation="relu")(classhead)
    classhead = Dense(32,activation="relu")(classhead)
    classhead = Dense(4,activation="softmax", name = 'class_label')(classhead)

    model = Model(inputs = vgg.input, outputs = [classhead, bboxhead])

    losses = {
        "class_label": "categorical_crossentropy",
        "bounding_box": "mse",
        }

    model.compile(optimizer="adam", loss=losses, metrics=["accuracy"])
    return model

def draw_bbox(image, bbox):
    startX, startY, endX, endY = bbox
    # Normilized coordinates, multyply by image width and height
    startX = int(startX * image.shape[1])
    startY = int(startY * image.shape[0])
    endX = int(endX * image.shape[1])
    endY = int(endY * image.shape[0])
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    return image

def eval_model(model, image):
    image = img_to_array(image)
    image = np.expand_dims(image,axis=0)
    car_pred = model.predict(image)
    # label and bbox predicitons
    return car_pred[0][0], car_pred[1][0]


class CustomDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, files, batch_size=32, input_size=(224,224,3), n_classes=4, shuffle=True, train = True):
        'Initialization'
        self.files = files
        self.batch_size = batch_size
        self.input_size = input_size

        # self.n_channels = n_channels
        # Number of label classes
        self.n_classes = n_classes
        self.shuffle = shuffle

        self.train = train
        
        self.classes_id = {0 : 'car', 1 : 'motorbike', 2 : 'bus', 3 : 'truck'}
        self.n = len(self.files)


        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.n // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        batches = self.files[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)        
        return X, y

    def __get_input(self, path,  target_size):
        
        image = load_img(path, target_size=target_size)
        image_arr = img_to_array(image)
        
        if self.train:
            # Implment data augmentation
            pass

        return image_arr/255.
    
    def __get_output_class(self, label, num_classes):
        return keras.utils.to_categorical(label, num_classes=num_classes)

    def __get_output_bbox(self, bbox, ):
        return bbox
    
    def __parse_data(self, files):
        paths = []
        labels = []
        bboxes = []
        for file in files:
            path, bb_l = file.split(' ')
            label_id = bb_l.split(',')[-1]

            # We have to load image and get its shape
            image = img_to_array(load_img(path))
            image_shape = image.shape
            bbox = bb_l.split(',')[:-1]
            x, y, w, h = [float(bbox[i]) for i in range(4)]
            bbox = [x, y, x+w, y+h]
            # transform bbox to normalized coordinates
            bbox = [bbox[0]/image_shape[1], bbox[1]/image_shape[0], bbox[2]/image_shape[1], bbox[3]/image_shape[0]]
            bbox = np.array(bbox)

            paths.append(path)
            labels.append(label_id)
            bboxes.append(bbox)
        return paths, labels, bboxes

    def __get_data(self, batches):
        # Generates data containing batch_size samples
        path_batch, label_batch, bbox_batch = self.__parse_data(batches)


        X_batch = np.asarray([self.__get_input(x, self.input_size) for x in path_batch])

        y0_batch = np.asarray([self.__get_output_class(y, self.n_classes) for y in label_batch])
        y1_batch = np.asarray([self.__get_output_bbox(y,) for y in bbox_batch])

        # input image, output [classhead and output bbox]
        return X_batch, tuple([y0_batch, y1_batch])

    def on_epoch_end(self):
        if self.shuffle:
            'shuffle list of files'
            np.random.shuffle(self.files)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--weights', type=str, default='imagenet', help='Use weights, default to None, use imagenet if not specified')
    parser.add_argument('--load', type=str, default=None, help='Specify to load model, provide path to model .h5 file')
    parser.add_argument('--train', type=bool, default=1, help='Specify to train model, default to True')

    args = parser.parse_args()
    lb = LabelBinarizer()
    


    if args.load is not None:
        assert args.load.endswith('.h5'), 'Model must be .h5 file'
        model = keras.models.load_model(args.load)
    else: # create model
        model = create_model(None if args.weights != 'imagenet' else args.weights)

    print("loaded/created model")

    train_list = 'data/train_data.txt'
    # valid_list = 'data/valid_data.txt'

    with open(train_list) as f:
        lines_train = f.readlines()
    np.random.seed(42)
    np.random.shuffle(lines_train)
    np.random.seed(None)

    lines_val = lines_train[:int(len(lines_train)*0.1)]
    lines_train = lines_train[int(len(lines_train)*0.1):]


    # with open(valid_list) as f:
        # lines_val = f.readlines()
    np.random.seed(42)
    np.random.shuffle(lines_val)
    np.random.seed(None)

    batch_size = 16
    traingen = CustomDataGenerator(lines_train, batch_size=batch_size, input_size=(224,224,3), n_classes=4, shuffle=True)
    validgen = CustomDataGenerator(lines_val, batch_size=batch_size, input_size=(224,224,3), n_classes=4, shuffle=True, train=False)




    if args.train:
        # Train model
        early_stopping = EarlyStopping(
            monitor="loss", 
            patience=2, 
            restore_best_weights=True
        )

        tensorboard = TensorBoard(
            histogram_freq=1, 
            write_images=True,
        )
        model.fit(traingen,
                epochs=10, 
                batch_size=batch_size, 
                validation_data= validgen,
                callbacks = [early_stopping, tensorboard]
                )

        model.save(f'vgg16_{args.weights}.h5')


    custom_id = {'car' : 0, 'motorbike' : 1, 'bus' : 2, 'truck' : 3}

