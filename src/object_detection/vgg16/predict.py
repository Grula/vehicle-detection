import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import sys
# Add src directory to path
sys.path.append('src')

import numpy as np

import csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


import cv2
import tensorflow as tf
from tensorflow import keras

from keras.applications import VGG16
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
# from tensorflow.keras.layers import Input, Flatten, Dense
# from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

from keras.utils import load_img, img_to_array

import argparse


if __name__ == '__main__':
    model = VGG16()
    image = load_img('bus.jpg', target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    yhat = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    # print the classification
    print('%s (%.2f%%)' % (label[1], label[2]*100))
    