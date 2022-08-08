import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np

import csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import pickle

import cv2

from tensorflow import keras

from keras.applications import VGG16
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping, TensorBoard

from keras.utils import load_img
from keras.utils import img_to_array

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



def data_generator(files, train = True, batch_size = 32):

    def get_classes(files):
        classes = []
        for file in files:
            path, _ = file.split(' ')
            label = path.split('/')[-2]
            classes.append(label)
        return classes

    def parse_file(file):
        path, bbox = file.split(' ')

        # Get label from path
        label = path.split('/')[-2]
        
        # Load image
        image = load_img(path, target_size=(224,224))
        image = img_to_array(image)
        image = image / 255.0 # Normalize image
        image = np.expand_dims(image,axis=0)
        
        # get bbox
        bb_lab = [float(x) for x in bbox.split(',')] # ignore ID on last spot
        bbox, label_id = bb_lab[:-1], int(bb_lab[-1])
        bbox = [bbox[0]/image.shape[1], bbox[1]/image.shape[0], bbox[2]/image.shape[1], bbox[3]/image.shape[0]] # normilize bbox
        bbox = np.array(bbox)

        return image, label_id, bbox

    n = len(files)
    i = 0

    classes = get_classes(files)
    n_classes = 4 # Hack

    # Check if lb.pickle exists
    # if os.path.isfile('lb.pickle'):
    #     with open('lb.pickle', 'rb') as f:
    #         lb = pickle.load(f)
    # else:
    #     lb.fit(classes)
    #     classes = lb.transform(classes)
    #     with open('lb.pickle', 'wb') as f:
    #         pickle.dump(lb, f)

    while True:

        batch_image = np.zeros((batch_size, 224, 224, 3))
        classes_batch = np.zeros((batch_size, 1))
        bboxes_batch = np.zeros((batch_size, 4))

        for num in range(batch_size):
            if i == 0:
                np.random.shuffle(files)

            image, label, bbox = parse_file(files[num])
            
            # label = lb.transform([label])

            batch_image[num, :, :, :] = image
            classes_batch[num, :] = keras.utils.to_categorical(label, n_classes)
            bboxes_batch[num, :] = bbox

            i = (i + 1) % n
        yield [batch_image, classes_batch, bboxes_batch]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--weights', type=str, default=None, help='Use weights, default to None, use imagenet if not specified')
    parser.add_argument('--load', type=str, default=None, help='Specify to load model, provide path to model .h5 file')
    parser.add_argument('--train', type=bool, default=1, help='Specify to train model, default to True')

    args = parser.parse_args()
    lb = LabelBinarizer()

    if args.load is not None:
        assert args.load.endswith('.h5'), 'Model must be .h5 file'
        model = keras.models.load_model(args.load)
    else: # create model
        model = create_model(args.weights)

    print("loaded/created model")

    train_list = 'data/train_data.txt'
    valid_list = 'data/valid_data.txt'

    with open(train_list) as f:
        lines_train = f.readlines()
    np.random.seed(42)
    np.random.shuffle(lines_train)
    np.random.seed(None)

    

    with open(valid_list) as f:
        lines_val = f.readlines()
    np.random.seed(42)
    np.random.shuffle(lines_train)
    np.random.seed(None)



    if args.train:
        # Train model
        early_stopping = EarlyStopping(
            monitor="loss", 
            patience=10, 
            restore_best_weights=True
        )

        tensorboard = TensorBoard(
            histogram_freq=1, 
            write_images=True,
        )
        batch_size = 16
        model.fit(data_generator(lines_val, train = True, batch_size = batch_size),
                epochs=100, 
                batch_size=batch_size, 
                validation_data= data_generator(valid_list, batch_size = batch_size),
                callbacks = [early_stopping, tensorboard]
                )

        model.save(f'model_weigh:{args.weights}_appen:{args.augment}.h5')


    custom_id = {'car' : 0, 'motorbike' : 1, 'bus' : 2, 'truck' : 3}



    exit()
    # Data pathts
    data_paths = ['data/data.csv']
    if args.augment:
        data_paths.append('data/data_augmented.csv')


    print("Loading data...")
    data, classes, bboxes = load_data(data_paths, train=True)

    # Labelize classes
    lb = LabelBinarizer()
    lb.fit(classes)
    classes = lb.transform(classes)
    # classes = lb.fit_transform(classes)
    # save lb for future use with pickle
    with open('lb.pickle', 'wb') as f:
        pickle.dump(lb, f)
    classes = np.array(classes, dtype = 'float32')

    print("Splitting data...")
    # Split into Train and Test
    X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(data, classes, bboxes, test_size=0.3, random_state=42)
    X_test, X_valid, y1_test, y1_valid, y2_test, y2_valid  = train_test_split(X_test, y1_test, y2_test, test_size=0.35, random_state=42)

    print("Creating model...")
    model = create_model(weights=args.weights)

    trainTargets = {
        "class_label": y1_train,
        "bounding_box": y2_train
    }

    testTargets = {
        "class_label": y1_test,
        "bounding_box": y2_test
    }

    early_stopping_patience = 10
    early_stopping = EarlyStopping(
        monitor="loss", 
        patience=early_stopping_patience, 
        restore_best_weights=True
    )

    tensorboard = TensorBoard(
        histogram_freq=100, 
        write_images=True,
    )

    model.fit(X_train, trainTargets,
            epochs=100, 
            batch_size=32, 
            validation_data=(X_test, testTargets),
            callbacks = [early_stopping, tensorboard]
            )

    model.save(f'model_weigh:{args.weights}_appen:{args.augment}.h5')




# def data_generator(annotation_lines, batch_size, anchors, num_classes, max_bbox_per_scale, annotation_type):
#     '''data generator for fit_generator'''
#     n = len(annotation_lines)
#     i = 0
#     # train_input_sizes = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
#     # train_input_sizes = [128, 160, 192, 224, 256, 288, 320, 352, 384, 416]  
#     train_input_sizes = [224, 256, 288, 320, 352, 384, 416, 448, 480, 512]
#     strides = np.array([8, 16, 32])

#     while True:
#         train_input_size = random.choice(train_input_sizes)

#         train_output_sizes = train_input_size // strides

#         batch_image = np.zeros((batch_size, train_input_size, train_input_size, 3))

#         batch_label_sbbox = np.zeros((batch_size, train_output_sizes[0], train_output_sizes[0],
#                                       3, 5 + num_classes))
#         batch_label_mbbox = np.zeros((batch_size, train_output_sizes[1], train_output_sizes[1],
#                                       3, 5 + num_classes))
#         batch_label_lbbox = np.zeros((batch_size, train_output_sizes[2], train_output_sizes[2],
#                                       3, 5 + num_classes))

#         batch_sbboxes = np.zeros((batch_size, max_bbox_per_scale, 4))
#         batch_mbboxes = np.zeros((batch_size, max_bbox_per_scale, 4))
#         batch_lbboxes = np.zeros((batch_size, max_bbox_per_scale, 4))

#         for num in range(batch_size):
#             if i == 0:
#                 np.random.shuffle(annotation_lines)

#             image, bboxes, exist_boxes = parse_annotation(annotation_lines[i], train_input_size, annotation_type)
#             label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = preprocess_true_boxes(bboxes, train_output_sizes, strides, num_classes, max_bbox_per_scale, anchors)
#             # tf.print("############################")
#             # tf.print("sbboxes ", sbboxes)
#             # tf.print("mbboxes ", mbboxes)
#             # tf.print("lbboxes ", lbboxes)
#             # tf.print("############################")
#             batch_image[num, :, :, :] = image
#             if exist_boxes:
#                 batch_label_sbbox[num, :, :, :, :] = label_sbbox
#                 batch_label_mbbox[num, :, :, :, :] = label_mbbox
#                 batch_label_lbbox[num, :, :, :, :] = label_lbbox
#                 batch_sbboxes[num, :, :] = sbboxes
#                 batch_mbboxes[num, :, :] = mbboxes
#                 batch_lbboxes[num, :, :] = lbboxes
#             i = (i + 1) % n
#         yield [batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, batch_sbboxes, batch_mbboxes, batch_lbboxes], np.zeros(batch_size)
    