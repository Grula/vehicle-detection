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

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

import argparse

# Open and read CSV with images and labels

def load_data(paths,  train = True):
    data = []
    classes = []
    targets =  []
    for csv_path in paths:
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader, None)
            for row in csv_reader:
                filename, label, startX, startY, w, h = row
                
                # Skip validation set if training
                if 'valid' in filename and train:
                    continue

                # Skip training set if validation
                if 'valid' not in filename and not train:
                    continue                    

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
                classes.append(label)
                targets.append((startX, startY, endX, endY))

    data = np.array(data, dtype="float32") / 255.
    classes = np.array(classes)
    targets = np.array(targets, dtype="float32")


    return data, classes, targets

def create_model(weights = False):
    # Load VGG16 model
    if weights:
        weights = 'imagenet'
    else:
        weights = None
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
    (startX, startY, endX, endY) = bbox
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

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--weights', type=int, default=0, help='Use weights, default 0 (False)')
    parser.add_argument('--augment', type=int, default=0, help='Use augmenttation, default 0 (False)')

    args = parser.parse_args()

    args.weights = bool(args.weights)
    args.augment = bool(args.augment)
    model_loaded = False

    # Check if model exists and load instead
    if os.path.isfile(f'model_weigh:{args.weights}_appen:{args.augment}.h5'):
        print("Loading model...")
        model = create_model(args.weights)
        model.load_weights(f'model_weigh:{args.weights}_appen:{args.augment}.h5')
        model_loaded = True

    # Data pathts
    data_paths = ['data/data.csv']
    if args.augment:
        data_paths.append('data/data_augmented.csv')


    print("Loading data...")
    data, classes, targets = load_data(data_paths, train=True)

    # Labelize classes
    lb = LabelBinarizer()
    classes = lb.fit_transform(classes)
    classes = np.array(classes, dtype = 'float32')

    print("Splitting data...")
    # Split into Train and Test
    X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(data, classes, targets, test_size=0.3, random_state=42)
    X_test, X_valid, y1_test, y1_valid, y2_test, y2_valid  = train_test_split(X_test, y1_test, y2_test, test_size=0.35, random_state=42)

    if not model_loaded:
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

    # Evalute model
    data, classes, targets = load_data(['data/data.csv'], train=False)

    vehicles = {'car': [], 'truck': [], 'bus': [], 'motorcycle': [], }
    # Pick one image from each set 
    for i in range(len(classes)):
        if classes[i] == 'car':
            vehicles['car'] = [data[i], targets[i]]
        elif classes[i] == 'bus':
            vehicles['bus'] = [data[i], targets[i]]
        elif classes[i] == 'truck':
            vehicles['truck'] = [data[i], targets[i]]
        elif classes[i] == 'motorbike':
            vehicles['motorcycle'] = [data[i], targets[i]]
        if all(vehicles.values()):
            break

    # Check if res folder exists
    if not os.path.exists('res'):
        os.mkdir('res')

    # Evalute images
    for vehicle, data in vehicles.items():
        image, bbox = data
        # convert from rgb to bgr
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'res/{vehicle}_real.jpg', draw_bbox(image*255, bbox))
        pred_label, pred_bbox = eval_model(model, image)
        cv2.imwrite(f'res/{vehicle}_pred.jpg', draw_bbox(image*255, pred_bbox))
        # Inverse transform to get real label
        real_label = lb.inverse_transform(np.array([pred_label]))[0]
        print(f'{vehicle}: ', real_label)




    