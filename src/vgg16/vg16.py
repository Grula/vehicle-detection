import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import sys
# Add src directory to path
sys.path.append('src')

import numpy as np

import csv


# https://github.com/zubairsamo/Object-Detection-With-Tensorflow-Using-VGG16/blob/main/Object_Detection_Using_VGG16_With_Tensorflow.ipynb

from sklearn.model_selection import train_test_split
import cv2

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer

from loss import categorical_focal_loss

# Open and read CSV with images and labels

def load_data(paths, undersamples = []):
    data = []
    classes = []
    targets =  []
    i = 0
    for csv_path in paths:
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader, None)
            for row in csv_reader:
                
                # Skip validation set
                if 'valid' in filename:
                    continue

                filename, label, startX, startY, w, h = row
                
                # check if lavel is in undersample list
                # if label in undersamples:
                #     continue

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
                # Just testing on my pc remove for colab
                if i > 10:
                    # break
                    pass
                i += 1
    data = np.array(data, dtype="float32") / 255.
    classes = np.array(classes)
    targets = np.array(targets, dtype="float32")


    return data, classes, targets


data, classes, targets = load_data(['data/data.csv','data/synthetic/data.csv' ])

# transfrom classes to one-hot encoding with keras
# transform classes to one-hot encoding
print(np.unique(classes))
lb = LabelBinarizer()
classes = lb.fit_transform(classes)
classes = np.array(classes, dtype = 'float32')


X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(data, classes, targets, test_size=0.3, random_state=42)
X_test, X_valid, y1_test, y1_valid, y2_test, y2_valid  = train_test_split(X_test, y1_test, y2_test, test_size=0.35, random_state=42)



vgg=VGG16(weights='imagenet',include_top=False,input_tensor=Input(shape=(224,224,3)))
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

model = Model(inputs = vgg.input,outputs = [classhead, bboxhead])

losses = {
	# "class_label": [categorical_focal_loss(alpha=[[.25, .25, .25, .25]], gamma=2)],
	"class_label": "categorical_crossentropy",
	"bounding_box": "mse",
    }

model.compile(optimizer="adam", loss=losses, metrics=["accuracy"])
# model.compile(optimizer="ftrl", loss=losses, metrics=["accuracy"])
# model.compile(optimizer="", loss=losses, metrics=["accuracy"])


trainTargets = {
    "class_label": y1_train,
    "bounding_box": y2_train
}

testTargets = {
    "class_label": y1_test,
    "bounding_box": y2_test
}

early_stopping_patience = 7
early_stopping = EarlyStopping(
    monitor="loss", 
    patience=early_stopping_patience, 
    restore_best_weights=True
)

tensorboard = TensorBoard(
    histogram_freq=1, 
    write_images=True,
)

model.fit(X_train, trainTargets,
         epochs=45, 
         batch_size=32, 
         validation_data=(X_test, testTargets),
         callbacks = [early_stopping, tensorboard]
         )

model.save('adam-cre-model.h5')