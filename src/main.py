import tensorflow as tf
from tensorflow import keras
from keras import backend as K

from tensorflow.keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input

from tensorflow.keras.layers import Dense, AveragePooling2D, Input, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical

import numpy as np
import cv2
import os

import csv

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    tpu_strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
  raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

def categorical_focal_loss(alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed


# Create a model VGG16
def create_model(classes = 2): 
    vgg = VGG16(weights=None, include_top=True, input_shape=(224, 224, 3), classes=classes)
    # vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    out = vgg.output

    # out = AveragePooling2D(pool_size=(2, 2))(out)
    out = Flatten()(out)
    
    bboxHead = Dense(128, activation="relu")(out)
    bboxHead = Dense(64, activation="relu")(bboxHead)
    bboxHead = Dense(32, activation="relu")(bboxHead)
    bboxHead = Dense(4, name="bounding_box")(bboxHead)

    classHead = Dense(512, activation="relu")(out)
    classHead = Dropout(0.2)(classHead)
    classHead = Dense(512, activation="relu")(classHead)
    classHead = Dropout(0.2)(classHead)
    classHead = Dense(classes, activation="softmax",  name="class_label")(classHead)

    model = Model(inputs=vgg.input, outputs=(bboxHead, classHead))

    losses = {
	"class_label": [categorical_focal_loss(alpha=[[.25, .25, .25, .25]], gamma=2)],
	"bounding_box": "mean_squared_error",
    }

    model.compile(loss=losses, optimizer='adam', metrics=["accuracy"])

    
    return model


# Open and read CSV with images and labels
def read_csv(csv_path, skip = 'valid'):
    images = []
    labels = []
    bbox = []
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if skip in row[0]:
                continue
            images.append(row[0])
            labels.append(row[1])
            bbox.append(row[2:])
            
    return images[1:], labels[1:], bbox[1:]


# Convert BBOX to double coordinates
def convert_bbox(bbox):
    new_bb = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    new_bb = [int(i) for i in new_bb]
    return new_bb

# Convert Cooridantes to relative coordinates
# def convert_coor(bbox, h, w):
#     new_bb = (bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h)
#     return new_bb

image_path, labels, bbox = read_csv('data/data.csv')


images = []
for i in image_path:
    images.append(preprocess_input(cv2.imread(i)))

bboxes = []
for i, box in enumerate(bbox):
    new_bbox = [int(i) for i in box]
    new_bbox = convert_bbox(new_bbox)
    # new_bbox = convert_coor(new_bbox, images[i].shape[0], images[i].shape[1])
    bboxes.append(new_bbox)
# resize images
# for i, img in enumerate(images):
#     images[i] = cv2.resize(img, (MH_HEIGHT, MH_WIDTH), interpolation = cv2.INTER_AREA)

# Convert images to numpy array and normilze 
# images = np.array(images, dtype="float32") / 255.0
# Convert labels to numpy array
labels = np.array(labels)
# Convert bboxes to numpy array to float32 
bboxes = np.array(bboxes, dtype="float32")


# Perform one hot encoder on labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
# transfrom to float32
labels = labels.astype("float32")


# train test split  
split = train_test_split(images, labels, bboxes, test_size=0.10, random_state=42)
# unpack the data split
(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainBBoxes, testBBoxes) = split[4:6]


with tpu_strategy.scope(): # creating the model in the TPUStrategy scope means we will train the model on the TPU
    model = create_model(len(lb.classes_))
print(model.summary())



trainTargets = {
	"class_label": trainLabels,
	"bounding_box": trainBBoxes
}

# construct a second dictionary, this one for our target testing
testTargets = {
	"class_label": testLabels,
	"bounding_box": testBBoxes
}

# Add early stopping
early_stopping_patience = 70
early_stopping = keras.callbacks.EarlyStopping(
    monitor="class_label_loss", 
    patience=early_stopping_patience, 
    restore_best_weights=True
)

try:
    # Load model
    model = keras.models.load_model("model.h5")
    print("Model loaded")
except:

    H = model.fit(
        trainImages, trainTargets,
        validation_data=(testImages, testTargets),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        # verbose=1,
        callbacks=[early_stopping])

model.save('model.h5')

# lb.inverse_transform()
def load_image(path):
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array

    img = load_img(path, target_size=(256, 256))
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)
    return img

# load valid data
valid_images, valid_labels, valid_bbox = read_csv('data/data.csv', skip = 'train')
i = np.random.randint(0, len(valid_images))
image = load_image(image_path[i-1])


# predict
bbox_prediction, class_prediction = model.predict(image)
print(lb.inverse_transform(class_prediction), ' == ', valid_labels[i-1])