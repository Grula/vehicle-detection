"""
Retrain the YOLO model for your own dataset.
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import math
import random
import os
import cv2
import argparse

import tensorflow as tf
import numpy as np
from tensorflow import keras

import keras.backend as K
from keras.layers import Input
from keras.models import Model, load_model
from keras.optimizers import adam_v2
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import keras.layers as layers



from yolo4.model import preprocess_true_boxes, yolo4_body, yolo_loss, Mish
from yolo4.utils import get_random_data

from callback_eval import Evaluate

def get_relative_path(path):
    return os.path.join(sys.path[-1], path)


# python src/object_detection/v4Yolo/train.py --model_data src/object_detection/v4Yolo/model_data  --weights_name 512_yolo4_weights.h5 --log_dir  src/object_detection/v4Yolo/yolo_logs/


def _main():
     # global program arguments parser
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--model_data', type=str , default='model_data/', help='path to model data')
    parser.add_argument('--weights_name', type=str , default='yolo4_weights.h5', help='name of model weights')
    parser.add_argument('--model_name', type=str , default='512_yolo4.h5', help='name of model')
    parser.add_argument('--log_dir', type=str , default='logs/')


    args = vars(parser.parse_args())

    annotation_train_path = 'data/train_data.txt'
    # annotation_val_path = 'data/valid_data.txt'
    # annotation_train_path = '2012_train.txt'
    # annotation_val_path = '2012_val.txt'
    log_dir = args['log_dir']


    classes_path = os.path.join(args['model_data'], 'custom_classes.txt')
    anchors_path = os.path.join(args['model_data'], 'yolo4_anchors.txt')

    weights_path = os.path.join(args['model_data'],args['weights_name'])
    model_path = os.path.join(args['model_data'],args['model_name'])
    # classes_path = get_relative_path('model_data/custom_classes.txt')
    # anchors_path = get_relative_path('model_data/yolo4_anchors.txt')

    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    class_index = ['{}'.format(i) for i in range(num_classes)]
    anchors = get_anchors(anchors_path)

    max_bbox_per_scale = 4

    anchors_stride_base = np.array([
        [[12, 16], [19, 36], [40, 28]],
        [[36, 75], [76, 55], [72, 146]],
        [[142, 110], [192, 243], [459, 401]]
    ])

    anchors_stride_base = anchors_stride_base.astype(np.float32)
    anchors_stride_base[0] /= 8
    anchors_stride_base[1] /= 16
    anchors_stride_base[2] /= 32


    input_shape = (512, 512) # multiple of 32, hw

    with open(annotation_train_path) as f:
        lines_train = f.readlines()


    np.random.seed(7)
    np.random.shuffle(lines_train)
    np.random.seed(None)   

    lines_train = lines_train[:10]



    # We have to be carefull here, what if all instances of class go to one set
    lines_val = lines_train[:int(len(lines_train)*0.2)]
    lines_train = lines_train[int(len(lines_train)*0.2):]
    # get all unique classes in lines_val ( last number in each line )
    val_classes = set([line.split(',')[-1].strip() for line in lines_val])
    print("In validation set we have classes: ", val_classes)
    train_classes = set([line.split(',')[-1].strip() for line in lines_train])
    print("In training set we have classes: ", train_classes)
    num_train = len(lines_train)


    np.random.seed(42)
    np.random.shuffle(lines_val)
    np.random.seed(None)
    num_val = len(lines_val)




    model, model_body = create_model(input_shape, anchors_stride_base, num_classes,
                                    load_pretrained=True, freeze_body=0, # freeze from 1 or 2
                                     weights_path=weights_path, model_path = model_path)

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(os.path.join(args['log_dir'], 'best_weights.h5'),
        monitor='loss', save_weights_only=True, save_best_only=True, save_freq='epoch')

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.7, patience=3, verbose=1)
    
    early_stopping_1 = EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=1)
    early_stopping_2 = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1)

    csv = tf.keras.callbacks.CSVLogger(args['log_dir'] + "history.csv", append=True)

    evaluation = Evaluate(model_body=model_body, anchors=anchors, class_names=class_index,
         score_threshold=0.05, tensorboard=logging, weighted_average=True, eval_lines=lines_val, log_dir=log_dir,
         image_shape = input_shape)
    stop_on_nan = tf.keras.callbacks.TerminateOnNaN()




    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    n_epochs = 0 
    if True:
        epoch = 200
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=1000,
            decay_rate=0.9
            )
        model.compile(optimizer=adam_v2.Adam(learning_rate=lr_schedule), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        # model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        # model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-2), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        # model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=1e-1), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        batch_size = 1
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        h = model.fit(data_generator_wrapper(lines_train, batch_size, anchors_stride_base, num_classes, max_bbox_per_scale, 'train'),
                steps_per_epoch=max(1, num_train//batch_size),
                epochs=epoch,
                initial_epoch=0,
                callbacks=[logging, checkpoint, early_stopping_1, stop_on_nan, csv])
        n_epochs = len(h.history['loss'])

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if False:
        epoch = 200 + n_epochs
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=adam_v2.Adam(learning_rate=1e-5), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 4 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit(data_generator_wrapper(lines_train, batch_size, anchors_stride_base, num_classes, max_bbox_per_scale, 'train'),
            steps_per_epoch=max(1, num_train//batch_size),
            epochs=epoch,
            initial_epoch=n_epochs,
            # callbacks=[logging, checkpoint, reduce_lr, early_stopping])
            callbacks=[logging, checkpoint, reduce_lr, early_stopping_2, evaluation, stop_on_nan, csv])

    # Further training if needed.

    # model.save_weights(os.path.join(args['model_data'], 'final_weights.h5'))


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)



def create_model(input_shape, anchors_stride_base, num_classes, load_pretrained=True, freeze_body=2,
            weights_path=None, model_path=None):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    # image_input = Input(shape=input_shape+(3,))
    h, w = input_shape  
    num_anchors = len(anchors_stride_base)

    max_bbox_per_scale = 4
    iou_loss_thresh = 0.5



    model_body = yolo4_body(image_input, num_anchors, num_classes)
    # model_body = load_model(weights_path, custom_objects={'Mish':Mish})
    model_body = load_model(model_path, custom_objects={'Mish':Mish})
    
    print('Create YOLOv4 model with {} anchors and {} classes.'.format(num_anchors*3, num_classes))

    if load_pretrained:

        try:
            model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print('Load weights {}.'.format(weights_path))
        except:
            print('Load weights failed.')
        
        if freeze_body in [1, 2]:   
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (250, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    y_true = [
        layers.Input(name='input_2', shape=(None, None, 3, (num_classes + 5))),  # label_sbbox
        layers.Input(name='input_3', shape=(None, None, 3, (num_classes + 5))),  # label_mbbox
        layers.Input(name='input_4', shape=(None, None, 3, (num_classes + 5))),  # label_lbbox
        layers.Input(name='input_5', shape=(max_bbox_per_scale, 4)),             # true_sbboxes
        layers.Input(name='input_6', shape=(max_bbox_per_scale, 4)),             # true_mbboxes
        layers.Input(name='input_7', shape=(max_bbox_per_scale, 4))              # true_lbboxes
    ]

    loss_list = layers.Lambda(yolo_loss, name='yolo_loss',
                           arguments={'num_classes': num_classes, 'iou_loss_thresh': iou_loss_thresh,
                                      'anchors': anchors_stride_base})([*model_body.output, *y_true])

    # sys.stdout = open('train_summ.txt', 'w')
    # model_body.summary()
    # sys.stdout.close()
    # sys.stdout = sys.__stdout__
    # import os
    # os._exit(0)

    model = Model([model_body.input, *y_true], loss_list)


    return model, model_body

def random_fill(image, bboxes):
    if random.random() < 0.5:
        h, w, _ = image.shape
        # Fill with black borders in the horizontal direction to train small object detection
        if random.random() < 0.5:
            dx = random.randint(int(0.5*w), int(1.5*w))
            black_1 = np.zeros((h, dx, 3), dtype='uint8')
            black_2 = np.zeros((h, dx, 3), dtype='uint8')
            image = np.concatenate([black_1, image, black_2], axis=1)
            bboxes[:, [0, 2]] += dx
        # Fill with black borders vertically to train small object detection
        else:
            dy = random.randint(int(0.5*h), int(1.5*h))
            black_1 = np.zeros((dy, w, 3), dtype='uint8')
            black_2 = np.zeros((dy, w, 3), dtype='uint8')
            image = np.concatenate([black_1, image, black_2], axis=0)
            bboxes[:, [1, 3]] += dy
    return image, bboxes

def random_horizontal_flip(image, bboxes):
    if random.random() < 0.5:
        _, w, _ = image.shape
        image = image[:, ::-1, :]
        bboxes[:, [0,2]] = w - bboxes[:, [2,0]]
    return image, bboxes

def random_crop(image, bboxes):
    if random.random() < 0.5:
        h, w, _ = image.shape
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]

        crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
        crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
        crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
        crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

        image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
    return image, bboxes

def random_translate(image, bboxes):
    if random.random() < 0.5:
        h, w, _ = image.shape
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]

        tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
        ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

        M = np.array([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (w, h))

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
    return image, bboxes

def image_preprocess(image, target_size, gt_boxes):
    # The incoming training image is in rgb format
    ih, iw = target_size
    h, w = image.shape[:2]
    interps = [   # 随机选一种插值方式
        cv2.INTER_NEAREST,
        # cv2.INTER_LINEAR,
        # cv2.INTER_AREA,
        # cv2.INTER_CUBIC,
        # cv2.INTER_LANCZOS4,
    ]
    method = np.random.choice(interps)   # 随机选一种插值方式
    scale_x = float(iw) / w
    scale_y = float(ih) / h
    image = cv2.resize(image, None, None, fx=scale_x, fy=scale_y, interpolation=method)

    pimage = image.astype(np.float32) / 255.
    if gt_boxes is None:
        return pimage
    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale_x
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale_y
        return pimage, gt_boxes

def parse_annotation(annotation, train_input_size, annotation_type):
    line = annotation.split()
    image_path = line[0]
    if not os.path.exists(image_path):
        raise KeyError("%s does not exist ... " %image_path)
    image = np.array(cv2.imread(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # No items are marked, that is, each grid is treated as a background
    exist_boxes = True
    if len(line) == 1:
        bboxes = np.array([[10, 10, 101, 103, 0]])
        exist_boxes = False
    else:
        bboxes = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in line[1:]])
    if annotation_type == 'train':
        image, bboxes = random_fill(np.copy(image), np.copy(bboxes))    # Open when dataset lacks small objects
        image, bboxes = random_horizontal_flip(np.copy(image), np.copy(bboxes))
        image, bboxes = random_crop(np.copy(image), np.copy(bboxes))
        image, bboxes = random_translate(np.copy(image), np.copy(bboxes))
        pass
    image, bboxes = image_preprocess(np.copy(image), [train_input_size, train_input_size], np.copy(bboxes))
    return image, bboxes, exist_boxes

def data_generator(annotation_lines, batch_size, anchors, num_classes, max_bbox_per_scale, annotation_type):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    # train_input_sizes = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
    # train_input_sizes = [320, 352, 384, 416, 448, 480, 512]
    train_input_sizes = [512]
    strides = np.array([8, 16, 32])

    while True:
        train_input_size = random.choice(train_input_sizes)

        train_output_sizes = train_input_size // strides

        batch_image = np.zeros((batch_size, train_input_size, train_input_size, 3))

        batch_label_sbbox = np.zeros((batch_size, train_output_sizes[0], train_output_sizes[0],
                                      3, 5 + num_classes))
        batch_label_mbbox = np.zeros((batch_size, train_output_sizes[1], train_output_sizes[1],
                                      3, 5 + num_classes))
        batch_label_lbbox = np.zeros((batch_size, train_output_sizes[2], train_output_sizes[2],
                                      3, 5 + num_classes))

        batch_sbboxes = np.zeros((batch_size, max_bbox_per_scale, 4))
        batch_mbboxes = np.zeros((batch_size, max_bbox_per_scale, 4))
        batch_lbboxes = np.zeros((batch_size, max_bbox_per_scale, 4))

        for num in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, bboxes, exist_boxes = parse_annotation(annotation_lines[i], train_input_size, annotation_type)
            label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = preprocess_true_boxes(bboxes, train_output_sizes, strides, num_classes, max_bbox_per_scale, anchors)
            batch_image[num, :, :, :] = image
            if exist_boxes:
                batch_label_sbbox[num, :, :, :, :] = label_sbbox
                batch_label_mbbox[num, :, :, :, :] = label_mbbox
                batch_label_lbbox[num, :, :, :, :] = label_lbbox
                batch_sbboxes[num, :, :] = sbboxes
                batch_mbboxes[num, :, :] = mbboxes
                batch_lbboxes[num, :, :] = lbboxes
            i = (i + 1) % n
        yield [batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, batch_sbboxes, batch_mbboxes, batch_lbboxes], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, anchors, num_classes, max_bbox_per_scale, annotation_type):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None

    return data_generator(annotation_lines, batch_size, anchors, num_classes, max_bbox_per_scale, annotation_type)

if __name__ == '__main__':
    _main()
