# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

# suplemented by Nikola Grulovic
import cv2
import numpy as np


import tensorflow as tf



def DiffAugment(images, policy='', channels_first=True):
    if policy:
        if channels_first:
            images = tf.transpose(images, [0, 2, 3, 1])
        
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                images = f(images)
        
        if channels_first:
            images = tf.transpose(images, [0, 3, 1, 2])
    return images


def rand_brightness(x):
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) - 0.5
    x = x + magnitude
    return x


def rand_saturation(x):
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) * 2
    x_mean = tf.reduce_mean(x, axis=3, keepdims=True)
    x = (x - x_mean) * magnitude + x_mean
    return x


def rand_contrast(x):
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) + 0.5
    x_mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
    x = (x - x_mean) * magnitude + x_mean
    return x


def rand_translation(x, ratio=0.125):
    batch_size = tf.shape(x)[0]
    image_size = tf.shape(x)[1:3]
    shift = tf.cast(tf.cast(image_size, tf.float32) * ratio + 0.5, tf.int32)
    translation_x = tf.random.uniform([batch_size, 1], -shift[0], shift[0] + 1, dtype=tf.int32)
    translation_y = tf.random.uniform([batch_size, 1], -shift[1], shift[1] + 1, dtype=tf.int32)
    grid_x = tf.clip_by_value(tf.expand_dims(tf.range(image_size[0], dtype=tf.int32), 0) + translation_x + 1, 0, image_size[0] + 1)
    grid_y = tf.clip_by_value(tf.expand_dims(tf.range(image_size[1], dtype=tf.int32), 0) + translation_y + 1, 0, image_size[1] + 1)
    x = tf.gather_nd(tf.pad(x, [[0, 0], [1, 1], [0, 0], [0, 0]]), tf.expand_dims(grid_x, -1), batch_dims=1)
    x = tf.transpose(tf.gather_nd(tf.pad(tf.transpose(x, [0, 2, 1, 3]), [[0, 0], [1, 1], [0, 0], [0, 0]]), tf.expand_dims(grid_y, -1), batch_dims=1), [0, 2, 1, 3])
    return x


def rand_cutout(x, ratio=0.5):
    batch_size = tf.shape(x)[0]
    image_size = tf.shape(x)[1:3]
    cutout_size = tf.cast(tf.cast(image_size, tf.float32) * ratio + 0.5, tf.int32)
    offset_x = tf.random.uniform([tf.shape(x)[0], 1, 1], maxval=image_size[0] + (1 - cutout_size[0] % 2), dtype=tf.int32)
    offset_y = tf.random.uniform([tf.shape(x)[0], 1, 1], maxval=image_size[1] + (1 - cutout_size[1] % 2), dtype=tf.int32)
    grid_batch, grid_x, grid_y = tf.meshgrid(tf.range(batch_size, dtype=tf.int32), tf.range(cutout_size[0], dtype=tf.int32), tf.range(cutout_size[1], dtype=tf.int32), indexing='ij')
    cutout_grid = tf.stack([grid_batch, grid_x + offset_x - cutout_size[0] // 2, grid_y + offset_y - cutout_size[1] // 2], axis=-1)
    mask_shape = tf.stack([batch_size, image_size[0], image_size[1]])
    cutout_grid = tf.maximum(cutout_grid, 0)
    cutout_grid = tf.minimum(cutout_grid, tf.reshape(mask_shape - 1, [1, 1, 1, 3]))
    mask = tf.maximum(1 - tf.scatter_nd(cutout_grid, tf.ones([batch_size, cutout_size[0], cutout_size[1]], dtype=tf.float32), mask_shape), 0)
    x = x * tf.expand_dims(mask, axis=3)
    return x


def rand_color_shift(x:tf.Tensor):
    #TODO: 1. For each image in batch, convert values to 0-255 (for now)
    #TODO: 2. Each image will be converted to HSV color space
    #TODO: 3. Create a mask for excluded area
    #TODO: 4. Change color of excluded area
    #TODO: 5. Convert back to RGB color space
    #TODO: 6. Normilize to -1,1 (for now)

    tmp = x.numpy()

    # STEP 1:
    # >>> convert values to 0-255 (for now)
    # >>> for batch size apply to each image with np.map
    for img in tmp:
        # convert from -1,1 to 0,255
        img = (img + 1) * 127.5
        img = img.astype(np.uint8)
        # convert to HSV
        # STEP 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # STEP 3:
        # create a mask for excluded area
        # >>> create a mask for excluded area
        # >>> hue wont matter
        # >>> saturation will go from  0-10
        # >>> value will go from 0-45
        # H S V
        lower = (0,0,0)
        upper = (359,10,35)
        mask = cv2.inRange(img, lower, upper)

        # Save mask
        cv2.imwrite('mask.png', mask)

        #STEP 4:
        # hue will be incremeted by in range from (0-180)


        cv2.imwrite("test.jpg", img)
        break
    # tmp = tmp.astype(np.uint8)






    import os
    os._exit(1)
    return x






    # import os
    # os._exit(1)
    # https://www.tensorflow.org/tutorials/images/segmentation
    return x


AUGMENT_FNS = {
    'color_shift': [rand_color_shift],
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}
