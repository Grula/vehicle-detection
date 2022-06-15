import os
import numpy as np
import tensorflow as tf


def _parse_file(filename, res = 256):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)

    image = tf.image.resize(image, (res, res))
    image = tf.cast(image, tf.float32)
    image = image / 127.5 - 1.0
    image = tf.transpose(image, perm=[2, 0, 1])
    return image

def create_dataset(data_base_dir, batch_size, resolution = 256, epochs=None,):
    # creating absolute path
    data_base_dir = os.path.abspath(data_base_dir)
    
    filenames = os.listdir(data_base_dir)
    # add absolute path to each file
    filenames = [os.path.join(data_base_dir, filename) for filename in filenames]

    # filenames = tf.constant(os.listdir(data_base_dir))
    filenames = tf.constant(filenames)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
   
    dataset = dataset.map(lambda x: _parse_file(x, resolution))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset

def _create_dataset(data_base_dir, batch_size, epochs=None, ):
    # create absolute path
    data_base_dir = os.path.abspath(data_base_dir)
    return tf.keras.utils.image_dataset_from_directory(data_base_dir, batch_size = batch_size)

def main():
    from PIL import Image

    res = 64
    batch_size = 4
    epochs = 1

    


if __name__ == '__main__':
    main()
