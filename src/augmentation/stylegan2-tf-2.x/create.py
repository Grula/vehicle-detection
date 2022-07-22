from dataclasses import dataclass
import os

from pyrsistent import thaw
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import time
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from utils import str_to_bool
from tf_utils import allow_memory_growth, split_gpu_for_testing
from load_models import load_generator, load_discriminator

from dataset import create_dataset

from losses import d_logistic, d_logistic_r1_reg, g_logistic_non_saturating, g_logistic_ns_pathreg


def initiate_models(g_params, d_params, ckpt_dir, use_custom_cuda):
    discriminator = load_discriminator(d_params, ckpt_dir=None, custom_cuda=use_custom_cuda)

    generator = load_generator(g_params=g_params, is_g_clone=False, ckpt_dir=None, custom_cuda=use_custom_cuda)
    g_clone = load_generator(g_params=g_params, is_g_clone=True, ckpt_dir=None, custom_cuda=use_custom_cuda)

    # set initial g_clone weights same as generator
    g_clone.set_weights(generator.get_weights())
    return discriminator, generator, g_clone
    pass

def initiate_model(g_params, ckpt_dir, use_custom_cuda):
    generator = load_generator(g_params=g_params, is_g_clone=False, ckpt_dir=ckpt_dir, custom_cuda=use_custom_cuda)
    # g_clone = load_generator(g_params=g_params, is_g_clone=True, ckpt_dir=None, custom_cuda=use_custom_cuda)
    return generator
    return generator, g_clone

class Creator(object):
    def __init__(self, t_params):
        self.use_custom_cuda = t_params['use_custom_cuda']
        self.model_base_dir = t_params['model_base_dir']
        self.output_dir = t_params['output_dir']

        #create output folder
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # copy network params
        self.g_params = t_params['g_params']

        self.generator= initiate_model(self.g_params, self.model_base_dir, self.use_custom_cuda)


    def gen_sample(self, num_samples = 1):
        # generate samples
        for idx in range(num_samples):
            test_z = tf.random.normal(shape=(1, self.g_params['z_dim']), dtype=tf.dtypes.float32)
            test_labels = tf.ones((1, self.g_params['labels_dim']), dtype=tf.dtypes.float32)
            
            fake = self.generator([test_z, test_labels], truncation_psi=0.5, training=False)
            as_tensor = tf.transpose(fake, [0, 2, 3, 1])[0]
            as_tensor = (tf.clip_by_value(as_tensor, -1.0, 1.0) + 1.0) * 127.5
            as_tensor = tf.cast(as_tensor, tf.uint8)
            # save image
            tf.keras.utils.save_img(f'{self.output_dir}/image{idx}.png',as_tensor, data_format='channels_last')


def filter_resolutions_featuremaps(resolutions, featuremaps, res):
    index = resolutions.index(res)
    filtered_resolutions = resolutions[:index + 1]
    filtered_featuremaps = featuremaps[:index + 1]
    return filtered_resolutions, filtered_featuremaps


def main():
    # global program arguments parser
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--use_custom_cuda', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--model_dir', required=True, type=str)
    parser.add_argument('--output_dir', default='./output', nargs='?', type=str)
    parser.add_argument('--num_samples', default=1, type=int)
    args = vars(parser.parse_args())


    # For folders in model_dir print them and ask user to select one
    model_dir_list = os.listdir(args['model_dir'])
    print('Available models:')
    for i, model_dir in enumerate(model_dir_list):
        print(f'\t{i}: {model_dir}')


    while True:
        try:
            model_dir_index = int(input('Select model: '))
            model_dir = model_dir_list[model_dir_index]
            break
        except:
            print('Invalid model index')
            continue

    _, name, res = model_dir.split('-')
    res = int(res.split('x')[0])

    # network params
    resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    featuremaps = [512, 512, 512, 512, 512, 256, 128, 64, 32]
    train_resolutions, train_featuremaps = filter_resolutions_featuremaps(resolutions, featuremaps, res)
    
    g_params = {
        'z_dim': 512,
        'w_dim': 512,
        'labels_dim': 0,
        'n_mapping': 8,
        'resolutions': train_resolutions,
        'featuremaps': train_featuremaps,
    }


    parameters = {
        # global params
        'use_custom_cuda': args['use_custom_cuda'],
        'model_base_dir': os.path.join(args['model_dir'], model_dir),

        # network params
        'g_params': g_params,

        # output params
        'output_dir': os.path.join(args['output_dir'],name),
        'train_res': res,
    }

    trainer = Creator(parameters)
    trainer.gen_sample(num_samples = args['num_samples'])

    return


if __name__ == '__main__':
    main()
