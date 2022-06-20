# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738



import tensorflow as tf



def DiffAugment(images, policy='', channels_first=False):
    if policy:
        if channels_first:
            for idx, image in enumerate(images):
                images[idx] = tf.transpose(image, [0, 2, 3, 1])
        
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                #! REMOVE CONTINUE IF YOU WANT TO USE RANDOM POLICY
                continue
                images = f(images)
        
        if channels_first:
            for idx, image in enumerate(images):
                images[idx] =  tf.transpose(image, [0, 3, 1, 2])
    return images


def rand_brightness(x):
    return x


def rand_saturation(x):
    return x


def rand_contrast(x):
    return x


def rand_translation(x, ratio=0.125):
   return x


def rand_cutout(x, ratio=0.5):
    return x


def rand_color_shift(x):
   return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
    'color_shift': [rand_color_shift]
}
