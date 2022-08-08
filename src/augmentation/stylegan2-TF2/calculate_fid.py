# example of calculating the frechet inception distance in Keras
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
from load_models import load_generator

import argparse
import os
import cv2
import tensorflow as tf
import numpy as np

import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def filter_resolutions_featuremaps(resolutions, featuremaps, res):
    index = resolutions.index(res)
    filtered_resolutions = resolutions[:index + 1]
    filtered_featuremaps = featuremaps[:index + 1]
    return filtered_resolutions, filtered_featuremaps

def generate_images(n_images, generator, ):
    fake_images = []
    for _ in range(n_images):
        test_z = tf.random.normal(shape=(1, g_params['z_dim']), dtype=tf.dtypes.float32)
        test_labels = tf.ones((1, g_params['labels_dim']), dtype=tf.dtypes.float32)

        psi = tf.random.uniform(shape=(1, 1), minval=0, maxval=1, dtype=tf.dtypes.float32)
        fake = generator([test_z, test_labels], truncation_psi=psi, training=False, truncation_cutoff = None)
        as_tensor = tf.transpose(fake, [0, 2, 3, 1])[0]
        as_tensor = (tf.clip_by_value(as_tensor, -1.0, 1.0) + 1.0) * 127.5
        as_tensor = tf.cast(as_tensor, tf.uint8)

        fake_images.append(as_tensor)
    return numpy.array(fake_images)

# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)
 
# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid
 
# prepare the inception v3 model
interception = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

# global program arguments parser
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_dir', required=True, type=str)
args = vars(parser.parse_args())


# For folders in model_dir print them and ask user to select one
model_dir_list = os.listdir(args['model_dir'])
model_dir_list.sort()
print('Available models:')
for i, model_dir in enumerate(model_dir_list):
    print(f'\t{i}: {model_dir}')


while True:
    try:
        model_dir_index = int(input('Select model: '))
        model_dir = model_dir_list[model_dir_index]
        break
    except Exception as e:
        print('Invalid model index')
        continue

splited_name = model_dir.split('-')
name = splited_name[1]
res = splited_name[2]
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
model_base_dir =  os.path.join(args['model_dir'], model_dir)
generator = load_generator(g_params=g_params, is_g_clone=False, ckpt_dir=model_base_dir, custom_cuda=False)

n_images = 32*100

# expand real images paths 
real_images_dir = os.path.join('data', name)
real_images = os.listdir(real_images_dir)
real_images = real_images * (n_images // len(real_images)) + real_images[:n_images % len(real_images)]
real_images = [os.path.join(real_images_dir, image) for image in real_images]

# shuffle
numpy.random.shuffle(real_images) 


# define params
batch_size = 32
n_batches = int(n_images / batch_size)
fids = []

for idx in range(n_batches):
    # start time    
    start_time = time.time()

    real_batch = real_images[idx*batch_size:(idx+1)*batch_size]

    real_batch = [cv2.resize(cv2.imread(image), (512,512)) for image in real_batch]
    real_batch = numpy.array(real_batch)
    real_batch = real_batch.astype('float32')

    real_batch = scale_images(real_batch, (299,299,3))
    real_batch = preprocess_input(real_batch)
 
    fake_batch = generate_images(batch_size, generator)
    fake_batch = fake_batch.astype('float32')

    fake_batch = scale_images(fake_batch, (299,299,3))
    fake_batch = preprocess_input(fake_batch)

    fid = calculate_fid(interception, real_batch, fake_batch)
    print('FID', fid)
    print('----------------------------------------------------')
    # end time
    end_time = time.time()
    
    print(f'Batch {idx} took : {end_time - start_time}s')
    fids.append(fid)

print("Average FID: %.3f" % (sum(fids) / n_batches))
print('----------------------------------------------------')



# Create plot 