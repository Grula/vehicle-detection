import numpy as np
from scipy.linalg import sqrtm

import tensorflow as tf
import tensorflow_probability as tfp
# import tensorflow probability as tf

from keras.applications.inception_v3 import preprocess_input



from DiffAugment_tf2 import DiffAugment




def d_logistic(real_images, generator, discriminator, z_dim, policy, labels=None):
    batch_size = tf.shape(real_images)[0]
    z = tf.random.normal(shape=[batch_size, z_dim], dtype=tf.float32)
    if labels is None:
        labels = tf.random.normal(shape=[batch_size, 0], dtype=tf.float32)

    # forward pass
    fake_images = generator([z, labels], training=True)
    
    real_images = DiffAugment(real_images, policy=policy)
    fake_images = DiffAugment(fake_images, policy=policy)

    real_scores = discriminator([real_images, labels], training=True)
    fake_scores = discriminator([fake_images, labels], training=True)

    # gan loss
    d_loss = tf.math.softplus(fake_scores)
    d_loss += tf.math.softplus(-real_scores)
    return d_loss


def d_logistic_r1_reg(real_images, generator, discriminator, z_dim, policy, labels=None):
    batch_size = tf.shape(real_images)[0]
    z = tf.random.normal(shape=[batch_size, z_dim], dtype=tf.float32)
    if labels is None:
        labels = tf.random.normal(shape=[batch_size, 0], dtype=tf.float32)

    # forward pass
    fake_images = generator([z, labels], training=True)

    real_images = DiffAugment(real_images, policy=policy)
    fake_images = DiffAugment(fake_images, policy=policy)

    real_scores = discriminator([real_images, labels], training=True)
    fake_scores = discriminator([fake_images, labels], training=True)

    # gan loss
    d_loss = tf.math.softplus(fake_scores)
    d_loss += tf.math.softplus(-real_scores)

    # gradient penalty
    with tf.GradientTape() as r1_tape:
        r1_tape.watch([real_images, labels])
        real_loss = tf.reduce_sum(discriminator([real_images, labels], training=True))

    real_grads = r1_tape.gradient(real_loss, real_images)
    r1_penalty = tf.reduce_sum(tf.math.square(real_grads), axis=[1, 2, 3])
    r1_penalty = tf.expand_dims(r1_penalty, axis=1)
    return d_loss, r1_penalty


def g_logistic_non_saturating(real_images, generator, discriminator, z_dim, policy, labels=None, ):
    batch_size = tf.shape(real_images)[0]
    z = tf.random.normal(shape=[batch_size, z_dim], dtype=tf.float32)
    if labels is None:
        labels = tf.random.normal(shape=[batch_size, 0], dtype=tf.float32)

    # forward pass
    fake_images = generator([z, labels], training=True)
    fake_images = DiffAugment(fake_images)
    
    fake_scores = discriminator([fake_images, labels], training=True)

    # gan loss
    g_loss = tf.math.softplus(-fake_scores)
    return g_loss


def g_logistic_ns_pathreg(real_images, generator, discriminator, z_dim,
                          pl_mean, pl_minibatch_shrink, pl_denorm, pl_decay,
                          policy, labels=None, ):
    batch_size = tf.shape(real_images)[0]
    z = tf.random.normal(shape=[batch_size, z_dim], dtype=tf.float32)
    if labels is None:
        labels = tf.random.normal(shape=[batch_size, 0], dtype=tf.float32)

    pl_minibatch = tf.maximum(1, tf.math.floordiv(batch_size, pl_minibatch_shrink))
    pl_z = tf.random.normal(shape=[pl_minibatch, z_dim], dtype=tf.float32)
    if labels is None:
        pl_labels = tf.random.normal(shape=[pl_minibatch, 0], dtype=tf.float32)
    else:
        pl_labels = labels[:pl_minibatch]

    # forward pass
    fake_images, w_broadcasted = generator([z, labels], ret_w_broadcasted=True, training=True)
    fake_images = DiffAugment(fake_images)
    
    fake_scores = discriminator([fake_images, labels], training=True)
    g_loss = tf.math.softplus(-fake_scores)

    # Evaluate the regularization term using a smaller minibatch to conserve memory.
    with tf.GradientTape() as pl_tape:
        pl_tape.watch([pl_z, pl_labels])
        pl_fake_images, pl_w_broadcasted = generator([pl_z, pl_labels], ret_w_broadcasted=True, training=True)

        pl_noise = tf.random.normal(tf.shape(pl_fake_images)) * pl_denorm
        pl_noise_applied = tf.reduce_sum(pl_fake_images * pl_noise)

    pl_grads = pl_tape.gradient(pl_noise_applied, pl_w_broadcasted)
    pl_lengths = tf.math.sqrt(tf.reduce_mean(tf.reduce_sum(tf.math.square(pl_grads), axis=2), axis=1))

    # Track exponential moving average of |J*y|.
    pl_mean_val = pl_mean + pl_decay * (tf.reduce_mean(pl_lengths) - pl_mean)
    pl_mean.assign(pl_mean_val)

    # Calculate (|J*y|-a)^2.
    pl_penalty = tf.square(pl_lengths - pl_mean)
    return g_loss, pl_penalty


def _tf_cov(x):
    mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
    mx = tf.matmul(tf.transpose(mean_x), mean_x)
    vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float32)
    cov_xx = vx - mx
    return cov_xx


def g_fid(real_images, interception, generator, discriminator, z_dim, policy, labels = None):
    batch_size = tf.shape(real_images)[0]
    z = tf.random.normal(shape=[batch_size, z_dim], dtype=tf.float32)
    if labels is None:
        labels = tf.random.normal(shape=[batch_size, 0], dtype=tf.float32)

    # forward pass
    fake_images = generator([z, labels], training=True)

    # traftorm to channel last in batch
    real_images = tf.transpose(real_images, perm=[0, 3, 2, 1])
    fake_images = tf.transpose(fake_images, perm=[0, 3, 2, 1])
    
    # scale images to at least 75x75 if they are lower than 75x75
    if real_images.shape[2] < 75:
        real_images = tf.image.resize(real_images, [75, 75])
        fake_images = tf.image.resize(fake_images, [75, 75])

    fake_images = preprocess_input(fake_images)


    # fake_scores = discriminator([fake_images, labels], training=True)
    act1 = interception.predict(real_images)
    act2 = interception.predict(fake_images)

    # act1 = tf.make_tensor_proto(act1)  
    # act2 = tf.make_tensor_proto(act2)

    # mu1, sigma1 = tf.reduce_mean(act1, axis=0), tfp.stats.covariance(act1)
    # mu2, sigma2 = tf.reduce_mean(act2, axis=0), tfp.stats.covariance(act2)

    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # ssdiff = tf.reduce_sum(tf.math.square(mu1 - mu2))

    # calculate sqrt of product between cov,
    covmean = sqrtm(sigma1.dot(sigma2))

    # covmean = tf.linalg.sqrtm(tf.experimental.numpy.dot(sigma1, sigma2))

    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    # fid = ssdiff + tf.linalg.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid