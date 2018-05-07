

import numpy as np
import tensorflow as tf
# from ops import * 
from tensorflow.contrib.layers import batch_norm, fully_connected, flatten



def discriminator(self, low, reuse=False):
    
    def leakyrelu(x, alpha=0.2, name='lrelu'):
        return tf.maximum(x, alpha * x, name=name)

    def gaussian_noise_layer(self, input_layer, std):
        noise = tf.random_normal(shape=tf.shape(input_layer),
                                     mean=0.0,
                                     stddev=std,
                                     dtype=tf.float32)
        return input_layer + noise
    
    down_sampling_layers = []
    n_filters = [16, 32, 64, 128, 256]
    n_filter_sizes = [65, 33, 17, 9, 9]

    print (low.shape)
    x = tf.reshape(low, [self.batch_size, low.shape[1], low.shape[2]])
    X = x
    layers = 5

    with tf.variable_scope('d_model') as scope:
        if reuse:
            scope.reuse_variables()
        print (':::Discriminator:::')
        print ('D-Block: ', x.shape)
    # conv
        # x = gaussian_noise_layer(self, x, self.discriminator_noise_std)
        for layer, nf, fs in zip(range(layers), n_filters, n_filter_sizes):
            x = tf.layers.conv1d(x, filters=nf ,kernel_size=fs, strides=2, activation=None, padding='same', kernel_initializer=tf.orthogonal_initializer(gain=1.0), bias_initializer=tf.constant_initializer(0.)) # cf: fitler, fs:kernel_size
            x = tf.layers.dropout(x, rate=self.keep_prob_var)
            x = leakyrelu(x, 0.2)
            print ('D-Block: ', x.shape)

    # output
        x = flatten(x)
        x = tf.squeeze(x)
        x = fully_connected(x, 1, activation_fn=None)
    
    return x