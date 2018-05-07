

import numpy as np
import tensorflow as tf
from ops import * 



def generator(self, low, spk=None, name='g_com'):
	
	def make_z(shape, mean=0., std=0., name='z'):
		with tf.variable_scope(name) as scope:
			z_init = tf.random_normal_initializer(mean=mean, stddev = std)
			z = tf.get_variable('z', shape, initializer=z_init, trainable=False)
		return z

	def SubPixel1D(I, r):
		"""One-dimensional subpixel upsampling layer

	  Calls a tensorflow function that directly implements this functionality.
	  We assume input has dim (batch, width, r)
	  """
		with tf.name_scope('subpixel'):
			X = tf.transpose(I, [2,1,0]) # (r, w, b)
			X = tf.batch_to_space_nd(X, [r], [[0,0]]) # (1, r*w, b)
			X = tf.transpose(X, [2,1,0])
			return X
	

	down_sampling_layers = []
	n_filters = [64, 128, 256, 512, 512, 512, 512, 512, 512]
	# [128, 128, 128, 256, 256, 256, 512, 512, 512]
	n_filter_sizes = [129, 65, 33, 17, 9, 9, 9, 9, 9 ,9]
	# n_filters = [16, 32, 64, 128, 128, 256, 256, 512, 512, 512, 1024]
	# [128, 128, 128, 256, 256, 256, 512, 512, 512]
	# n_filter_sizes = [31, 31, 31, 31, 31, 31, 31, 31 ,31, 31, 31, 31, 31, 31]

	x = low
	X = x
	layers = 9

	with tf.variable_scope(name):
		# if reuse:
		# 	scope.reuse_variables()
		print (':::Generator:::')
	# downsampling
		for layer, nf, fs in zip(range(layers), n_filters, n_filter_sizes):
			x = tf.layers.conv1d(x, filters=nf ,kernel_size=fs, strides=2, activation=None, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)) # cf: fitler, fs:kernel_size
			# if layer > 0: x = tf.layers.batch_normalization(x, training=True)
			x = leakyrelu(x, 0.2)
			print ('D-Block: ', x.shape)
			down_sampling_layers.append(x)

	# bottleneck
		x = tf.layers.conv1d(x, filters=n_filters[-1], kernel_size=n_filter_sizes[-1], strides=2, activation=None, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
		# x = tf.layers.batch_normalization(x, training=True)
		x = tf.layers.dropout(x, rate=self.keep_prob_var)
		x = leakyrelu(x, 0.2)

	# add noise(z)
		z = make_z([self.batch_size, x.get_shape().as_list()[1], n_filters[-1]])
		x = tf.concat([z, x], axis=2)

	# upsampling
		for layer, nf, fs, scl in reversed(list(zip(range(layers), n_filters, n_filter_sizes, down_sampling_layers))):
			x = tf.layers.conv1d(x, filters=nf*2, kernel_size=fs, activation=None, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
			# x = tf.layers.batch_normalization(x, training=True)
			x = tf.layers.dropout(x, rate=self.keep_prob_var)
			x = tf.nn.relu(x)
			# print x.shape
			x = SubPixel1D(x, r=2)
			# print x.shape
			x = tf.concat([x, scl], axis=2)
			print ('U-Block: ', x.shape)

	# output
		x = tf.layers.conv1d(x, filters=2, kernel_size=9, activation=None, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
		# print x.shape
		x = SubPixel1D(x, r=2)
		print ("final conv layer shape: ", x.shape)
		# g = add([x,X])
		# g = x
		g = tf.add(x, X)
		# self.gen_wave_summ = histogram_summary('gen_wave', g)

		print ('final shape: ', g.shape)
		self.generator_built = True
		g_com_output = [g]
		g_com_output.append(z)
	
	return g_com_output

