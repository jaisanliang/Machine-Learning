import math
import numpy as np
import tensorflow as tf

def autoencoder(d,c=5,tied_weights=False):
	'''
	An autoencoder network with one hidden layer (containing the encoding),
	and sigmoid activation functions.

	Args:
		d: dimension of input.
		c: dimension of code.
		tied_weights: True if w1^T=w2
	Returns:
		Dictionary containing input placeholder Tensor and loss Variable
	Raises:
	'''
	inputs = tf.placeholder(tf.float32, shape=[None,d], name='input')

	w1 = tf.Variable(tf.truncated_normal([d,c], stddev=1.0/math.sqrt(d)))
	b1 = tf.Variable(tf.zeros([c]))

	w2 = tf.Variable(tf.truncated_normal([c,d], stddev=1.0/math.sqrt(c)))
	# TODO: Implement tied weights
	b2 = tf.Variable(tf.zeros([d]))

	code = tf.nn.sigmoid(tf.matmul(inputs, w1)+b1, name='encoding')
	reconstruction = tf.nn.sigmoid(tf.matmul(code, w2)+b2, name='reconstruction')
	loss = tf.reduce_mean(tf.square(reconstruction - inputs))
	tf.scalar_summary('loss', loss)
	return {'inputs': inputs, 'loss': loss}

def sparse_autoencoder(d,c=5,tied_weights=False):
	'''
	An sparse autoencoder network with one hidden layer (containing the encoding),
	and sigmoid activation functions.
	Implements sparsity using different techniques, including k-sparsity
	and including extra loss terms.

	Args:
		d: dimension of input.
		c: dimension of code.
		tied_weights: True if w1^T=w2
	Returns:
		Dictionary containing input placeholder Tensor and loss Variable
	Raises:
	TODO: implement this
	'''
	pass

def denoising_autoencoder(d,e=5,noise_type='g',noise_level=0.1):
	'''
	A denoising autoencoder with one hidden layer (containing the encoding),
	as described in Stacked Denoising Autoencoders, Vincent et. al. 2010
	(http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf).

	Args:
		d: dimension of input.
		c: dimension of code.
		tied_weights: True if w1^T=w2
		noise_type: 'g' for additive Gaussian noise,
		'm' for masking noise (fraction of elements forced to be 0),
		's' for salt-and-pepper noise (fraction of elements forced to min or max value,
			each with probability 0.5)
		noise_level: standard deviation for Gaussian noise,
			or fraction of corrupted inputs for masking/salt-and-pepper noise
	Returns:
		Dictionary containing input placeholder Tensor and loss Variable
	Raises:
	'''
	inputs = tf.placeholder(tf.float32, shape=[None,d], name='inputs')
	input_shape = tf.shape(inputs)
	corrupted_inputs = inputs + tf.truncated_normal(shape=input_shape, stddev=noise_level)
	if noise_type == 'm':
		random = tf.random_uniform(shape=input_shape)
		mask = tf.select(random > noise_level, tf.ones(input_shape), tf.zeros(input_shape))
		corrupted_inputs = inputs * mask
	elif noise_type == 's':
		random = tf.random_uniform(shape=input_shape)
		mask = tf.select(random > noise_level, tf.ones(input_shape), tf.zeros(input_shape))
		salt_pepper = tf.random_uniform(shape=input_shape)
		noise_mask = tf.select(salt_pepper > 0.5, tf.ones(input_shape), tf.zeros(input_shape))
		noise = tf.ones(input_shape) - mask
		corrupted_inputs = inputs * mask + noise * noise_mask

	w1 = tf.Variable(tf.truncated_normal([d,e], stddev=1.0/math.sqrt(d)))
	b1 = tf.Variable(tf.zeros([e]))

	w2 = tf.Variable(tf.truncated_normal([e,d], stddev=1.0/math.sqrt(e)))
	b2 = tf.Variable(tf.zeros([d]))

	encoding = tf.nn.sigmoid(tf.matmul(corrupted_inputs, w1)+b1, name='encoding')
	reconstruction = tf.matmul(encoding, w2)+b2		# squashing vs. nonsquashing activation function
	# reconstruction = tf.nn.sigmoid(tf.matmul(encoding, w2)+b2, name='reconstruction')
	loss = tf.reduce_mean(tf.square(reconstruction - inputs))
	tf.scalar_summary('loss', loss)
	return {'inputs': inputs, 'loss': loss}

def autoencoder_test():
	d = 50
	e = 20
	# ae = autoencoder(d,e)
	ae = denoising_autoencoder(d,e,noise_type='s')

	optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(ae['loss'])
	summary_op = tf.merge_all_summaries()
	writer = tf.train.SummaryWriter('logs', graph=tf.get_default_graph())

	with tf.Session() as sess:
		init = tf.initialize_all_variables()
		result = sess.run(init)
		for i in range(100):
			train = np.random.rand(100,d)
			_, train_error, summary = sess.run([optimizer, ae['loss'], summary_op], feed_dict={ae['inputs']: train})
			writer.add_summary(summary, i)
		test = np.random.rand(1000,d)
		_, test_error = sess.run([optimizer,ae['loss']], feed_dict={ae['inputs']: test})
		print test_error

autoencoder_test()
