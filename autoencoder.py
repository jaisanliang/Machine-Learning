import math
import numpy as np
import tensorflow as tf

def autoencoder(d,e=5):
	'''
	An autoencoder network with one hidden layer (containing the encoding).

	Args:
		d: dimension of input.
		e: dimension of encoding.
	Returns:
		Dictionary containing input placeholder Tensor and loss Variable
	Raises:
	'''
	inputs = tf.placeholder(tf.float32, shape=[None,d], name='input')

	w1 = tf.Variable(tf.truncated_normal([d,e], stddev=1.0/math.sqrt(d)))
	b1 = tf.Variable(tf.zeros([e]))

	w2 = tf.Variable(tf.truncated_normal([e,d], stddev=1.0/math.sqrt(e)))
	b2 = tf.Variable(tf.zeros([d]))

	encoding = tf.nn.sigmoid(tf.matmul(inputs, w1)+b1, name='encoding')
	reconstruction = tf.nn.sigmoid(tf.matmul(encoding, w2)+b2, name='reconstruction')
	loss = tf.reduce_mean(tf.square(reconstruction - inputs))
	tf.scalar_summary('loss', loss)
	return {'inputs': inputs, 'loss': loss}

def denoising_autoencoder(d,e=5):
	'''
	A denoising autoencoder with one hidden layer (containing the encoding),
	as described in Stacked Denoising Autoencoders, Vincent et. al. 2010
	(http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf).
	'''
	inputs = tf.placeholder(tf.float32, shape=[None,d], name='inputs')

	w1 = tf.Variable(tf.truncated_normal([d,e], stddev=1.0/math.sqrt(d)))
	b1 = tf.Variable(tf.zeros([e]))

	w2 = tf.Variable(tf.truncated_normal([e,d], stddev=1.0/math.sqrt(e)))
	b2 = tf.Variable(tf.zeros([d]))

	encoding = tf.nn.sigmoid(tf.matmul(inputs, w1)+b1, name='encoding')
	reconstruction = tf.nn.sigmoid(tf.matmul(encoding, w2)+b2, name='reconstruction')
	loss = tf.reduce_mean(tf.square(reconstruction - inputs))
	tf.scalar_summary('loss', loss)
	return {'inputs': inputs, 'loss': loss}

def autoencoder_test():
	d = 50
	e = 20
	ae = autoencoder(d,e)

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
