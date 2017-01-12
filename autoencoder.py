import math
import numpy as np
import tensorflow as tf

n = 50
e = 20

inputs = tf.placeholder(tf.float32, shape=[None,n], name='inputs')

w1 = tf.Variable(tf.truncated_normal([n,e], stddev=1.0/math.sqrt(n)))
b1 = tf.Variable(tf.zeros([e]))

w2 = tf.Variable(tf.truncated_normal([e,n], stddev=1.0/math.sqrt(e)))
b2 = tf.Variable(tf.zeros([n]))

encoding = tf.nn.sigmoid(tf.matmul(inputs, w1)+b1, name='encoding')
reconstruction = tf.nn.sigmoid(tf.matmul(encoding, w2)+b2, name='reconstruction')
loss = tf.reduce_mean(tf.square(reconstruction - inputs))
tf.scalar_summary("loss", loss)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

summary_op = tf.merge_all_summaries()

writer = tf.train.SummaryWriter('logs', graph=tf.get_default_graph())

with tf.Session() as sess:
	init = tf.initialize_all_variables()
	result = sess.run(init)
	for i in range(100):
		train = np.random.rand(100,n)
		_, train_error, summary = sess.run([optimizer,loss, summary_op], feed_dict={inputs: train})
		writer.add_summary(summary, i)
	test = np.random.rand(1000,n)
	_, test_error = sess.run([optimizer,loss], feed_dict={inputs: test})
	print test_error
