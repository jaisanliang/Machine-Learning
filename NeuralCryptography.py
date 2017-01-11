import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle
import sys
import math
import time

def main():
	sess = tf.Session()

	# Hyperparameters
	BATCH_SIZE = 4096
	N = 16					# number of bits in message
	K = 16					# number of bits in shared symmetric key
	learning_rate = 0.0008

	train_key_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, K))
	train_message_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, N))

	l1_num_hidden = N + K
	l1_depth = 1

	l2_filter_size = 4
	l2_depth = 2
	l2_stride = 1

	l3_filter_size = 2
	l3_depth = 4
	l3_stride = 2

	l4_filter_size = 1
	l4_depth = 4
	l4_stride = 1

	l5_filter_size = 1
	l5_depth = 1
	l5_stride = 1

	# Enable dropout and weight decay normalization
	#dropout_prob = 1.0 # set to < 1.0 to apply dropout, 1.0 to remove
	#weight_penalty = 0.0 # set to > 0.0 to apply weight penalty, 0.0 to remove
	# Implement dropout
	#dropout_keep_prob = tf.placeholder(tf.float32)

	# Alice's network weights/parameters
	a_l1_weights = tf.Variable(tf.truncated_normal(
		[N+K, l1_num_hidden], stddev=0.1))
	a_l1_biases = tf.Variable(tf.zeros([l1_num_hidden]))

	a_l2_weights = tf.Variable(tf.truncated_normal(
		[l2_filter_size, l1_depth, l2_depth], stddev=0.1))
	a_l2_biases = tf.Variable(tf.constant(1.0, shape=[l2_depth]))
	a_l2_feat_map_size = int(math.ceil(float(l1_num_hidden) / l2_stride))

	a_l3_weights = tf.Variable(tf.truncated_normal(
		[l3_filter_size, l2_depth, l3_depth], stddev=0.1))
	a_l3_biases = tf.Variable(tf.constant(1.0, shape=[l3_depth]))
	a_l3_feat_map_size = int(math.ceil(float(a_l2_feat_map_size) / l3_stride))

	a_l4_weights = tf.Variable(tf.truncated_normal(
		[l4_filter_size, l3_depth, l4_depth], stddev=0.1))
	a_l4_biases = tf.Variable(tf.constant(1.0, shape=[l4_depth]))
	a_l4_feat_map_size = int(math.ceil(float(a_l3_feat_map_size) / l4_stride))

	a_l5_weights = tf.Variable(tf.truncated_normal(
		[l5_filter_size, l4_depth, l5_depth], stddev=0.1))
	a_l5_biases = tf.Variable(tf.constant(1.0, shape=[l5_depth]))
	a_l5_feat_map_size = int(math.ceil(float(a_l4_feat_map_size) / l5_stride))

	# Bob's network weights/parameters
	b_l1_weights = tf.Variable(tf.truncated_normal(
		[N+K, l1_num_hidden], stddev=0.1))
	b_l1_biases = tf.Variable(tf.zeros([l1_num_hidden]))

	b_l2_weights = tf.Variable(tf.truncated_normal(
		[l2_filter_size, l1_depth, l2_depth], stddev=0.1))
	b_l2_biases = tf.Variable(tf.constant(1.0, shape=[l2_depth]))
	b_l2_feat_map_size = int(math.ceil(float(l1_num_hidden) / l2_stride))

	b_l3_weights = tf.Variable(tf.truncated_normal(
		[l3_filter_size, l2_depth, l3_depth], stddev=0.1))
	b_l3_biases = tf.Variable(tf.constant(1.0, shape=[l3_depth]))
	b_l3_feat_map_size = int(math.ceil(float(b_l2_feat_map_size) / l3_stride))

	b_l4_weights = tf.Variable(tf.truncated_normal(
		[l4_filter_size, l3_depth, l4_depth], stddev=0.1))
	b_l4_biases = tf.Variable(tf.constant(1.0, shape=[l4_depth]))
	b_l4_feat_map_size = int(math.ceil(float(b_l3_feat_map_size) / l4_stride))

	b_l5_weights = tf.Variable(tf.truncated_normal(
		[l5_filter_size, l4_depth, l5_depth], stddev=0.1))
	b_l5_biases = tf.Variable(tf.constant(1.0, shape=[l5_depth]))
	b_l5_feat_map_size = int(math.ceil(float(b_l4_feat_map_size) / l5_stride))

	# Eve's network weights/parameters
	e_l1_weights = tf.Variable(tf.truncated_normal(
		[N, l1_num_hidden], stddev=0.1))
	e_l1_biases = tf.Variable(tf.zeros([l1_num_hidden]))

	e_l2_weights = tf.Variable(tf.truncated_normal(
		[l2_filter_size, l1_depth, l2_depth], stddev=0.1))
	e_l2_biases = tf.Variable(tf.constant(1.0, shape=[l2_depth]))
	e_l2_feat_map_size = int(math.ceil(float(l1_num_hidden) / l2_stride))

	e_l3_weights = tf.Variable(tf.truncated_normal(
		[l3_filter_size, l2_depth, l3_depth], stddev=0.1))
	e_l3_biases = tf.Variable(tf.constant(1.0, shape=[l3_depth]))
	e_l3_feat_map_size = int(math.ceil(float(e_l2_feat_map_size) / l3_stride))

	e_l4_weights = tf.Variable(tf.truncated_normal(
		[l4_filter_size, l3_depth, l4_depth], stddev=0.1))
	e_l4_biases = tf.Variable(tf.constant(1.0, shape=[l4_depth]))
	e_l4_feat_map_size = int(math.ceil(float(e_l3_feat_map_size) / l4_stride))

	e_l5_weights = tf.Variable(tf.truncated_normal(
		[l5_filter_size, l4_depth, l5_depth], stddev=0.1))
	e_l5_biases = tf.Variable(tf.constant(1.0, shape=[l5_depth]))
	e_l5_feat_map_size = int(math.ceil(float(e_l4_feat_map_size) / l5_stride))

	def model(train = False):
		# Alice's layers
		concat = tf.concat(1,[train_key_node,train_message_node])
		a1 = tf.nn.relu(tf.matmul(concat,a_l1_weights) + a_l1_biases)
		shape = a1.get_shape().as_list()
		a1 = tf.reshape(a1, [shape[0], shape[1], 1])

		a2 = tf.nn.conv1d(a1, a_l2_weights, l2_stride, padding='SAME')
		a2 = tf.nn.sigmoid(a2 + a_l2_biases)

		a3 = tf.nn.conv1d(a2, a_l3_weights, l3_stride, padding='SAME')
		a3 = tf.nn.sigmoid(a3 + a_l3_biases)

		a4 = tf.nn.conv1d(a3, a_l4_weights, l4_stride, padding='SAME')
		a4 = tf.nn.sigmoid(a4 + a_l4_biases)

		a5 = tf.nn.conv1d(a4, a_l5_weights, l5_stride, padding='SAME')
		a5 = tf.nn.sigmoid(a5 + a_l5_biases)
		shape = a5.get_shape().as_list()
		a5 = tf.reshape(a5, [shape[0], shape[1]])
		
		# Bob's layers
		concat = tf.concat(1,[train_key_node,a5])
		b1 = tf.nn.relu(tf.matmul(concat,b_l1_weights) + b_l1_biases)
		shape = b1.get_shape().as_list()
		b1 = tf.reshape(b1, [shape[0], shape[1], 1])

		b2 = tf.nn.conv1d(b1, b_l2_weights, l2_stride, padding='SAME')
		b2 = tf.nn.sigmoid(b2 + b_l2_biases)

		b3 = tf.nn.conv1d(b2, b_l3_weights, l3_stride, padding='SAME')
		b3 = tf.nn.sigmoid(b3 + b_l3_biases)

		b4 = tf.nn.conv1d(b3, b_l4_weights, l4_stride, padding='SAME')
		b4 = tf.nn.sigmoid(b4 + b_l4_biases)

		b5 = tf.nn.conv1d(b4, b_l5_weights, l5_stride, padding='SAME')
		b5 = tf.nn.sigmoid(b5 + b_l5_biases)
		shape = b5.get_shape().as_list()
		b5 = tf.reshape(b5, [shape[0], shape[1]])
		
		# Eve's layers
		e1 = tf.nn.relu(tf.matmul(a5,e_l1_weights) + e_l1_biases)
		shape = e1.get_shape().as_list()
		e1 = tf.reshape(e1, [shape[0], shape[1], 1])

		e2 = tf.nn.conv1d(e1, e_l2_weights, l2_stride, padding='SAME')
		e2 = tf.nn.sigmoid(e2 + e_l2_biases)

		e3 = tf.nn.conv1d(e2, e_l3_weights, l3_stride, padding='SAME')
		e3 = tf.nn.sigmoid(e3 + e_l3_biases)

		e4 = tf.nn.conv1d(e3, e_l4_weights, l4_stride, padding='SAME')
		e4 = tf.nn.sigmoid(e4 + e_l4_biases)

		e5 = tf.nn.conv1d(e4, e_l5_weights, l5_stride, padding='SAME')
		e5 = tf.nn.sigmoid(e5 + e_l5_biases)
		shape = e5.get_shape().as_list()
		e5 = tf.reshape(e5, [shape[0], shape[1]])
		return b5, e5

	# Training computation
	bob_decode, eve_decode = model()
	loss = tf.reduce_mean(tf.abs(bob_decode-train_message_node)
		+tf.square((N/2.0-tf.abs(eve_decode-train_message_node))/(N/2.0))
					+tf.abs(eve_decode-train_message_node))
	
	# Add weight decay penalty
	# loss = loss + weight_decay_penalty([layer3_weights, layer4_weights], weight_penalty)
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	'''
	# Predictions for the training, validation, and test data.
	batch_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(network_model(tf_valid_dataset))
	test_prediction = tf.nn.softmax(network_model(tf_test_dataset))
	train_prediction = tf.nn.softmax(network_model(tf_train_dataset))

	def train_model(num_steps=num_training_steps):
		Train the model with minibatches in a tensorflow session
		with tf.Session(graph=self.graph) as session:
			tf.initialize_all_variables().run()
			print 'Initializing variables...'
			
			for step in range(num_steps):
				offset = (step * batch_size) % (self.train_Y.shape[0] - batch_size)
				batch_data = self.train_X[offset:(offset + batch_size), :, :, :]
				batch_labels = self.train_Y[offset:(offset + batch_size), :]
				
				# Data to feed into the placeholder variables in the tensorflow graph
				feed_dict = {tf_train_batch : batch_data, tf_train_labels : batch_labels, 
							 dropout_keep_prob: dropout_prob}
				_, l, predictions = session.run(
				  [optimizer, loss, batch_prediction], feed_dict=feed_dict)
				if (step % 100 == 0):
					train_preds = session.run(train_prediction, feed_dict={tf_train_dataset: self.train_X,
																   dropout_keep_prob : 1.0})
					val_preds = session.run(valid_prediction, feed_dict={dropout_keep_prob : 1.0})
					print ''
					print('Batch loss at step %d: %f' % (step, l))
					print('Batch training accuracy: %.1f%%' % accuracy(predictions, batch_labels))
					print('Validation accuracy: %.1f%%' % accuracy(val_preds, self.val_Y))
					print('Full train accuracy: %.1f%%' % accuracy(train_preds, self.train_Y))

			# This code is for the final question
			if self.invariance:
				print "\n Obtaining final results on invariance sets!"
				sets = [self.val_X, self.translated_val_X, self.bright_val_X, self.dark_val_X, 
						self.high_contrast_val_X, self.low_contrast_val_X, self.flipped_val_X, 
						self.inverted_val_X,]
				set_names = ['normal validation', 'translated', 'brightened', 'darkened', 
							 'high contrast', 'low contrast', 'flipped', 'inverted']
				
				for i in range(len(sets)):
					preds = session.run(test_prediction, 
						feed_dict={tf_test_dataset: sets[i], dropout_keep_prob : 1.0})
					print 'Accuracy on', set_names[i], 'data: %.1f%%' % accuracy(preds, self.val_Y)

					# save final preds to make confusion matrix
					if i == 0:
						self.final_val_preds = preds 
	
	# save train model function so it can be called later
	self.train_model = train_model

def weight_decay_penalty(weights, penalty):
	return penalty * sum([tf.nn.l2_loss(w) for w in weights])

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
'''

if __name__ == '__main__':
	main()
