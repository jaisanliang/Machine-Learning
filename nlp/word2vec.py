import numpy as np
import tensorflow as tf

corpus = "I have two dogs one dog is named Alice and one dog is named Bob"
neighborhood = 1

def word2vec(inputs, targets):
	embeddings = tf.Variable(tf.random_uniform([v_size, e_size], -1.0, 1.0))
	w = tf.Variable(tf.truncated_normal([e_size, v_size], stddev = 1.0/math.sqrt(e_size)))
	b = tf.Variable(tf.zeros([v_size]))

	train_inputs = tf.placeholder(tf.int32, shape=None)
	train_labels = tf.placeholder(tf.int32, shape=None)

	embed = tf.nn.embedding_lookup(embeddings, train_inputs)

	loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels, num_sampled, vocabulary_size))

	optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

	train = optimizer.minimize(loss)

	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		result = sess.run(init)
		_ = sess.run(train, feed_dict={train_inputs: inputs, train_labels: targets})
		print result

def process_corpus(corpus):
	'''
	Expects corpus to be an untokenized string
	'''
	tokens = corpus.split()
	inputs = []
	targets = []
	enums = {}
	cur_enum = 0
	for i in range(neighborhood,len(tokens)-neighborhood):
		word = tokens[i]
		word_enum = cur_enum
		if word in enums:
			word_enum = enums[word]
		else:
			enums[word] = word_enum
			cur_enum += 1
		for j in range(i-neighborhood, i+neighborhood+1):
			if j == i:
				continue
			neighbor = tokens[j]
			neighbor_enum = cur_enum
			if neighbor in enums:
				neighbor_enum = enums[neighbor]
			else:
				enums[neighbor] = neighbor_enum
				cur_enum += 1
			inputs.append(word_enum)
			targets.append(neighbor_enum)
	return enums, np.array(inputs), np.array(targets)

enums, inputs, targets = process_corpus(corpus)
e_size = 3
v_size = len(enums)
word2vec(inputs, targets)


