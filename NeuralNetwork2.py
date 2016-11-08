import numpy as np

def ReLU(z):
	def ReLU_1d(z_1d):
		return max(0,z_1d)
	return np.array(map(ReLU_1d,z))

def ReLU_deriv(z):
	def ReLU_deriv_1d(z_1d):
		return float(z_1d > 0)
	return np.array(map(ReLU_deriv_1d,z))

def one_hot(num_classes,y):
	Y = np.zeros(num_classes)
	Y[int(y)] = 1.0
	return Y

def softmax(z):
	max_z = max(z)
	z = z - max_z
	denom = np.sum(np.exp(z))
	return np.exp(z)/denom

'''
Layers - array where there are k layers, including the input and output layers
'''
def neural_network_train(X, Y, layer_unit_counts, num_classes):
	num_layers = len(layer_unit_counts)
	# make matrices for weights, and initialize them randomly
	a = [np.zeros(num_units) for num_units in layer_unit_counts]
	z = [np.zeros(num_units) for num_units in layer_unit_counts]
	w = [np.random.normal(scale=1/layer_unit_counts[i]**0.5,size=(layer_unit_counts[i],layer_unit_counts[i+1])) for i in range(num_layers-1)]
	b = [np.zeros(num_units) for num_units in layer_unit_counts[1:]]
	delta = [np.zeros(num_units) for num_units in layer_unit_counts]
	# perform gradient descent
	eta = 1.0/1000
	n = len(X)
	#print "initial w is {0}".format(w)
	for iteration in range(10*n):
		# feedforward
		i = np.random.randint(n-1)
		#i = 0
		a[0] = X[i]
		for l in range(0,num_layers-2):
			z[l+1] = np.dot(np.transpose(w[l]),a[l])+b[l]
			a[l+1] = ReLU(z[l+1])
		# softmax
		z[-1] = np.dot(np.transpose(w[-1]),a[-2])+b[-1]
		a[-1] = softmax(z[-1])
		# output error calculation
		w_next = [np.copy(weights) for weights in w]
		b_next = [np.copy(bias) for bias in b]
		delta[-1] = a[-1]-one_hot(num_classes,Y[i][0])
		w_next[-1] -= eta*np.dot(a[-2][np.newaxis].T,delta[-1][np.newaxis])
		b_next[-1] -= eta*delta[-1]
		# backpropagation
		for l in range(num_layers-2,0,-1):
			delta[l] = np.dot(np.dot(np.diag(ReLU_deriv(z[l])),w[l]),delta[l+1])
			#gradient update
			w_next[l-1] -= eta*np.dot(a[l-1][np.newaxis].T,delta[l][np.newaxis])
			b_next[l-1] -= eta*delta[l]
		w = w_next
		b = b_next
		#print "a is {0}".format(a)
		#print "z is {0}".format(z)
		#print "delta is {0}".format(delta)
	#print "final w is {0}".format(w)
	return w,b

def neural_network_test(X, Y, w, b):
	num_layers = len(w)+1
	a = [[] for i in range(num_layers)]
	z = [[] for i in range(num_layers)]
	errors = 0
	for i in range(len(X)):
		# feedforward
		a[0] = X[i]
		for l in range(0,num_layers-2):
			z[l+1] = np.dot(np.transpose(w[l]),a[l])+b[l]
			a[l+1] = ReLU(z[l+1])
		# softmax
		z[-1] = np.dot(np.transpose(w[-1]),a[-2])+b[-1]
		a[-1] = softmax(z[-1])
		pred = np.argmax(a[-1])
		if pred != int(Y[i][0]):
			errors += 1
	return 1.0*errors/len(X)

dataset = 1
data_train = np.loadtxt('data/data{0}_train.csv'.format(dataset))
data_val = np.loadtxt('data/data{0}_validate.csv'.format(dataset))
data_test = np.loadtxt('data/data{0}_test.csv'.format(dataset))
X_train = data_train[:,:-1]
Y_train = (data_train[:,-1:]+1)/2
X_val = data_val[:,:-1]
Y_val = (data_val[:,-1:]+1)/2
X_test = data_test[:,:-1]
Y_test = (data_test[:,-1:]+1)/2
w,b = neural_network_train(X_train,Y_train,[2,5,2],2)
#print w,b
print "Error rate is {0}".format(neural_network_test(X_test,Y_test,w,b))
