import numpy as np

def ReLU(z):
	def ReLU_1d(z_1d):
		if z_1d < 0:
			return 0
		return z_1d
	return np.array(map(ReLU_1d,z))

def ReLU_deriv(z):
	def ReLU_deriv_1d(z_1d):
		return int(z_1d > 0)
	return np.array(map(ReLU_deriv_1d,z))

def one_hot(num_classes,y):
	Y = np.zeros(num_classes)
	Y[int(y)] = 1.0
	return Y

def softmax(z):
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
	eta = 1.0/100
	for i in range(len(X)):
		# feedforward
		a[0] = X[i]
		for l in range(0,num_layers-2):
			z[l+1] = np.dot(np.transpose(w[l]),a[l])+b[l]
			a[l+1] = ReLU(z[l+1])
		# softmax
		z[-1] = np.dot(np.transpose(w[-1]),a[-2])+b[-1]
		a[-1] = softmax(z[-1])
		# output error calculation
		delta[-1] = a[-1]-one_hot(num_classes,Y[i][0])
		w[-1] -= eta*a[-1][np.newaxis]*delta[-1]
		b[-1] -= eta*delta[-1]
		# backpropagation
		for l in range(num_layers-2,0,-1):
			delta[l] = np.dot(np.dot(np.diag(ReLU_deriv(z[l])),w[l]),delta[l+1])
			#gradient update
			w[l-1] -= eta*a[l][np.newaxis]*delta[l]
			b[l-1] -= eta*delta[l]
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

train = np.loadtxt('data/data_3class.csv')
X = train[:,0:2]
Y = train[:,2:3]
w,b = neural_network_train(X,Y,[2,4,3],3)
print neural_network_test(X,Y,w,b)
