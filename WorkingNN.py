import numpy as np
from sklearn.preprocessing import MinMaxScaler

class neural_network(object):
  def __init__(self):
    #parameters
    self.inputSize = 2
    self.outputSize = 1
    self.hiddenSize = 3

    #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # weight matrix from input to hidden layer
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # weight matrix from hidden to output layer

  def forward(self, X):
    #forward propagation through our network
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
    self.z2 = self.sigmoid(self.z) # activation function
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
    o = self.sigmoid(self.z3) # final activation function
    return o

  def backward(self, X, y, o):  # backward propagate through the network
    self.o_error = y - o # error in output
    self.o_delta = self.o_error * self.sigmoidPrime(o) # applying derivative of sigmoid to error
    self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error
    self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

  def sigmoid(self, s):
    # activation function
    return 1 / (1 + np.exp(-s))

  def sigmoidPrime(self, s):  #derivative of sigmoid
    return s * (1 - s)

  def train(self, X, y):
    o = self.forward(X)
    self.backward(X, y, 0)

  def saveWeights(self):
    np.savetxt("w1.txt", self.W1, fmt="%s")
    np.savetxt("w2.txt", self.W2, fmt="%s")

  def predict(self):
    print("Predicted data based on trained weights: ")
    print("Input (scaled): \n" + str(x_predicted))
    print("Output: \n" + str(self.forward(x_predicted)))


# Original Input Data
x_all = np.array(([2, 9], [1, 5], [3, 6], [5, 10]), dtype=float)
# Predicted Output
y = np.array(([92], [86], [89]), dtype=float) 

# Scale units
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
x_all = scaler1.fit_transform(x_all)
y = scaler2.fit_transform(y)
#x_all = x_all/np.amax(x_all, axis=0) # scaling input data
#y = y/np.amax(y, axis=0)

# split data

# Training data
X = np.split(x_all, [3])[0] 
# Testing Data
x_predicted = np.split(x_all, [3])[1] 


nn = neural_network()
inverse1 = scaler1.inverse_transform(X)
# Trains the NN 1,000 times
for i in range(1000): 
  print("Input: \n" + str(inverse1))
  y_inverse = scaler2.inverse_transform(y)
  print("Actual Output: \n" + str(y_inverse))
  o = nn.forward(X)
  inverse = scaler2.inverse_transform(o)
  print("Predicted Output: \n" + str(inverse))
  print("Loss: \n" + str(np.mean(np.square(y - nn.forward(X))))) # mean squared error
  print("\n")
  nn.train(X, y)
