import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler

PCT_TRAINING = 80
EPOCHS = 1000

class neural_network(object):
  def __init__(self):
    #parameters
    self.inputSize = 2
    self.outputSize = 1
    self.hiddenSize = 3

    #weights
    # weight matrix of dimension (size of layer l, size of layer l-1)

    # weight matrix from input to hidden layer
  
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) 
    # weight matrix from hidden to output layer

    self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

    # bias vector of dimension (size of layer l, 1)
    # bias vector from input to hidden layer
    # self.b1 = np.zeros((1, self.hiddenSize))
    # bias vector from hidden to output layer
    # self.b2 = np.zeros((self.inputSize, 1))

  def forward(self, X):
    # forward propagation through our network
    # dot product of X (input) and first set of weights and bias
    self.z = np.dot(X, self.W1)
    # activation function
    self.z2 = self.sigmoid(self.z) 
    # dot product of hidden layer (z2) and second set of weights and bias
    self.z3 = np.dot(self.z2, self.W2)
    # final activation function
    o = self.sigmoid(self.z3) 
    return o

  def sigmoid(self, s):
    # activation function
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    #derivative of sigmoid
    return s * (1 - s)

  def backward(self, X, y, o):
    # backward propagate through the network
    # error in output
    self.o_error = y - o 
    # applying derivative of sigmoid to error
    self.o_delta = self.o_error*self.sigmoidPrime(o) 

    self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

    self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

  def train(self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)

  def saveWeights(self):
    np.savetxt("w1.txt", self.W1, fmt="%s")
    np.savetxt("w2.txt", self.W2, fmt="%s")

  def predict(self):
    print("Predicted data based on trained weights: ")
    print('Validation Data Input: \n' + str(scaler_x.inverse_transform(x_validation)))
    #print("Input (scaled): \n" + str(x_validation))
    print("Validation Data Output: \n" + str(scaler_y.inverse_transform(self.forward(x_validation))))

# sequence is the user input
seq = [int(x) for x in input('Input a series of numbers separated by spaces (Press enter when done): ').split()]

# create record id
int_id = list(range(len(seq)))

# create matrix
sequence_of_integers = np.column_stack((int_id, seq))
# slice matrix on second value
follow_up_sequence = sequence_of_integers[1:,1]
follow_up_sequence = np.array(follow_up_sequence)
follow_up_sequence = follow_up_sequence.reshape(follow_up_sequence.shape[0],-1)

# all x and y
x_all_orig = np.array((sequence_of_integers), dtype=float) 
y_orig = np.array((follow_up_sequence), dtype=float) # output

# scale all x and y
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
x_all_trans = scaler_x.fit_transform(x_all_orig)
y_trans = scaler_y.fit_transform(y_orig)

# split data
num_rows = np.shape(x_all_trans)[0]
splitPoint = math.trunc(num_rows * (PCT_TRAINING / 100))

# create training and validation data sets using split point
X_train = np.split(x_all_trans, [splitPoint])[0] 
x_validation = np.split(x_all_trans, [splitPoint])[1] 
y_to_pass_to_train_function = y_trans[:splitPoint,:]

nn = neural_network()

for i in range(EPOCHS): # trains the nn 1,000 times
  print("# " + str(i) + "\n")
  print("Training Data Input: \n" + str(scaler_x.inverse_transform(X_train)))
# print("Input (scaled): \n" + str(X))
  print("Training Data Output: \n" + str(scaler_y.inverse_transform(y_to_pass_to_train_function)))
# print("Actual Output (scaled): \n" + str(y))
  print("Training Data Predicted Output: \n" + str(scaler_y.inverse_transform(nn.forward(X_train))))
# print("Predicted Output (scaled): \n" + str(nn.forward(X)))
# print("Loss: \n" + str(np.mean(np.square(y - nn.forward(X))))) # mean squared error
  print("\n")
  nn.train(X_train, y_to_pass_to_train_function)

nn.saveWeights()
nn.predict()

