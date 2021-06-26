from NeuralNetwork import neural_network
import numpy as np

def main():

    # X = (hours studying, hours sleeping), y = score on test
    x_all = np.array(([2, 9], [1, 5], [3, 6], [5, 10]), dtype=float) # input data
    y = np.array(([92], [86], [89]), dtype=float) # output
    
    # scale units
    x_all = x_all / np.max(x_all, axis=0) # scaling input data
    y = y / 100 # scaling output data (max test score is 100)
    
    # split data
    X = np.split(x_all, [3])[0] # training data
    x_predicted = np.split(x_all, [3])[1] # testing data

    nn = neural_network()
    
    #defining our output
    o = nn.forward(X)
    
    print("Predicted Output: \n" + str(o))
    print("Actual Output: \n" + str(y))

if __name__ == "__main__":
    main()