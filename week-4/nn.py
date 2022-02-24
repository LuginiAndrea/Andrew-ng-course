import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import utility


def predict_neural_network(thetas, X, at_layer=None):
    """ intuition:
        theta is k x m x n matrix, with k = number of layers, m = neurons in a layer, n = features,
        so the thetas for the first layer are the ones where k = 0 => theta[0] = layer_weights = the 2D array for the
        first layer, and now the weights for the first node in the first layer are the ones = layer_weights[0], and this
        logic is used for every node in the neural network.

        We calculate the results of each node in a layer at the same time using:
        layer k: a = sigmoid (X @ theta[k].T)
    """
    at_layer = len(thetas) if at_layer is None else at_layer-1
    a = X
    for k in range(0, at_layer):
        print(a.shape, thetas[k].shape)
        a = utility.add_intercept(a)  # add the bias unit
        a = utility.sigmoid(a @ thetas[k].T)

    if at_layer == len(thetas):
        return np.argmax(a, axis=1)  # get the array with the predicted values
    else:
        return a


def displayData(X):
    # Compute rows, cols
    m, n = X.shape

    nbImagesPerRow = int(sqrt(m))
    columnsCount = (20 + 2) * nbImagesPerRow

    result = np.empty((0, columnsCount))
    row = 0
    while row < m:
        new_row = np.empty((20, 0))
        for col in range(nbImagesPerRow):
            new_row = np.c_[new_row, X[row].reshape(20, 20).T]
            new_row = np.c_[new_row, np.zeros((20, 2))]
            row = row + 1
        result = np.r_[result, new_row]
        result = np.r_[result, np.zeros((1, columnsCount))]

    # Display Image
    plt.imshow(result, cmap='gray', interpolation='nearest')


"""
    In this case, from the shape of the theta array we can deduce that the neural network configuration is:
    input layer: 400 inputs + bias unit
    hidden layer: 25 neurons + bias unit
    output layer: 10 neurons
"""


def __main__():
    X = utility.csv_to_np_arr("features.csv", None, False)
    y = utility.csv_to_np_arr("labels.csv", None, None)
    thetas = [utility.csv_to_np_arr("theta_1.csv", None, None), utility.csv_to_np_arr("theta_2.csv", None, None)]
    # ---- Display random inputs ----
    rand_indices = np.random.choice(X.shape[0], 100)
    displayData(X[rand_indices])
    plt.show()
    # ---- Print how accurate our model is
    utility.print_accuracy(
        res=utility.predict_neural_network(thetas, X),
        y=y.T.flatten()
    )


__main__()
