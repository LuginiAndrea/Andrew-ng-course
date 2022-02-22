import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import utility


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


def mycostfunc(theta, X, y, reg_lambda):
    """LRCOSTFUNCTION Compute cost and gradient for logistic regression with
      regularization
      J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
      theta as the parameter for regularized logistic regression and the
      gradient of the cost w.r.t. to the parameters.
    """

    m, n = X.shape  # number of training examples
    theta = theta.reshape((n, 1))

    prediction = utility.sigmoid(X @ theta)

    cost_y_1 = (1 - y) * np.log(1 - prediction)
    cost_y_0 = -1 * y * np.log(prediction)

    J = (1.0 / m) * np.sum(cost_y_0 - cost_y_1) + (reg_lambda / (2.0 * m)) * np.sum(np.power(theta[1:], 2))

    return J

def __main__():
    X = utility.csv_to_np_arr("features.csv", None, False)
    y = utility.csv_to_np_arr("labels.csv", None, None)
    np.place(y, y == 10, 0)  # replace the label 10 with 0
    # ---- Display random inputs ----
    rand_indices = np.random.choice(X.shape[0], 100)
    displayData(X[rand_indices])
    plt.show()
    # ---- Train our model ----
    X, y, _ = utility.prepare_data(X, y)
    thetas, costs = utility.oneVsAll(n_labels=10, reg_lambda=0.1)(X, y, 0.2, 400)
    thetas = thetas.squeeze(axis=2)  # Bring it from 3 dimensions to 2
    predictions = utility.classifier_oneVsAll_predict(thetas, X)
    print(predictions.shape)
    utility.print_accuracy(predictions, y)


__main__()
