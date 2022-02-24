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


def __main__():
    X = utility.csv_to_np_arr("features.csv", None, False)
    y = utility.csv_to_np_arr("labels.csv", None, False)
    # Get weights from last exercise for testing purposes
    thetas = [utility.csv_to_np_arr("theta_1.csv", None, None), utility.csv_to_np_arr("theta_2.csv", None, None)]
    # ---- Display random inputs ----
    rand_indices = np.random.choice(X.shape[0], 100)
    displayData(X[rand_indices])
    # plt.show()
    utility.print_structure(thetas)
    # ---- display initial cost ----
    print("Initial cost:\n", utility.J_nn(num_labels=10, reg_lambda=1)(thetas, X, y))
    gradients = utility.gradient_nn(thetas, X, y, 10, 0)
    # print(gradients)
    # # # ---- random init parameters ----
    # thetas = utility.random_init_thetas([X.shape[1], 25, 10], 0.13)
    # utility.computer_numerical_gradient(, thetas)
    utility.check_nn_gradients()




__main__()
