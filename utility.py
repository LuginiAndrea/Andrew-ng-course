import numpy
import pandas as pd
import numpy as np
from typing import Callable, List, Tuple
import matplotlib.pyplot as plt


# --- Utility functions ---

# Converts csv data into 2 arrays, one with features and one with results
def csv_to_np_arr(file_name: str, labels: List[str] = None, split=True):
    if labels is None:
        data = pd.read_csv(file_name, header=None)
    else:
        data = pd.read_csv(file_name, names=labels)
    split_point = data.shape[1] - 1
    if split:
        [X, y] = data.iloc[:, 0:split_point], data.iloc[:, split_point]
        y = y.to_numpy()
        X = X.to_numpy()
        return X, y
    else:
        X = data.to_numpy()
        return X


def add_intercept(X: np.ndarray):
    m = X.shape[0]
    if len(X.shape) == 1:
        return np.hstack((
            np.ones(1),
            X
        ))
    else:
        return np.hstack((
            np.ones((m, 1)),  # Add the intercept in all the rows
            X
        ))


def prepare_data(X, y, X_norm=None):
    if len(y.shape) < len(X.shape):
        y = y[:, np.newaxis]
    X = add_intercept(X)
    theta = np.zeros([X.shape[1], 1])
    if X_norm is None:
        return X, y, theta
    else:
        X_norm = add_intercept(X_norm)
        return X, y, X_norm, theta


def mean_normalization(X: numpy.ndarray):
    standard_deviation = np.std(X, axis=0)
    mean = np.mean(X, axis=0)
    return (X - mean) / standard_deviation, mean, standard_deviation  # This gives error when we have a constant column

# --- h and J ---


def sigmoid(Z):
    sig = 1 / (1 + np.exp(-Z))  # For very large Z this operation returns 1, so, in order to avoid having 1 as result,
    sig = np.minimum(sig, 0.999999)  # we set an upper bound
    sig = np.maximum(sig, 0.000001)  # and a lower bound
    return sig


def h(linear_regression):  # Vectorized version
    if linear_regression:  # Theta is considered already transposed
        return lambda theta, X: X @ theta
    else:
        return lambda theta, X: sigmoid(X @ theta)  # Sigmoid function


def J(linear_regression, reg_lambda=None):
    inner_h = h(linear_regression)  # h function to use in this case
    if linear_regression:
        no_reg_func = lambda theta, X, y: ((inner_h(theta, X) - y) ** 2) / (2 * y.shape[0])
    else:
        def no_reg_func(theta, X, y):
            predicted_vals = inner_h(theta, X)
            return sum(-y * np.log(predicted_vals) - (1 - y) * np.log(1 - predicted_vals)) / y.shape[0]

    if reg_lambda is None:
        func = no_reg_func
    else:
        def func(theta, X, y):
            cost = no_reg_func(theta, X, y)
            theta = theta[1:]  # Remove the intercept term theta
            return cost + (reg_lambda / (2 * y.shape[0])) * np.sum(theta ** 2)
    return func

# --- Gradient, training and predict


def gradient(linear_regression, reg_lambda=None):  # The only thing that changes in the gradient is the h function
    inner_h = h(linear_regression)
    non_reg_func = lambda theta, X, y: X.T @ (inner_h(theta, X) - y) / y.shape[0]
    if reg_lambda is None:
        func = non_reg_func
    else:
        def func(theta, X, y):
            grad = non_reg_func(theta, X, y)
            theta = theta[1:]  # Remove the intercept term theta
            return np.vstack((
                grad[0][:, np.newaxis],  # The grad of the intercept term
                grad[1:] + (reg_lambda / y.shape[0]) * theta
            ))
    return func


def gradient_descent(linear_regression, reg_lambda=None):
    inner_J = J(linear_regression, reg_lambda)
    inner_gradient = gradient(linear_regression, reg_lambda)

    def func(theta, X, y, alpha, iterations):
        J_history = []
        for _ in range(iterations):
            cost = inner_J(theta, X, y)
            grad = inner_gradient(theta, X, y)
            theta = theta - (alpha * grad)
            J_history.append(cost)
        return theta, J_history

    return func


def oneVsAll(n_labels, reg_lambda=None):
    inner_gradient_descent = gradient_descent(linear_regression=False, reg_lambda=reg_lambda)
    theta_arr, cost_arr = [], []

    def func(X, y, alpha, iterations):
        n_features = X.shape[1]
        for i in range(n_labels):
            theta, cost = inner_gradient_descent(
                np.zeros([n_features, 1]),
                X,
                np.where(y == i, [1], [0]),  # Train with the OneVsAll methodology
                alpha,
                iterations
            )
            theta_arr.append(theta)
            cost_arr.append(cost)
        return np.array(theta_arr), cost_arr
    return func


def classifier_predict(linear_regression):
    inner_h = h(linear_regression)

    def func(theta, X, mean=0, std=1, intercepted=False):  # The predict function
        X = (X - mean) / std
        X = add_intercept(X) if not intercepted else X
        percentage = inner_h(theta, X)
        if len(percentage) == 1:
            percentage = percentage[0]
        return percentage, percentage > 0.5

    return func


def classifier_oneVsAll_predict(thetas, X, mean=0, std=1, intercepted=False):
    predictions = sigmoid(X @ thetas.T)
    return np.argmax(predictions, axis=1)  # Returns the number of which classifier had the best result


# --- Plotting and printing ---

def plot_cost_history(J_history, figure_number):
    plt.figure(figure_number)
    plt.xlabel("Iterations")
    plt.ylabel("J")
    plt.plot(J_history)
    return plt.figure(figure_number)


def print_accuracy(res, y):
    count = 0
    for i in range(len(res)):
        if (res[i] and y[i] == 0) or (not res[i] and y[i] == 1):
            count = count + 1
    print(" Errors:", count, "\n", "Total size:", (len(y)), "\n", "% :", 100 - (count / len(y) * 100))







