import numpy
import pandas as pd
import numpy as np
from typing import Callable, List, Tuple
import matplotlib.pyplot as plt
import random


# --- Utility functions ---

# Converts csv data into 2 arrays, one with features and one with results
def csv_to_np_arr(file_name: str, labels: List[str] = None, split=True):
    if labels is None:
        data = pd.read_csv(file_name, header=None)
    else:
        data = pd.read_csv(file_name, names=labels)
    if split:
        split_point = data.shape[1] - 1  # split before the last column
        [X, y] = data.iloc[:, 0:split_point], data.iloc[:, split_point]  # last column into y, the rest into X
        y = y.to_numpy()
        X = X.to_numpy()
        return X, y
    else:  # if there is nothing to split dump all into X
        X = data.to_numpy()
        return X


def add_intercept(X: np.ndarray):
    m = X.shape[0]
    if len(X.shape) == 1:  # Only one row
        return np.hstack((
            np.ones(1),
            X
        ))
    else:  # more than 1 row
        return np.hstack((
            np.ones((m, 1)),  # Add the intercept in the first column of all rows
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
    sig = np.minimum(sig, 0.9999999)  # we set an upper bound
    sig = np.maximum(sig, 0.0000001)  # and a lower bound
    return sig                        # this is done in order to prevent having elements = 1 in sig
                                      # because that would cause an error in logistic regression


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
            return np.sum(-y * np.log(predicted_vals) - (1 - y) * np.log(1 - predicted_vals)) / y.shape[0]

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
    return np.argmax(predictions, axis=1)  # Returns the number of the classifier which had the best result


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


# --- Neural Networks ---
def print_structure(thetas):
    vals = [theta.shape[0] for theta in thetas]
    structure = "Structure: " + str(thetas[0].shape[1]-1)
    for val in vals:
        structure = structure + " x " + str(val)
    print(structure)


def forward_propagation(thetas, X, history=False):
    activation_arr = []
    a = X  # X is the array of the first activation values
    for k in range(0, len(thetas)):
        a = add_intercept(a)  # add the bias unit
        a = sigmoid(a @ thetas[k].T)
        if history:
            activation_arr.append(a)
    return activation_arr if history else a


def partial_derivative(a):
    return a * (1 - a)  # element wise multiplication


def J_nn(num_labels, reg_lambda=None):

    def non_reg_func(thetas, X, y):
        n_examples = X.shape[0]
        Y = np.zeros((  # creates a n_examples X num_labels matrix
            n_examples,  # n of examples
            num_labels
        ))
        for i in range(n_examples):
            Y[i, y[i, 0]] = 1  # the index corresponding to the correct label for each row has value = 1
        prediction = forward_propagation(thetas, X)
        return np.sum(np.sum(-Y * np.log(prediction) - (1 - Y) * np.log(1 - prediction))) / n_examples

    if reg_lambda is None:
        func = non_reg_func
    else:  # regularization
        def func(thetas, X, y):
            cost = non_reg_func(thetas, X, y)
            for theta in thetas:  # regularize for every layer
                theta = theta[1:]  # remove bias unit
                cost = cost + (reg_lambda / (2 * y.shape[0])) * np.sum(np.sum(theta[:, ] ** 2))
            return cost

    return func


def gradient_nn(thetas, X, y, num_labels, reg_lambda=None):
    n_examples = X.shape[0]
    Y = np.zeros((  # creates a n_examples X num_labels matrix
        n_examples,  # n of examples
        num_labels
    ))
    for i in range(n_examples):
        Y[i, y[i, 0]] = 1  # the index corresponding to the correct label for each row has value = 1
    # add intercepted X to the activation array
    activation_arr = [add_intercept(X)] + forward_propagation(thetas, X, history=True)
    sigma = [activation_arr[-1] - Y]  # sigma^L = a^L - y
    delta = [sigma[-1].T @ activation_arr[-2]]  # find delta for the first row (no need to not consider bias)
    thetas_grad = []
    # Calculate sigma and delta
    for idx in range(1, len(thetas)):  # skip last iteration
        sigma = [
            (sigma[0] @ thetas[-idx][:, 1:]) * partial_derivative(activation_arr[-1-idx])
        ] + sigma
        delta = [
            sigma[0].T @ activation_arr[-2-idx]
        ] + delta

    # apply regularization
    # if reg_lambda is not None:
    #     reg_lambda = 0
    # thetas_grad.append(
    #         (delta[0][:, 1:] + reg_lambda * thetas[0][:, 1:])
    # )
    # for idx in range(1, len(thetas)):
    #     thetas_grad.append(
    #         (delta[idx] + reg_lambda * thetas[idx][:, 1:])
    # )
    thetas_grad = delta

    return [np.sum(d)/n_examples for d in thetas_grad]


def random_init_thetas(sizes, init_epsilon, already_biased=False):
    thetas = []
    already_biased = int(not already_biased)  # 1 if we have to add bias, 0 if we don't
    for i in range(0, len(sizes)-1):
        thetas.append(  # the + 1 is for the bias term
            np.random.uniform(-init_epsilon, init_epsilon, (sizes[i+1], sizes[i] + already_biased))
        )
    return thetas


def unroll_thetas(thetas):
    return np.array(
        [x for theta in [theta.ravel() for theta in thetas] for x in theta]
    )

def roll_thetas(unrolled_theta, sizes):



def computer_numerical_gradient(cost_function, thetas):
    # Unroll parameters
    nn_params = unroll_thetas(thetas)
    num_grad = np.zeros(nn_params.shape)
    perturb = np.zeros(nn_params.shape)
    length = nn_params.shape[0]

    e = 1e-4
    for p in range(length):
        # Set perturbation vector
        perturb[p] = e
        minus_theta = nn_params - perturb
        plut_theta = nn_params + perturb
        loss1, tmp = cost_function()
        loss2, tmp = cost_function()
        # Compute Numerical Gradient
        num_grad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0

    return num_grad




def check_nn_gradients(reg_lambda=None):
    """
        Creates a small neural network (max 8 x 8 x 8) and checks that
        the implementation of the backpropagation algorithm is good
    """
    n_examples, sizes = random.randint(5, 10), [random.randint(2, 8), random.randint(2, 8), random.randint(1, 8)]
    n_labels = sizes[-1]  # Last size is equal to the number of labels
    init_epsilon = 0.0001  # this value is used for debugging purposes
    thetas = random_init_thetas(sizes, init_epsilon)
    X = np.array(
        random_init_thetas([sizes[0]-1, n_examples], init_epsilon)
    ).squeeze()  # We squeeze it because random_init_thetas returns a 3D array, but we want X to be 2D
    y = np.array([random.randint(0, n_labels-1) for _ in X])
    y = y[:, np.newaxis]

    inner_cost = lambda nn_params: J_nn(n_labels, reg_lambda)(nn_params, X, y)
    gradients = gradient_nn(thetas, X, y, n_labels, 0)

    # finite difference method
    grad_checking_epsilon = 1e-4
    num_grad = computer_numerical_gradient(inner_cost, thetas)

    diff = np.linalg.norm(num_grad - gradients) / np.linalg.norm(num_grad + gradients)

    print('If your backpropagation implementation is correct, then \n',
          'the relative difference will be small (less than 1e-9). \n',
          '\nRelative Difference: \n', diff)









def predict_neural_network(thetas, X):
    return np.argmax(forward_propagation(thetas, X), axis=1)

