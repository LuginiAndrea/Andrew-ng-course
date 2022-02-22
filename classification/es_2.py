import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import utility


def map_feature_plot(x1, x2, degree):
    out = np.ones(1)
    for i in range(1, degree+1):
        for j in range(i+1):
            terms = (x1**(i-j) * x2**j)
            out = np.hstack((out, terms))
    return out


def map_feature_two_columns(X1, X2, degree):
    out = []
    for i in range(1, degree + 1):
        for j in range(i+1):
            out.append(
                (X1 ** (i-j)) * (X2 ** j)
            )
    return np.array(out).T


def __main__():
    X, y = utility.csv_to_np_arr("es_2.csv", ["test-1", "test-2", "accepted"])
    # --- Plotting data ---
    plt.figure(1)
    plt.xlabel("test-1")
    plt.ylabel("test-2")
    plt.scatter(X[:, 0], X[:, 1], c=y)
    # --- Prepare data ---
    # We can see that the data would fit well with a non-linear model,
    # so we do feature mapping in order to obtain a more circle-like model
    X = map_feature_two_columns(X[:, 0], X[:, 1], 6)
    X, y, theta = utility.prepare_data(X, y)
    reg_lambda = 1
    cost, grad = [
            utility.J(linear_regression=False, reg_lambda=reg_lambda)(theta, X, y),
            utility.gradient(linear_regression=False, reg_lambda=reg_lambda)(theta, X, y)
    ]
    print("--------- Initial values: ---------\n", cost[0])
    # --- Gradient descent ---
    reg_lambda = 1
    theta, cost = utility.gradient_descent(linear_regression=False, reg_lambda=reg_lambda)(theta, X, y, 3.3, 100)
    fig = utility.plot_cost_history(cost, 0)
    print("--------- After logistic regression ---------\n", cost[-1], "\n", theta)
    # --- Tests ---
    p, res = utility.classifier_predict(linear_regression=False)(theta, X[1:], intercepted=True)
    count = 0
    for i in range(len(res)):
        if (res[i] and y[i] == 0) or (not res[i] and y[i] == 1):
            count = count + 1
    print(count, (len(y)), ":", 100 - (count / len(y) * 100), "%")

    # --- Plot decision boundary ---
    plt.figure(1)
    u_vals = np.linspace(-1, 1.5, 50)
    v_vals = np.linspace(-1, 1.5, 50)
    z = np.zeros((len(u_vals), len(v_vals)))
    for i in range(len(u_vals)):
        for j in range(len(v_vals)):
            z[i, j] = map_feature_plot(u_vals[i], v_vals[j], 6) @ theta

    plt.contour(u_vals, v_vals, z.T, 0)
    plt.show()


__main__()


#https://towardsdatascience.com/andrew-ngs-machine-learning-course-in-python-regularized-logistic-regression-lasso-regression-721f311130fb