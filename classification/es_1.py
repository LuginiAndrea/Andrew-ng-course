import matplotlib.pyplot as plt
import numpy as np
import utility


# Special thanks to
# https://towardsdatascience.com/andrew-ngs-machine-learning-course-in-python-logistic-regression-c0ae25509feb


def __main__():
    X, y = utility.csv_to_np_arr("es_1.csv", ["test-1", "test-2", "scores"])
    X_norm, mean, std = utility.mean_normalization(X)  # normalize X
    # ------- Plotting --------
    plt.xlabel("test-1")
    plt.ylabel("test-2")
    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y)
    # --- Prepare data ---
    X, y, X_norm, theta = utility.prepare_data(X, y, X_norm)
    # --- Cost and gradient at initial theta ---
    cost, grad = utility.J(linear_regression=False)(theta, X, y), utility.gradient(linear_regression=False)(theta, X, y)
    print("--------- Initial values: ---------\n", cost[0], "\n", grad)
    # --- Gradient descent ---
    theta, cost = utility.gradient_descent(linear_regression=False)(theta, X_norm, y, 1, 400)
    print("--------- After logistic regression ---------\n", cost[-1], "\n", theta)
    # --- Plot the decision boundary ---
    x_val = np.array([
        np.min(X_norm[:, 1]),
        np.max(X_norm[:, 1])
    ])
    y_val = -1/theta[2] * (theta[1] * x_val + theta[0])
    plt.plot(x_val, y_val, "r")
    plt.show()
    # --- Tests ---
    scores = [int(x) for x in input("Scores: ").split()]
    result = utility.classifier_predict(linear_regression=False)(theta, scores, mean, std)
    print(f"For a student with scores {scores}, we predict an admission probability of", result[0])


__main__()
