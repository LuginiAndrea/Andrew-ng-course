import numpy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def compute_cost(X: numpy.ndarray, y: numpy.ndarray, theta_arr: numpy.ndarray):
    m = y.shape[0]  # numero di righe
    print("Shapes:\n", m, X.shape)
    # Basandoci sulla formula per calcolare l'ipotesi
    h = X.dot(theta_arr.transpose())
    # Basandoci sulla formula per calcolare il costo
    return (np.sum(np.subtract(h, y) ** 2)) / (2 * m)


def mean_normalization(X: numpy.ndarray):
    standard_deviation = np.std(X, axis=0)  # max - min
    mean = np.mean(X,axis=0)
    return (X - mean) / standard_deviation, mean, standard_deviation


def gradient_descent(X, y, theta_arr, alpha, max_iterations):
    m = y.shape[0]
    theta_arr = theta_arr.transpose()
    J_history = []
    for _ in range(0, max_iterations):
        h = np.dot(X, theta_arr)
        h = np.subtract(h, y)
        h = np.dot(X.transpose(), h)
        theta_arr = np.subtract(theta_arr, h.dot((alpha/m)))
        J_history.append(compute_cost(X,y,theta_arr.transpose()))
    return theta_arr.transpose(),J_history


def file_to_np_arr(file_name, labels):  # Converts csv data into 2 arrays, one with features and one with results
    data = pd.read_csv(file_name, names=labels)
    split_point = data.shape[1] - 1
    [X, y] = data.iloc[:, 0:split_point], data.iloc[:, split_point]
    X = X.to_numpy()
    y = y.to_numpy()
    return X, y


def __main__():
    [X, y] = file_to_np_arr("es1_data.csv", ["pop", "profits"]) #Qui, in quanto solo una feature, non serve normalizzare
    print(X)
    y = y[:, np.newaxis]
    m = y.shape[0]  # Numero di righe

    alpha = 0.01
    iterations = 1500
    X = np.hstack((np.ones((m, 1)), X))  # Aggiunge una colonna di uno "in testa" a X
    theta_arr = np.zeros([1, X.shape[1]])  # Crea np array con valori zero di dimensione 1x2 e fa l'inversione

    [theta_arr,J_history] = gradient_descent(X, y, theta_arr, alpha, iterations)
    print("\n\n---------------\n\n")

    [X, y] = file_to_np_arr("es_2_data.csv", ["size", "bedrooms", "price"])
    not_norm_X = X
    [X, mu, sigma] = mean_normalization(X)

    plt.plot()
    print('Computed mean:', mu)
    print('Computed standard deviation:', sigma)

    y = y[:, np.newaxis]
    m = y.shape[0]  # Numero di righe

    alpha = 0.01
    iterations = 400
    X = np.hstack((np.ones((m, 1)), X))  # Aggiunge una colonna di uno "in testa" a X
    theta_arr = np.zeros([1, X.shape[1]])  # Crea np array con valori zero di dimensione 1x2 e fa l'inversione

    theta_1, J_history_1 = gradient_descent(X, y, theta_arr, 0.3, 400)
    theta_2, J_history_2 = gradient_descent(X, y, theta_arr, 0.1, 400)
    theta_3, J_history_3 = gradient_descent(X, y, theta_arr, 0.03, 400)
    theta_4, J_history_4 = gradient_descent(X, y, theta_arr, 0.01, 400)
    theta_5, J_history_5 = gradient_descent(X, y, theta_arr, 0.003, 400)
    theta_6, J_history_6 = gradient_descent(X, y, theta_arr, 0.001, 400)
    plt.plot(J_history_1, label='0.3')
    plt.plot(J_history_2, label='0.1')
    plt.plot(J_history_3, label='0.02')
    plt.plot(J_history_4, label='0.01')
    plt.plot(J_history_5, label='0.003')
    plt.plot(J_history_6, label='0.001')
    #plt.show()
    print(theta_arr)
    # Con l'equazione della normale

    not_norm_X = np.hstack((np.ones((m,1)),not_norm_X))
    X = not_norm_X
    normal_theta = np.linalg.pinv(X.transpose().dot(X)).dot(X.transpose().dot(y))
    print(normal_theta)
    # Predict del costo di casa 1650 m^2 e 3 stanze da letto
    size_n = (1650 - mu[0]) / sigma[0]
    bed_n = (3 - mu[1]) / sigma[1]
    print(theta_1.dot([1,size_n,bed_n]))

    theta_arr.dot()
    #plt.plot(X[:, 1], X.dot(theta_arr), color="red")
    #plt.scatter(X[:,1], y, color="blue")
    #plt.show()



__main__()

# https://machinelearningmedium.com/2017/08/23/multivariate-linear-regression/
# https://nbviewer.org/github/susilvaalmeida/machine-learning-andrew-ng/blob/master/Programming%20Exercise%201%20-%20Linear%20Regression.ipynb
# https://www.kaggle.com/enespolat/andrew-ng-machine-learning-with-python-1
# Aggiungere negli appunti la parte di regressione polinomiale e di debugging del valore di alpha
