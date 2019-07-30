import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def h(X, theta):
    return sigmoid(X @ theta)


def compute_cost_reg(theta, X, y, lambda_r):
    """
    Compute cost with regularization. 'theta[0]' is not regularized.
    """
    m = X.shape[0] # number of examples
    reg = (lambda_r / (2 * m)) * (theta[1:].T @ theta[1:])
    J = ((1 / m) * (-y.T @ np.log(h(X, theta)) - (1 - y.T) @ np.log(1 - h(X, theta)))) + reg

    return J


def compute_gradient_reg(theta, X, y, lambda_r):
    """
    Compute gradient of cost with regularization. 'theta[0]' is not regularized.
    """
    m = X.shape[0]
    theta = theta.reshape((-1, 1))
    theta_c = theta.copy()
    theta_c[0] = 0
    gradient = ((1/m) * (X.T @ (h(X, theta) - y))) + (lambda_r / m) * theta_c

    return gradient


def one_vs_all(X, y, num_labels, lambda_r):
    """
    Train 'num_labels' regularized logistic regression classifiers, one for each of the label.
    """
    thetas = np.zeros((num_labels, X.shape[1]))
    init_theta = np.zeros((X.shape[1]))
    
    for i in range(num_labels):
        y_i = (y == i) * 1
        y_i = y_i.reshape((-1, 1))

        fmin = minimize(fun=compute_cost_reg, x0=init_theta, args=(X, y_i, lambda_r), method='TNC', jac=compute_gradient_reg)
        thetas[i,:] = fmin.x

    return thetas


def predict_one_vs_all(X, thetas):
    """
    Predict digit label using one-vs-all method -> get label with the highest probability/
    """
    predictions = h(X, thetas.T)

    # Get index with max probability - digit label
    labels = np.argmax(predictions, axis=1).reshape((-1, 1))

    return labels


def load_data():
    data = loadmat('data/ex3data1.mat')
    X = data['X']
    y = data['y']
    X = np.insert(X, 0, 1, axis=1)
    y[y == 10] = 0 # label digit '0' as '0', not as '10'
    
    return X, y


def main():
    X, y = load_data()

    # Train classifiers and make predictions
    thetas = one_vs_all(X, y, 10, 1)
    pred = predict_one_vs_all(X, thetas)
    print('\nTraining Set Accuracy: {0}%\n'.format(np.mean((pred == y) * 1) * 100))


if __name__ == "__main__":
    main()