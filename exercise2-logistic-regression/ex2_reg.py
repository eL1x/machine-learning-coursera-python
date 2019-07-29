import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def h(X, theta):
    return sigmoid(X @ theta)


def map_features(x1, x2, degree):
    """
    Map the features 'x1', 'x2' into all polynomial terms up to the 'degree' power.
    """
    X_poly = pd.DataFrame()

    counter = 0
    for i in range(0, degree + 1):
        for j in range(0, i + 1):
            X_poly[counter] = np.power(x1, i - j) * np.power(x2, j)
            counter += 1

    X_poly = np.array(X_poly)

    return X_poly


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


def predict(predict_data, theta):
    predictions = h(predict_data, theta)
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0

    return predictions


def accuracy(predict_data, target, theta):
    predicted_vals = predict(predict_data, theta)
    accuracy = (predicted_vals == target) * 1 # (* 1) to convert from boolean to int
    accuracy = np.mean(accuracy)

    print('accuracy = {0}%'.format(accuracy * 100))


def load_data():
    data = pd.read_csv('data/ex2data2.txt', header=None, names=['Test1', 'Test2', 'Accepted'])
    X = data[['Test1', 'Test2']]
    X.insert(0, 'Ones', 1)
    y = data['Accepted']

    X = np.array(X)
    y = np.array(y).reshape((-1, 1))

    return X, y


def plot_data(X, y):
    plt.figure()
    pos_mask = (y == 1).reshape((-1,))
    neg_mask = (y == 0).reshape((-1,))
    plt.scatter(X[pos_mask, 1], X[pos_mask, 2], color='g', marker='+', label='Accepted')
    plt.scatter(X[neg_mask, 1], X[neg_mask, 2], color='r', marker='o', label='Not accepted')
    plt.xlabel('Microchip test 1')
    plt.ylabel('Microchip test 2')


def plot_all(X, y, theta):
    plot_data(X, y)
    plot_decision_boundary(theta)

    plt.legend(loc=1)
    plt.show()


def plot_decision_boundary(theta):
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)

    z = np.zeros((u.shape[0], v.shape[0]))
    for i in range(u.shape[0]):
        for j in range(v.shape[0]):
            z[i, j] = map_features(np.array([u[i]]), np.array([v[j]]), 6) @ theta

    z = z.T

    plt.contour(u, v, z, [0])


def main():
    X, y = load_data()
    X = map_features(X[:, 1], X[:, 2], 6)
    init_theta = np.zeros((X.shape[1], 1))
    lambda_r = 1

    # Find optimal theta
    theta_o, _, _ = opt.fmin_tnc(func=compute_cost_reg, x0=init_theta, fprime=compute_gradient_reg, args=(X, y, lambda_r), messages=opt.tnc.MSG_NONE)
    theta_o = theta_o.reshape((-1, 1))

    print("Optimal theta: ")
    print(theta_o)
    accuracy(X, y, theta_o)
    plot_all(X, y, theta_o)


if __name__ == "__main__":
    main()