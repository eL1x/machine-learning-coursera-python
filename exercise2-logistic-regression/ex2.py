import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def h(X, theta):
    return sigmoid(X @ theta)


def compute_cost(theta, X, y):
    m = X.shape[0] # number of examples
    J = (1 / m) * (-y.T @ np.log(h(X, theta)) - (1 - y.T) @ np.log(1 - h(X, theta)))

    return J


def compute_gradient(theta, X, y):
    m = X.shape[0]
    theta = theta.reshape((-1, 1))
    gradient = (1/m) * (X.T @ (h(X, theta) - y))

    return gradient


def predict(predict_data, theta):
    predictions = h(predict_data, theta)
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0

    return predictions


def accuracy(predict_data, target, theta):
    predicted_vals = predict(predict_data, theta)
    accuracy = (predicted_vals == target.reshape((-1,))) * 1 # (* 1) to convert from boolean to int
    accuracy = np.mean(accuracy)

    print('accuracy = {0}%'.format(accuracy * 100))


def load_data():
    data = pd.read_csv('data/ex2data1.txt', header=None, names=['Exam1', 'Exam2', 'Admitted'])
    X = data[['Exam1', 'Exam2']]
    X.insert(0, 'Ones', 1)
    y = data['Admitted']

    X = np.array(X)
    y = np.array(y).reshape((-1, 1))

    return X, y

    
def plot_data(X, y):
    plt.figure()
    pos_mask = (y == 1).reshape((-1,))
    neg_mask = (y == 0).reshape((-1,))
    plt.scatter(X[pos_mask, 1], X[pos_mask, 2], color='g', marker='+', label='Admitted')
    plt.scatter(X[neg_mask, 1], X[neg_mask, 2], color='r', marker='o', label='Not admitted')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')


def plot_decision_boundary(theta, X):
    x_vals = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    y_vals = -theta[0]/theta[2] - theta[1]/theta[2] * x_vals

    plt.plot(x_vals, y_vals, label='Decision boundary', c='b')


def plot_all(theta_o, X, y):
    plot_data(X, y)
    plot_decision_boundary(theta_o, X)

    plt.legend(loc=1)
    plt.show()


def main():
    X, y = load_data()
    init_theta = np.zeros((X.shape[1], 1))

    # Find optimal theta
    theta_o, _, _ = opt.fmin_tnc(func=compute_cost, x0=init_theta, fprime=compute_gradient, args=(X, y), messages=opt.tnc.MSG_NONE)

    print("Optimal theta: ")
    print(theta_o)
    accuracy(X, y, theta_o)
    plot_all(theta_o, X, y)


if __name__ == "__main__":
    main()