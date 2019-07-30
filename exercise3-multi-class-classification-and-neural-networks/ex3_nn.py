import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def feedforward(X, theta1, theta2):
    # Layer1 -> Layer2
    z2 = theta1 @ X.T
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, 1, axis=0)

    # Layer2 -> Layer3 (Output)
    z3 = theta2 @ a2
    a3 = sigmoid(z3)

    return a3


def predict(X, theta1, theta2):
    output = feedforward(X, theta1, theta2)
    labels = np.argmax(output, axis=0).reshape((-1, 1)) + 1 # digits are labeled from 1 to 10

    return labels


def load_data():
    data = loadmat('data/ex3data1.mat')
    X = data['X']
    y = data['y']
    X = np.insert(X, 0, 1, axis=1)
    # y[y == 10] = 0 # label digit '0' as '0', not as '10'
    
    return X, y


def load_weights():
    weights = loadmat('data/ex3weights.mat')
    theta1 = weights['Theta1']
    theta2 = weights['Theta2']

    return theta1, theta2


def main():
    X, y = load_data()

    theta1, theta2 = load_weights()
    pred = predict(X, theta1, theta2)
    print('\nTraining Set Accuracy: {0}%\n'.format(np.mean((pred == y) * 1) * 100))


if __name__ == "__main__":
    main()