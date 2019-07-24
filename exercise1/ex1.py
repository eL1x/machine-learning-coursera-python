import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def h(X, theta):
    """
    Hypothesis function.
    Vectorized version:
    h(X) = X * theta
    """
    return X @ theta


def compute_cost(X, y, theta):
    """
    Compute cost function. 
    Vectorized version:
    J ;= 1/2m (X*theta - y).T (x*theta - y)
    """
    m = y.shape[0] # number of examples
    error = h(X, theta) - y
    J = 1/(2 * m) * (error.T @ error)

    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn 'theta'.
    Vectorized version:
    theta := theta - alpha/m * X.T * (X * theta - y)
    """
    m = y.shape[0]
    cost = np.zeros(num_iters)

    for i in range(num_iters):
        theta = theta - (alpha / m) * (X.T @ (X @ theta - y))
        cost[i] = compute_cost(X, y, theta)

    return theta, cost


def load_data():
    """
    Load training data from file.
    """
    data = pd.read_csv('data/ex1data1.txt', header=None, names=['Population', 'Profit'])
    data.insert(0, 'Ones', 1)

    # Set X (training data) and y (target values)
    X = data[['Ones', 'Population']]
    y = data['Profit']

    # Convert to numpy array
    X = np.array(X.values)
    y = np.array(y.values).reshape((-1, 1))

    return X, y


def plot_data(X, y):
    """
    Plot training data.
    """
    plt.scatter(X, y, label='Training data')
    plt.xlabel('Profit')
    plt.ylabel('Population')


def plot_all(X, y, theta_o, cost, iterations):
    """
    Plot data with regression line.
    Plot cost function against iterations.
    """
    # Plot training data and hypothesis for optimal theta
    plt.figure()
    plot_data(X[:, 1], y)
    plt.plot(X[:, 1], h(X, theta_o), 'r', label='Prediction')
    plt.title('Predicted profit vs. Population')
    plt.legend(loc=2)
    
    # Plot values of cost function agains iterations
    plt.figure()
    plt.plot(np.arange(1, iterations+1), cost, 'r')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost function vs. Iterations')

    # Show all figures
    plt.show()


def main():
    X, y = load_data()
    theta = np.zeros((2, 1))

    # Set params and run gradient descent
    iterations = 1500
    alpha = 0.01
    theta, cost = gradient_descent(X, y, theta, alpha, iterations)

    plot_all(X, y, theta, cost, iterations)


if __name__ == "__main__":
    main()