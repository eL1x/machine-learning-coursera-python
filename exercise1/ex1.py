import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


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
        theta = theta - (alpha / m) * (X.T @ (h(X, theta) - y))
        cost[i] = compute_cost(X, y, theta)

    return theta, cost


def feature_normalize(X):
    """
    Feature normalization by by subtraction mean and dividing by standard deviation:
    X = (X - mu) / sigma
    """
    X_norm = X.copy()
    num_features = X.shape[1]
    mu = np.zeros(num_features)
    sigma = np.zeros(num_features)

    for i in range(1, num_features):
        mu[i] = np.mean(X[:, i])
        sigma[i] = np.std(X[:, i])
        X_norm[:, i] = (X[:, i] - mu[i]) / sigma[i]

    return X_norm, mu, sigma


def normal_equation(X, y):
    """
    Compute theta from normal equation:
    theta = (X.T * X)^(-1) * X.T * y
    """
    theta = np.linalg.inv(X.T @ X) @ X.T @ y

    return theta


def load_data_single():
    """
    Load training data from file for linear regression with single variable.
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


def load_data_multi():
    """
    Load training data from file for linear regression with multiple variables.
    """
    data = pd.read_csv('data/ex1data2.txt', header=None, names=['Size', 'Bedrooms', 'Price'])
    data.insert(0, 'Ones', 1)

    # Set X (training data) and y (target values)
    X = data[['Ones', 'Size', 'Bedrooms']]
    y = data['Price']

    # Convert to numpy array
    X = np.array(X.values, dtype=np.float64)
    y = np.array(y.values, dtype=np.float64).reshape((-1, 1))

    return X, y


def plot_data(X, y):
    """
    Plot training data.
    """
    if X.shape[1] == 2:
        # For multivariate linear regression
        ax = plt.axes(projection='3d')
        ax.scatter3D(X[:, 0], X[:, 1], y, label='Training data')
        ax.set_xlabel('Size')
        ax.set_ylabel('Bedrooms')
        ax.set_zlabel('Price')
    else:
        # For single variate linear regression
        plt.scatter(X, y, label='Training data')
        plt.xlabel('Profit')
        plt.ylabel('Population')


def plot_regression_line(X, theta_o):
    """
    Plot regression line.
    """
    if X.shape[1] == 2:
        # For multivariate linear regression
        x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        y_vals = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
        vec_vals = np.zeros((100, 3))
        vec_vals[:, 0] = 1
        vec_vals[:, 1] = x_vals
        vec_vals[:, 2] = y_vals
        plt.plot(x_vals, y_vals, zs=h(vec_vals, theta_o).reshape((100,)), label='Prediction', c='r')
    else:
        # For single variate linear regression
        X = np.insert(X, 0, 1, axis=1)
        plt.plot(X[:, 1:], h(X, theta_o), 'r', label='Prediction')
        plt.title('Predicted profit vs. Population')   


def plot_all(X, y, theta_o, cost, iterations):
    """
    Plot data with regression line.
    Plot cost function against iterations.
    """
    # Plot training data and hypothesis for optimal theta
    plt.figure()
    plot_data(X[:, 1:], y)
    plot_regression_line(X[:, 1:], theta_o)
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
    ## Choose between load_data_single() and load_data_multi()
    ## to run singlevariate or multivariate linear regression
    # X, y = load_data_single()
    X, y = load_data_multi()
    theta = np.zeros((X.shape[1], 1))
    X, mu, sigma = feature_normalize(X)

    # Set params and run gradient descent
    iterations = 400
    alpha = 0.1
    theta, cost = gradient_descent(X, y, theta, alpha, iterations)
    theta_eq = normal_equation(X, y)

    # 'theta' and 'theta_eq' are almost equal
    print('theta: \n', theta)
    print('theta_eq: \n', theta_eq)

    # Plotting data with regression line and cost function 
    plot_all(X, y, theta, cost, iterations)


if __name__ == "__main__":
    main()