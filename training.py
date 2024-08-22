import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def model(X, theta):
    return theta[0] + (theta[1] * X)


def cost_function(X, y, theta):
    m = len(y)
    predictions = model(X, theta)
    return 1 / (2 * m) * np.sum((predictions - y) ** 2)


def gradient_descent(X, y, theta, learning_rate, iterations=1000):
    m = len(y)
    cost_history = []
    for i in range(0, iterations):
        cost_history.append(cost_function(X, y, theta))
        predictions = model(X, theta)
        theta[0] -= learning_rate * (1/m) * np.sum(predictions - y)
        theta[1] -= learning_rate * (1/m) * np.sum((predictions - y) * X)
    
    with open('trained_theta.npy', 'wb') as file:
        np.save(file, theta)
    return theta, cost_history


def normalize_data(x, y):
    normalized_feature, normalized_target = np.ones(x.shape), np.ones(y.shape)
    for i in range(0, len(x)):
        normalized_feature[i] = (x[i] - x.min()) / (x.max() - x.min())
        normalized_target[i] = (y[i] - y.min()) / (y.max() - y.min())
    return normalized_feature, normalized_target


def denormalize_data(normalized_value, min_val, max_val):
    return normalized_value * (max_val - min_val) + min_val


def my_r2_score(y, y_hat):
    SSR = np.sum((y - y_hat) ** 2)
    SST = np.sum((y - y.mean()) ** 2)
    return 1 - (SSR/SST)


def plot_or_not():
    try:
        user_input = input("Do you want to plot the results ? (y/n)\n")
        if user_input != 'y' and user_input != 'n':
            raise TypeError("Invalid input")
    except TypeError as e:
        print(e)
        return -1
    if user_input == 'y':
            plt.scatter(denormalize_data(X_scaled, X.min(), X.max()), denormalize_data(y_scaled, y.min(), y.max()))
            plt.plot(denormalize_data(X_scaled, X.min(), X.max()), denormalize_data(predictions, y.min(), y.max()), c='r')
            plt.show()
            
            plt.plot(np.arange(0, iterations), cost_history)
            plt.show()
    else:
        return


if __name__ == "__main__":
    path = 'data.csv'
    data = pd.read_csv(path)
    
    X = np.array(data['km']).flatten()
    y = np.array(data['price']).flatten()
    
    X_scaled, y_scaled = normalize_data(X, y)
    
    theta = np.zeros(2)
    
    iterations = 20000
    theta_final, cost_history = gradient_descent(X_scaled, y_scaled, theta, 0.01, iterations)
    print(theta_final)
    
    predictions = model(X_scaled, theta_final)
    print(my_r2_score(y_scaled, predictions))
    plot_or_not()
