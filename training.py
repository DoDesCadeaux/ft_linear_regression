import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

def model(X, theta):
    return X.dot(theta)

def cost_function(X, y, theta):
    m = len(y)
    return 1 / (2 * m) * np.sum((model(X, theta) - y) ** 2)

def gradient(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X, theta) - y) 

def gradient_descent(X, y, theta, learning_rate, iterations=1000):
    cost_history = []
    for i in range(0, iterations):
        cost_history.append(cost_function(X, y, theta))
        theta = theta - learning_rate * gradient(X, y, theta)

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

def my_r2_score(y_hat, y):
    SSR = np.sum((y - y_hat) ** 2)
    SST = np.sum((y - y.mean()) ** 2)
    return 1 - (SSR/SST)


path = 'data.csv'
data = pd.read_csv(path)

x = np.array(data['km'])
y = np.array(data['price'])

x = x.reshape(x.shape[0], 1)
y = y.reshape(y.shape[0], 1)

x_scaled, y_scaled = normalize_data(x, y)
X = np.hstack((x_scaled, np.ones(x.shape)))

theta = np.array([0, 0])
theta = theta.reshape(theta.shape[0], 1)


iterations = 1200
theta_final, cost_history = gradient_descent(X, y_scaled, theta, 0.1, iterations)

print(theta_final)

predictions = model(X, theta_final)

def plot_or_not():
    try:
        user_input = input("Do you want to plot the results ? (y/n)\n")
        if user_input != 'y' and user_input != 'n':
            raise TypeError("Invalid input")
    except TypeError as e:
        print(e)
        return -1
    if user_input == 'y':
            plt.scatter(denormalize_data(x_scaled, x.min(), x.max()), denormalize_data(y_scaled, y.min(), y.max()))
            plt.plot(denormalize_data(x_scaled, x.min(), x.max()), denormalize_data(predictions, y.min(), y.max()), color='red')
            plt.show()
            
            plt.plot(np.arange(0, iterations), cost_history)
            plt.show()
    else:
        return

print(my_r2_score(predictions, y_scaled))
print(r2_score(y_scaled, predictions))
plot_or_not()