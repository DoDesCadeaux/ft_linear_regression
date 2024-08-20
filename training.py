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
    return theta, cost_history

def normalize_data(x, y):
    normalized_feature, normalized_target = np.ones(x.shape), np.ones(y.shape)
    for i in range(0, len(x)):
        normalized_feature[i] = (x[i] - x.min()) / (x.max() - x.min())
        normalized_target[i] = (y[i] - y.min()) / (y.max() - y.min())
    return normalized_feature, normalized_target

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

theta_final, cost_history = gradient_descent(X, y, theta, 0.1, 500)

predictions = model(X, theta_final)
plt.scatter(x, y)
plt.plot(x, predictions, color='red')
plt.show()

plt.plot(np.arange(0, 500), cost_history)
plt.show()

print(my_r2_score(predictions, y))
print(r2_score(y, predictions))