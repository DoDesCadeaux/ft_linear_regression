import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = 'data.csv'
data = pd.read_csv(path)

x = np.array(data['km'])
y = np.array(data['price'])


x = x.reshape(x.shape[0], 1)
y = y.reshape(y.shape[0], 1)

X = np.hstack((x, np.ones(x.shape)))

theta = np.array([0, 0])
theta = theta.reshape(theta.shape[0], 1)

def model(X, theta):
    return X.dot(theta)

print(model(X, theta))

plt.scatter(x, y)
plt.plot(x, model(X, theta), c='red')
plt.show()
