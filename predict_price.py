import numpy as np
import pandas as pd

path = 'data.csv'
data = pd.read_csv(path)

x = data['km']
y = data['price']

def normalize_data(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

def denormalize_data(normalized_value, min_val, max_val):
    return normalized_value * (max_val - min_val) + min_val

try:
    mileage = float(input('Enter a mileage to estimate the price:\n'))
except ValueError as e:
    print(e)

try:
    with open('trained_theta.npy', 'rb') as file:
        trained_theta = np.load(file)
except FileNotFoundError as e:
    print(e)
    exit(1)

def estimator(mileage, trained_theta, x, y):
    normalized_mileage = normalize_data(mileage, x.min(), x.max())
    normalized_estimation = trained_theta[1] + (normalized_mileage * trained_theta[0])
    return denormalize_data(normalized_estimation, y.min(), y.max())
    

print(f"thetas: {trained_theta}")
estimation = estimator(mileage, trained_theta, x, y)
print(estimation)