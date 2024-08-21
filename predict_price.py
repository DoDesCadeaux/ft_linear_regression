import numpy as np
import pandas as pd

def normalize_data(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)


def denormalize_data(normalized_value, min_val, max_val):
    return normalized_value * (max_val - min_val) + min_val


def estimator(mileage, trained_theta, x, y):
    normalized_mileage = normalize_data(mileage, x.min(), x.max())
    normalized_estimation = trained_theta[1] + (normalized_mileage * trained_theta[0])
    return denormalize_data(normalized_estimation, y.min(), y.max())


if __name__ == "__main__":
    path = 'data.csv'
    data = pd.read_csv(path)
    
    x = data['km']
    y = data['price']
    
    try:
        mileage = float(input('Enter a mileage to estimate the price:\n'))
    except ValueError as e:
        print(e)
        exit(1)
    
    try:
        with open('trained_theta.npy', 'rb') as file:
            trained_theta = np.load(file)
    except FileNotFoundError as e:
        print(e)
        exit(1)
        
    estimation = estimator(mileage, trained_theta, x, y)
    try:
        print(f"Estimation price for {mileage} km is {estimation[0]:.2f}$")
    except IndexError as e:
        print("Theta is not trained or invalid")

    theta = np.array([0, 0])
    theta = theta.reshape(theta.shape[0], 1)
    with open('trained_theta.npy', 'wb') as file:
        np.save(file, theta)