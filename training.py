import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def model(X, theta):
    return theta[0] + (theta[1] * X)


def cost_function(X, y, theta):
    m = len(y)
    predictions = model(X, theta)
    return 1 / (2 * m) * np.sum((predictions - y) ** 2)


def gradient_descent(X, y, theta, learning_rate, iterations=1000):
    m = len(y)
    cost_history = []
    theta_history = [theta.copy()]
    for i in range(iterations):
        predictions = model(X, theta)
        cost_history.append(cost_function(X, y, theta))
        theta[0] -= learning_rate * (1 / m) * np.sum(predictions - y)
        theta[1] -= learning_rate * (1 / m) * np.sum((predictions - y) * X)
        theta_history.append(theta.copy())

    with open('trained_theta.npy', 'wb') as file:
        np.save(file, theta)
    return theta, cost_history, theta_history

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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))


        ax1.scatter(denormalize_data(X_scaled, X.min(), X.max()), denormalize_data(y_scaled, y.min(), y.max()), color='blue', label='Data')
        ax1.plot(denormalize_data(X_scaled, X.min(), X.max()), denormalize_data(predictions, y.min(), y.max()), color='red', label='Linear Regression')
        ax1.set_title("Prediction")
        ax1.set_xlabel('Mileage')
        ax1.set_ylabel('Price')
        ax1.legend()
        
        ax2.plot(np.arange(0, iterations), cost_history)
        ax2.set_title("Training Cost history")
        ax2.set_ylabel("Errors")
        ax2.set_xlabel('Training iterations')

        plt.tight_layout()
        plt.show()
    else:
        return


def animate_training(X, y, X_scaled, y_scaled, theta_history, cost_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    num_frames = min(100, len(theta_history))
    frame_indices = np.linspace(0, len(theta_history) - 1, num_frames, dtype=int)

    # Pre-calculate data for each frame
    predictions_history = [model(X_scaled, theta_history[i]) for i in frame_indices]
    denorm_predictions_history = [denormalize_data(pred, y.min(), y.max()) for pred in predictions_history]

    line1, = ax1.plot([], [], 'r-', lw=2)
    scatter1 = ax1.scatter(denormalize_data(X_scaled, X.min(), X.max()),
                           denormalize_data(y_scaled, y.min(), y.max()),
                           color='blue', s=10)
    ax1.set_xlabel('Mileage')
    ax1.set_ylabel('Price')
    ax1.set_title('Linear Regression Progress')

    line2, = ax2.plot([], [], 'b-', lw=2)
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Cost')
    ax2.set_title('Cost History')

    ax1.set_xlim(X.min(), X.max())
    ax1.set_ylim(y.min(), y.max())
    ax2.set_xlim(0, len(cost_history))
    ax2.set_ylim(min(cost_history), max(cost_history))

    def update(frame):
        i = frame_indices[frame]
        line1.set_data(X, denorm_predictions_history[frame])
        line2.set_data(range(i), cost_history[:i])
        return line1, line2

    anim = animation.FuncAnimation(fig, update, frames=num_frames,
                                   blit=True, interval=50)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    path = 'data.csv'
    data = pd.read_csv(path)

    X = np.array(data['km']).flatten()
    y = np.array(data['price']).flatten()

    X_scaled, y_scaled = normalize_data(X, y)

    theta = np.zeros(2)

    iterations = 1000
    learning_rate = 0.1

    theta_final, cost_history, theta_history = gradient_descent(X_scaled, y_scaled, theta, learning_rate, iterations)

    print(f"Final theta: {theta_final}")
    predictions = model(X_scaled, theta_final)
    print(f"R2 score: {my_r2_score(y_scaled, predictions)}")

    animate_training(X, y, X_scaled, y_scaled, theta_history, cost_history)
