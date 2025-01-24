import matplotlib.pyplot as plt
import numpy as np
from backprop_network import *
from backprop_data import *

# Loading Data
n_train = 50000
n_test = 10000
x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)





# Training configuration
epochs = 30
batch_size = 100
learning_rate = 0.1
learning_rates = [0.001, 0.01, 0.1, 1, 10]  # Different learning rates



# Network configuration
layer_dims = [784, 40, 10]
#net = Network(layer_dims)
#net.train(x_train, y_train, epochs, batch_size, learning_rate, x_test=x_test, y_test=y_test)

results = {}

for lr in learning_rates:
    print("Learning rate:", lr)
    # Initialize the network
    net = Network(layer_dims)

    # Train the network and capture metrics
    params, train_loss, test_loss, train_acc, test_acc = net.train(
        x_train, y_train, epochs, batch_size, lr, x_test=x_test, y_test=y_test
    )

    # Store the results
    results[lr] = {
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
    }


# Plot results
def plot_metrics(results, metric, ylabel, title):
    plt.figure(figsize=(8, 6))
    for lr, metrics in results.items():
        plt.plot(metrics[metric], label=f"LR={lr}")
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


# Plot Training Accuracy
plot_metrics(results, metric="train_accuracy", ylabel="Training Accuracy", title="Training Accuracy vs Epochs")

# Plot Training Loss
plot_metrics(results, metric="train_loss", ylabel="Training Loss", title="Training Loss vs Epochs")

# Plot Test Accuracy
plot_metrics(results, metric="test_accuracy", ylabel="Test Accuracy", title="Test Accuracy vs Epochs")
