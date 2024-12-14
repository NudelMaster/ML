#################################
# Your name: Yefim Nudelman
#################################


import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt
"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels



def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    n = data.shape[1]
    w = np.zeros(n)
    for t in range(1,T+1):
        idx = numpy.random.choice(n)
        x_i = data[idx]
        y_i = labels[idx]
        eta_t = eta_0/t
        cond = y_i * np.dot(w, x_i)
        if cond < 1:
            w = (1-eta_t) * w + eta_t * C * y_i * x_i
        else:
            w = (1-eta_t) * w
    return w


#################################

# Place for additional code
def cross_validate_best_eta_0(train_data, train_labels, validation_data, validation_labels, C, T, eta_0_values, num_runs):
    """
    Performs cross-validation to find the best eta_0 for SGD_hinge.
    """

    average_accuracies = {}
    average_accuracies_plt = []
    for eta_0 in eta_0_values:
        accuracies = []

        for run in range(num_runs):

            w = SGD_hinge(train_data, train_labels, C, eta_0, T)
            decision = np.dot(validation_data, w)
            predictions = np.where(decision >= 0, 1, -1)
            acc = calc_accuracy(predictions, validation_labels)
            accuracies.append(acc)

        avg_accuracy = np.mean(accuracies)
        average_accuracies[eta_0] = avg_accuracy
        average_accuracies_plt.append(avg_accuracy)

    best_eta_0 = max(average_accuracies, key=average_accuracies.get)
    best_accuracy = average_accuracies[best_eta_0]


    return best_eta_0, best_accuracy, average_accuracies_plt

def plot_validation_accuracy(eta_0_values, average_accuracies_plt):
    plt.figure(figsize=[12, 8])
    plt.xlabel('eta_0')
    plt.ylabel('Average validation accuracy')
    plt.title('SGD Hinge Loss: Validation Accuracy vs. $\eta_0$', fontsize=18)
    plt.semilogx(eta_0_values, average_accuracies_plt, marker='o', linestyle='-', color='b', label='Average Accuracy')
    plt.ylim(0.0, 1.0)
    plt.show()

def calc_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    return correct_predictions / len(y_true)

def main():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    C = 1.0
    T = 1000
    eta_0_values = np.logspace(-7, 5, num=11)
    num_runs = 10

    best_eta0, best_accuracy, average_accuracies = cross_validate_best_eta_0(train_data, train_labels, validation_data, validation_labels, C, T, eta_0_values, num_runs)
    plot_validation_accuracy(eta_0_values, average_accuracies)

if __name__ == '__main__':
    main()

#################################
