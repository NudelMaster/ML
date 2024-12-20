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
    n_samples, n_features = data.shape
    w = np.zeros(n_features)
    max_norm = 1e5
    random_indices = np.random.choice(n_samples, size=T, replace=True)
    for t in range(1,T+1):
        idx = random_indices[t-1]
        x_i = data[idx]
        y_i = labels[idx]
        eta_t = eta_0/t
        cond = y_i * np.dot(w, x_i)
        if cond < 1:
            w = (1-eta_t) * w + eta_t * C * y_i * x_i
        else:
            w = (1-eta_t) * w

        # overflow management
        norm_w = np.linalg.norm(w)
        if norm_w > max_norm:
            w = w * (max_norm / norm_w)
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
def cross_validate_best_C(train_data, train_labels, validation_data, validation_labels, C_values, T, eta_0, num_runs):
    """
    Performs cross-validation to find the best C for SGD_hinge.
    """
    average_accuracies = {}
    average_accuracies_plt = []
    for C in C_values:
        accuracies = []
        for run in range(num_runs):

            w = SGD_hinge(train_data, train_labels, C, eta_0, T)
            decision = np.dot(validation_data, w)
            predictions = np.where(decision >= 0, 1, -1)
            acc = calc_accuracy(predictions, validation_labels)
            accuracies.append(acc)

        avg_accuracy = np.mean(accuracies)
        average_accuracies[C] = avg_accuracy
        average_accuracies_plt.append(avg_accuracy)

    best_C = max(average_accuracies, key=average_accuracies.get)
    best_accuracy = average_accuracies[best_C]
    return best_C, best_accuracy, average_accuracies_plt



def plot_validation_accuracy(values, x_labels, label, average_accuracies_plt):
    plt.figure(figsize=[12, 8])
    plt.xlabel(f'{x_labels[label]}')
    plt.ylabel('Average validation accuracy')
    plt.title(f'SGD Hinge Loss: Validation Accuracy vs. ${x_labels[label]}$', fontsize=18)
    plt.semilogx(values, average_accuracies_plt, marker='o', linestyle='-', color='b', label='Average Accuracy')
    if label == 0:
        plt.ylim(0.0,1.0)
    else:
        plt.ylim(min(average_accuracies_plt)-0.01, 1.0)

    plt.show()

def calc_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    return correct_predictions / len(y_true)

def find_best_eta_0(eta_0_values, train_data, train_labels, validation_data, validation_labels):
    C = 1.0
    T = 1000
    num_runs = 10
    best_eta0, best_accuracy, average_accuracies = cross_validate_best_eta_0(train_data, train_labels, validation_data,
                                                                             validation_labels, C, T, eta_0_values,
                                                               num_runs)
    return best_eta0, best_accuracy, average_accuracies
def find_best_C(C_values, best_eta_0, train_data, train_labels, validation_data, validation_labels):

    T = 1000
    num_runs = 10
    best_C, best_accuracy, average_accuracies = cross_validate_best_C(train_data, train_labels, validation_data,
                                                                             validation_labels, C_values, T, best_eta_0,
                                                                             num_runs)
    return best_C, best_accuracy, average_accuracies

def visualize_w_final(w):
    w = np.reshape(w, (28, 28))
    plt.figure(figsize=[6, 6])
    plt.imshow(w, cmap = 'seismic', interpolation = 'nearest')
    plt.colorbar()
    plt.title('Weight vector w')
    plt.xlabel('Pixel Column Index (0-27)')
    plt.ylabel('Pixel Row Index (0-27)')

    tick_positions = np.arange(0, 28, 5)

    plt.xticks(tick_positions, labels=tick_positions)


    plt.yticks(tick_positions, labels=tick_positions)

    plt.show()
def main():

    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    #
    # x_labels = ["eta_0", "C"]
    # eta_0_values = [10**a for a in range(-5,5,1)]
    #
    # # calculate best eta0 over values ranging [-10^5,-10^4,..10^4]
    # best_eta0, best_accuracy, average_accuracies = find_best_eta_0(eta_0_values, train_data, train_labels, validation_data, validation_labels)
    # # Plot accuracy of validation over eta0 values
    #
    # plot_validation_accuracy(eta_0_values, x_labels, 0, average_accuracies)
    # print("best eta0 is ", best_eta0, "best accuracy is ", best_accuracy)
    #
    # C_values = eta_0_values
    #
    # # calculate best C values using best eta_0
    # best_C, best_accuracy, average_accuracies = find_best_C(C_values, best_eta0, train_data, train_labels, validation_data, validation_labels)
    #
    # # Plot accuracy of validation over C values
    # plot_validation_accuracy(C_values, x_labels, 1, average_accuracies)
    # print("Best C is ", best_C, "best accuracy is ", best_accuracy)
    #
    # # calculate final w using best_C and best_eta
    # w_final = SGD_hinge(train_data, train_labels, best_C, best_eta0, 20000)
    # print("Min value of w is ", np.min(w_final), "Max value of w is ", np.max(w_final))
    #
    # visualize_w_final(w_final)
    #
    # # Test accuracy over final w value
    #
    # predictions_test = np.where(np.dot(test_data, w_final) > 0, 1, -1)
    # test_accuracy = calc_accuracy(test_labels, predictions_test)
    # print("Test accuracy is ", test_accuracy, "w final is ", w_final)

if __name__ == '__main__':
    main()

#################################
