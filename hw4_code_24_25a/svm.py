import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def plot_results(models, titles, X, y, plot_sv=False):
    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(1, len(titles))  # 1, len(list(models)))

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    if len(titles) == 1:
        sub = [sub]
    else:
        sub = sub.flatten()
    for clf, title, ax in zip(models, titles, sub):
        # print(title)
        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        if plot_sv:
            sv = clf.support_vectors_
            ax.scatter(sv[:, 0], sv[:, 1], c='k', s=60)

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.show()

def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out



C = 10
n = 100


# Data is labeled by a circle

radius = np.hstack([np.random.random(n), np.random.random(n) + 1.5])
angles = 2 * math.pi * np.random.random(2 * n)
X1 = (radius * np.cos(angles)).reshape((2 * n, 1))
X2 = (radius * np.sin(angles)).reshape((2 * n, 1))

X = np.concatenate([X1,X2],axis=1)
y = np.concatenate([np.ones((n,1)), -np.ones((n,1))], axis=0).reshape([-1])


# a+b sections
clf_linear = svm.SVC(kernel='linear', C = 10)
clf_homogeneous_2 = svm.SVC(kernel = 'poly', degree = 2, C = 10)
clf_homogeneous_3 = svm.SVC(kernel = 'poly', degree = 3, C = 10)
clf_non_homogeneous_2 = svm.SVC(kernel = 'poly', degree = 2, coef0 = 1, C = 10)
clf_non_homogeneous_3 = svm.SVC(kernel = 'poly', degree = 3, coef0 = 1, C = 10)

clf_linear.fit(X,y)
clf_homogeneous_2.fit(X,y)
clf_homogeneous_3.fit(X,y)
clf_non_homogeneous_2.fit(X,y)
clf_non_homogeneous_3.fit(X,y)

plot_results([clf_linear, clf_homogeneous_2, clf_homogeneous_3, clf_non_homogeneous_2, clf_non_homogeneous_3], ['linear', 'homog_2', 'homog_3', 'non_homog_2', 'non_homog_3'], X, y)

# c section
def flip_labels(y):
  negative_indices = np.where(y < 0)[0]
  for i in negative_indices:
    y[i] = -y[i] if np.random.random() <= 0.1 else y[i]
  return y

clf_rbf = svm.SVC(kernel = 'rbf', gamma = 10, C = 10)
y_flipped = flip_labels(y)
clf_non_homogeneous_2.fit(X,y_flipped)
clf_rbf.fit(X,y_flipped)
plot_results([clf_non_homogeneous_2, clf_rbf], ['non_homog_2', 'rbf gamma 10'], X, y_flipped)

# changing plots
gammas = np.arange(1,50,5)
for gamma in gammas:
  clf_rbf = svm.SVC(kernel = 'rbf', gamma = gamma, C = 10)
  clf_rbf.fit(X,y_flipped)
  plot_results([clf_rbf], [f'rbf gamma {gamma}'], X, y_flipped)