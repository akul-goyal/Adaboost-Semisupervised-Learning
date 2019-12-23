import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import testing

from itertools import permutations


def run(x_res,y,clfs):
    # Parameters
    cmap = plt.cm.RdYlBu
    plot_step = 0.02  # fine step width for decision surface contours
    plot_step_coarser = 0.5  # step widths for coarse classifier guesses


    plot_idx = 1


    li = list(range(0, len(x_res[0])))
    perm = permutations(li, 2)
    pairs = (perm)
    for pair in pairs:
        # We only take the two corresponding features
        X = x_res[:, pair]

        # Standardize
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / std


        plt.subplot(3, 4, plot_idx)

        # Now plot the decision boundary using a fine mesh as input to a
        # filled contour plot
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))


        estimator_alpha = 1.0 / len(clfs)
        for tree in clfs:
            tree = tree[0]
            b = np.c_[xx.ravel(), yy.ravel()]
            Z = tree.predict(b)
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

        # Build a coarser grid to plot a set of ensemble classifications
        # to show how these are different to what we see in the decision
        # surfaces. These points are regularly space and do not have a
        # black outline
        # xx_coarser, yy_coarser = np.meshgrid(
        #     np.arange(x_min, x_max, plot_step_coarser),
        #     np.arange(y_min, y_max, plot_step_coarser))
        # Z_points_coarser = model.predict(np.c_[xx_coarser.ravel(),
        #                                  yy_coarser.ravel()]
        #                                  ).reshape(xx_coarser.shape)
        # cs_points = plt.scatter(xx_coarser, yy_coarser, s=15,
        #                         c=Z_points_coarser, cmap=cmap,
        #                         edgecolors="none")

        # Plot the training points, these are clustered together and have a
        # black outline
        plt.scatter(X[:, 0], X[:, 1], c=y,
                    cmap=ListedColormap(['r', 'y', 'b']),
                    edgecolor='k', s=20)
        plot_idx += 1  # move on to the next plot in sequence

    plt.axis("tight")
    plt.tight_layout(h_pad=0.2, w_pad=0.2, pad=2.5)
    plt.show()


#TODO: Need to make classifier trained on two features, not all 30 of them! IE: chnage clfs
x, y, clfs = testing.mlenoiseboost_viz(0.9)
run(x, y, clfs)