from scipy.linalg import eigh
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib import pyplot as plt


def cov_ellipse(x, y, cov, num_sigma=1):
    """
    Create an ellipse at the coordinates (x,y), that represents the
    covariance. 

    Returns
    -------
    ellipse: matplotlib.patches.Ellipse
    """

    # length^2 and orientation of ellipse axes is determined by
    # eigenvalues and vectors, respectively. Eigh is more stable for
    # symmetric / hermitian matrices.
    values, vectors = eigh(cov)

    width, height = np.sqrt(values) * num_sigma
    orientation = vectors[:, 0]

    angle = np.arctan2(orientation[1], orientation[0])

    return Ellipse(
        [x, y],
        width=width,
        height=height,
        angle=angle * 180 / np.pi,
        facecolor=(0.1, 0.1, 0.1, 0.2),
        edgecolor="k",
    )


def draw_ellipses(ax, xs, ys, covs, num_sigma=1):
    for x, y, cov in zip(xs, ys, covs):
        ax.add_patch(cov_ellipse(x, y, cov, num_sigma))


def plot_scatter_with_ellipses(xs, ys, covs, num_sigma=1):
    plt.scatter(xs, ys, marker=".")
    draw_ellipses(plt.gca(), xs, ys, covs, num_sigma=num_sigma)
