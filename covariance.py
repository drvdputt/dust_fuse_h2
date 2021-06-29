from scipy.linalg import eigh
import numpy as np
from math import sqrt
from matplotlib.patches import Ellipse
from matplotlib import pyplot as plt


def cov_ellipse(x, y, cov, num_sigma=1, **kwargs):
    """
    Create an ellipse at the coordinates (x,y), that represents the
    covariance. The style of the ellipse can be adjusted using the
    kwargs.

    Returns
    -------
    ellipse: matplotlib.patches.Ellipse
    """

    position = [x, y]

    if cov[0, 1] != 0:
        # length^2 and orientation of ellipse axes is determined by
        # eigenvalues and vectors, respectively. Eigh is more stable for
        # symmetric / hermitian matrices.
        values, vectors = eigh(cov)
        width, height = np.sqrt(np.abs(values)) * num_sigma
        orientation = vectors[:, 0]
        angle = np.arctan2(orientation[1], orientation[0])

        if width / x > 1000:
            pass

    else:
        width = sqrt(cov[0, 0])
        height = sqrt(cov[1, 1])
        angle = 0

    return Ellipse(
        position, width=width, height=height, angle=angle * 180 / np.pi, **kwargs
    )


def draw_ellipses(ax, xs, ys, covs, num_sigma=1, **kwargs):
    for x, y, cov in zip(xs, ys, covs):
        ax.add_patch(cov_ellipse(x, y, cov, num_sigma, **kwargs))


def plot_scatter_with_ellipses(ax, xs, ys, covs, num_sigma, **scatter_kwargs):
    scatter_kwargs.setdefault("marker", ".")
    pathcollection = ax.scatter(xs, ys, **scatter_kwargs)

    # use same color for point and ellipse
    color = pathcollection.get_facecolors()[0]
    alpha = color[3]
    ellipse_kwargs = {"facecolor": color, "edgecolor": color, "alpha": 0.3 * alpha}
    draw_ellipses(plt.gca(), xs, ys, covs, num_sigma=num_sigma, **ellipse_kwargs)


def get_cov_x_ydivx(x, y, xerr, yerr):
    """Cov(x, ydivx)
    """
    pass
