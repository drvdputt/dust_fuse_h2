from scipy.linalg import eigh
import numpy as np
from math import sqrt, cos, sin
from matplotlib.patches import Polygon


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
    else:
        width = sqrt(cov[0, 0])
        height = sqrt(cov[1, 1])
        vectors = np.array([[1, 0], [0, 1]])

    # I ended up using a Polygon just like Karl's plotting code. The
    # ellipse is buggy when the difference in axes is extreme (1e22). I
    # think it is because even a slight rotation will make the ellipse
    # look extremely strechted, as the extremely long axis (~1e22)
    # rotates into the short coordinate (~1).

    # two vectors representing the axes of the ellipse
    vw = vectors[:, 0] * width / 2
    vh = vectors[:, 1] * height / 2

    # generate corners
    num_corners = 16
    angles = np.linspace(0, 2 * np.pi, num_corners, endpoint=False)
    corners = np.row_stack([position + vw * cos(a) + vh * sin(a) for a in angles])
    return Polygon(corners, **kwargs)


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
    draw_ellipses(ax, xs, ys, covs, num_sigma=num_sigma, **ellipse_kwargs)


def make_cov_matrix(Vx, Vy, covs=None):
    cov_matrix = np.zeros((len(Vx), 2, 2))
    cov_matrix[:, 0, 0] = Vx
    cov_matrix[:, 1, 1] = Vy
    if covs is not None:
        cov_matrix[:, 0, 1] = covs
        cov_matrix[:, 1, 0] = covs
    return cov_matrix


def get_cov_x_xdy(x, xdy, xerr, xdy_err):
    """Covariance matrix for (x, x / y)"""
    Vx = xerr ** 2
    Vxdy = xdy_err ** 2
    # cov(x, x / y) = Vx / y = Vx / x * (x/y)
    cov = (Vx / x) * xdy
    return make_cov_matrix(Vx, Vxdy, cov)


def get_cov_x_ydx(x, ydx, xerr, ydx_err):
    """Covariance matrix for (x, y / x).

    Parameters
    ----------
    x : x values
    ydx : values of y / x
    xerr : values of sigma_x
    ydx_err : values of sigma_(y/x) (as calculated in get_data)

    Returns
    -------
    cov_matrix: ndarray
        covariance matrixes in the shape [len(x), 2, 2]. Indices are
        (data point, xy, xy).
    """
    # V(x) = sigma_x**2 (trivial)
    Vx = xerr ** 2

    # V(y/x) = (dg / dx)**2 sigma_x**2 + (dg / dy)**2 sigma_y**2
    # with g = y/x
    # -> y**2 / x**4 sigma_x**2 + sigma_y**2 / x**2
    # = (y/x)**2 * [ sigma_x**2 / x**2 + sigma_y**2 / y**2 ]
    Vydx = ydx_err ** 2  # is already calculated in get_data

    # cov(x, y/x) = (dx / dx) (dg / dx) Vx + (dx / dy) (dg / dy) Vy
    #             = -(y / x) Vx / x
    cov = -ydx * Vx / x

    # old code (in plot results script) works via correlation coefficient:
    # corr = cov / (sx * sy)
    # after simplifying
    # = -(sigma_x / x) / sqrt(sigma_x**2 / x**2 + sigma_y ** / y**2)
    # = -(sigma_x / x) / sigma_(y/x) / (y/x)
    return make_cov_matrix(Vx, Vydx, cov)


def get_cov_fh2_htot(hi, h2, hi_err, h2_err):
    """Covariance between [nhi + 2 nh2] and [2 nh2 / (nhi + 2 nh2)]"""

    # Should probably note: the implementation in get_data for fh2_unc
    # does not seem correct to me. It simply propagates using nh2_unc
    # and nhtot_unc, without taking into account the covariance between
    # those two parameters.

    # One could either calculate this covariance (probably not too
    # hard), and add a +2*cov(nhtot, nh2) term to fh2_unc. Or one could
    # intermediate steps, and work with the independent variables nhi
    # and nh2 directly. This is what I did.

    # here's the code I put into mathematica to calculate what I need,
    # with x = nhi and y = nh2, and vx and vy the square uncertainties.

    # htot[x_, y_] := x + 2 y
    # fh2[x_, y_] := (2 y)/(x + 2 y)
    # cov[x_, y_] := vx D[f[x, y], x] D[g[x, y], x] + vy D[f[x, y], y] D[g[x, y], y]
    # FullSimplify[cov[x, y]]
    # Output: (4 vy x - 2 vx y)/(x + 2 y)^2
    x = hi
    y = h2
    vx = hi_err ** 2
    vy = h2_err ** 2
    cov = (4 * vy * x - 2 * vx * y) / (x + 2 * y) ** 2

    # sigma_x**2 + (2 sigma_y)**2
    vhtot = vx + 4 * vy

    # analogously, vfh2[x_, y_] := vx D[g[x, y], x] D[g[x, y], x] + vy D[g[x, y], y] D[g[x, y], y]
    # Output: 4 (vy x^2 + vx y^2))/(x + 2 y)^4
    vfh2 = 4 * (vy * x ** 2 + vx * y ** 2) / (x + 2 * y) ** 4

    return make_cov_matrix(vfh2, vhtot, cov)


def new_cov_when_divide_y(cov_xy, y, A, A_err):
    """Turn cov(x, y) into cov(x / A, y), given x/A and A

    Where A is an independent variable with respect to x and y. """

    # when doing e.g. htot / AV, the covariance gets a factor 1 / AV
    new_cov = cov_xy.copy()
    new_cov[:, 0, 1] /= A
    new_cov[:, 1, 0] /= A

    # fh2 is not affected

    # V(y / A) becomes (y / A)**2 * (Vy / y**2 + sigma_A**2 / A**2)
    Vy = cov_xy[:, 1, 1]
    new_cov[:, 1, 1] = (y / A) ** 2 * (Vy / y ** 2 + (A_err / A) ** 2)
    return new_cov


def new_cov_when_divide_x(cov_xy, x, B, B_err):
    """Analogous to the above, but when x is divided by independent data.

    Need to give x/B and B."""
    # when doing e.g. RV = AV / EBV, the covariance gets a factor 1 / EBV
    new_cov = cov_xy.copy()
    new_cov[:, 0, 1] /= B
    new_cov[:, 1, 0] /= B

    # NH_AV is not affected

    # V(RV) = V(x / B) becomes (x / B)**2 * ( (Vx / x)**2 + (VB / B)**2 )
    # (value * relative error)
    Vx = cov_xy[:, 0, 0]
    new_cov[:, 0, 0] = (x / B) ** 2 * (Vx / x ** 2 + (B_err / B) ** 2)
    return new_cov
