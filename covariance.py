from scipy.linalg import eigh
import numpy as np
from math import sqrt, cos, sin
from matplotlib.patches import Polygon
from scipy.stats import multivariate_normal
from rescale import RescaledData
import cmasher


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
        width, height = np.sqrt(np.abs(values))
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
    # one sigma around center
    vw = vectors[:, 0] * width * num_sigma
    vh = vectors[:, 1] * height * num_sigma

    # generate corners
    num_corners = 16
    angles = np.linspace(0, 2 * np.pi, num_corners, endpoint=False)
    corners = np.row_stack([position + vw * cos(a) + vh * sin(a) for a in angles])
    return Polygon(corners, **kwargs)


def draw_ellipses(ax, xs, ys, covs, num_sigma=1, **kwargs):
    for x, y, cov in zip(xs, ys, covs):
        ax.add_patch(cov_ellipse(x, y, cov, num_sigma, **kwargs))


def plot_scatter_auto(ax, xs, ys, covs, num_sigma, **scatter_kwargs):
    """Automatically chooses between error bars or ellipses"""
    if np.any(np.nonzero(covs[:, 0, 1])):
        # if True:
        scatter_kwargs["marker"] = "+"
        # plot_scatter_with_ellipses(ax, xs, ys, covs, num_sigma, **scatter_kwargs)
        plot_scatter_density(ax, xs, ys, covs)
    else:
        scatter_kwargs["marker"] = "o"
        scatter_kwargs["s"] = 12
        plot_scatter_with_errbars(ax, xs, ys, covs, num_sigma, **scatter_kwargs)


def plot_scatter_with_ellipses(ax, xs, ys, covs, num_sigma, **scatter_kwargs):
    """
    Parameters
    ----------
    ax : matplotlib axes

    xs : array of shape (N,) containing x data

    ys : array of shape (N,) containing y data

    cov : array of shape (N, 2, 2)
    """
    scatter_kwargs.setdefault("marker", ".")
    scatter_kwargs.setdefault("alpha", 0.66)
    pathcollection = ax.scatter(xs, ys, **scatter_kwargs)

    # use same color for point and ellipse
    color = pathcollection.get_facecolors()[0]
    ellipse_kwargs = {"facecolor": color, "edgecolor": color, "alpha": 0.075}
    draw_ellipses(ax, xs, ys, covs, num_sigma=num_sigma, **ellipse_kwargs)


def plot_scatter_with_errbars(ax, xs, ys, covs, num_sigma, **scatter_kwargs):
    """
    Draw regular error bars instead of ellipses. Ignores covariance.
    Useful if covariance was zero anyway, in which case ellipses would be too much information anyway.
    """
    scatter_kwargs.setdefault("marker", ".")
    scatter_kwargs.setdefault("alpha", 0.66)
    if "s" in scatter_kwargs:
        scatter_kwargs["ms"] = np.sqrt(
            scatter_kwargs.pop("s")
        )  # / 6.3  # area to radius
    markers, caps, bars = ax.errorbar(
        xs,
        ys,
        ls="none",
        xerr=num_sigma * np.sqrt(covs[:, 0, 0]),
        yerr=num_sigma * np.sqrt(covs[:, 1, 1]),
        elinewidth=1.5,
        **scatter_kwargs
    )
    for bar in bars:
        bar.set_alpha(0.15)


def plot_scatter_density(ax, xs, ys, covs):
    """
    Heat map visualizing the collection of all data points and their covariance ellipses.

    Might be useful for talk or for later projects.
    """
    # rescale to avoid floating point/singular matrix problems
    rd = RescaledData(xs, ys, covs)
    # ranges
    sx = np.sqrt(covs[:, 0, 0])
    sy = np.sqrt(covs[:, 1, 1])
    minx = np.amin(xs - 3 * sx)
    maxx = np.amax(xs + 3 * sx)
    miny = np.amin(ys - 3 * sy)
    maxy = np.amax(ys + 3 * sy)
    # grid. 1000j = complex number, because the mgrid syntax is weird.
    # If you pass a complex integer, you get number of points instead of
    # step length)
    nx = 1000
    ny = 1000
    xx_yy = np.mgrid[minx : maxx : nx * 1j, miny : maxy : ny * 1j]
    print("grid_shape", xx_yy.shape)
    # indexed on (i, j, x or y)
    grid_of_xy_pairs = np.moveaxis(xx_yy, 0, -1)

    # calculating the multivariate normal density is the part where the
    # rescaling is needed. It can struggle with the covariance matrix if
    # the orders of magnitude of the elements are too different
    rescaled_grid = grid_of_xy_pairs.copy()
    rescaled_grid[:, :, 0] *= rd.factor_x
    rescaled_grid[:, :, 1] *= rd.factor_y
    density_array = np.zeros((nx, ny))
    for i in range(len(xs)):
        # calculate using the rescaled version of the problem
        density_array += multivariate_normal.pdf(
            rescaled_grid, mean=rd.xy[i], cov=rd.covs[i]
        )

    # we can choose to rescale the density array here, but probably not necessary for the visualization

    extent = (minx, maxx, miny, maxy)
    print(density_array)

    # cut the ends off the colormap, to make the low end more distinguishable
    # cmap = "cmr.arctic_r"
    cmap = cmasher.get_sub_cmap("cmr.arctic_r", 0, 0.9)

    ax.imshow(
        density_array.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap,
    )
    ax.scatter(xs, ys, marker="+", s=12, color="k", alpha=0.75, linewidth=0.6)
    draw_ellipses(ax, xs, ys, covs, facecolor="none", edgecolor="#00000020", lw=0.5)


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
    Vx = xerr**2
    Vxdy = xdy_err**2
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
    Vx = xerr**2

    # V(y/x) = (dg / dx)**2 sigma_x**2 + (dg / dy)**2 sigma_y**2
    # with g = y/x
    # -> y**2 / x**4 sigma_x**2 + sigma_y**2 / x**2
    # = (y/x)**2 * [ sigma_x**2 / x**2 + sigma_y**2 / y**2 ]
    Vydx = ydx_err**2  # is already calculated in get_data

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
    vx = hi_err**2
    vy = h2_err**2
    cov = (4 * vy * x - 2 * vx * y) / (x + 2 * y) ** 2

    # sigma_x**2 + (2 sigma_y)**2
    vhtot = vx + 4 * vy

    # analogously, vfh2[x_, y_] := vx D[g[x, y], x] D[g[x, y], x] + vy D[g[x, y], y] D[g[x, y], y]
    # Output: 4 (vy x^2 + vx y^2))/(x + 2 y)^4
    vfh2 = 4 * (vy * x**2 + vx * y**2) / (x + 2 * y) ** 4

    return make_cov_matrix(vfh2, vhtot, cov)


def new_cov_when_divide_y(cov_xy, y, A, A_err):
    """Turn cov(x, y) into cov(x / A, y), given x/A and A

    Where A is an independent variable with respect to x and y."""

    # when doing e.g. htot / AV, the covariance gets a factor 1 / AV
    new_cov = cov_xy.copy()
    new_cov[:, 0, 1] /= A
    new_cov[:, 1, 0] /= A

    # fh2 is not affected

    # V(y / A) becomes (y / A)**2 * (Vy / y**2 + sigma_A**2 / A**2)
    Vy = cov_xy[:, 1, 1]
    new_cov[:, 1, 1] = (y / A) ** 2 * (Vy / y**2 + (A_err / A) ** 2)
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
    new_cov[:, 0, 0] = (x / B) ** 2 * (Vx / x**2 + (B_err / B) ** 2)
    return new_cov


def cov_common_denominator(x_a, x_a_unc, y_a, y_a_unc, a, a_unc):
    """Covariance between an x and y of the form x = x0 / A and y = y0 / A,
       where x0, y0 and A are all variables with uncertainties.

    Parameters
    ----------
    x_a: y/A values

    X_a_unc: sigma(X/A)

    y_a: y/A values

    y_a_unc: sigma(y/A)

    a: A

    a_unc: sigma(A)

    AV can be substituted by something else of course.

    Returns
    -------
    cov_matrix: np.array of size(len(x_a), 2, 2)
        covariance matrix
        [[       V(x/A), cov(x/A, y/A)],
         [cov(x/A, y/A),        V(y,A)]]
    """
    return make_cov_matrix(x_a_unc**2, y_a_unc**2, x_a * y_a * a_unc**2 / a**2)
