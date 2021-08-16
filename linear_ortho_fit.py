import numpy as np
from scipy import optimize
import itertools
from astropy.modeling import models, fitting
from covariance import cov_ellipse


def perp(m):
    """Vector perpendicular to a line with slope m"""
    return np.array([-m, 1]) / np.sqrt(1 + m * m)


def dperp(m):
    """Derivative of perp"""
    m2 = m * m
    diag2 = 1 + m2
    root_m12 = 1 / np.sqrt(1 + m2)  # ^-1/2
    root_m32 = root_m12 / diag2  # ^-3/2
    return np.array([-root_m12 + m2 * root_m32, -m * root_m32])


def deltas(xy, m, b_perp):
    """Perpendicular distance for each point, to the line y = mx + b"""
    # reminder: b_perp = b cos(theta) = b / sqrt(1 + m^2)
    return xy.dot(perp(m)) - b_perp


def grad_deltas(xy, m, b_perp):
    """Gradient of Delta_i with respect to m, b_perp, indexed on i, p"""
    # vT . Zj
    dD_dm = np.einsum("i,ji->j", dperp(m), xy)
    dD_dbperp = -np.ones(len(xy))

    return np.column_stack([dD_dm, dD_dbperp])


def sigma2s(covs, m):
    """Projection of the covariance matrix, perpendicular to y = mx + b"""
    v = perp(m)
    # vT C v
    return np.einsum("i,kij,j->k", v, covs, v)


def grad_sigma2s(covs, m):
    """Gradient of Sigma2_i with respect to m, b_perp, indexed on i, p"""
    dS2_dbperp = np.zeros(len(covs))

    # 2 deriv(vT) C v
    dS2_dm = 2 * np.einsum("i,kij,j->k", dperp(m), covs, perp(m))
    return np.column_stack([dS2_dm, dS2_dbperp])


def logL(m, b, xy, covs):
    """
    Log likelihood function from Hogg et al. (2010), using the
    perpendicular distance and covariance.
    """
    square_devs = np.square(deltas(xy, m, b)) / sigma2s(covs, m)
    return -0.5 * square_devs.sum()


def grad_logL(m, b, xy, covs):
    """
    Calculate gradient of logL analytically, with respect to (m, b)
    """
    D = deltas(xy, m, b)
    S2 = sigma2s(covs, m)
    D_S2 = D / S2
    # p stands for parameter m or b_perp
    grad_logL = -(
        np.einsum("i,ip->p", D_S2, grad_deltas(xy, m, b))
        - np.einsum("i,ip->p", 0.5 * np.square(D_S2), grad_sigma2s(covs, m))
    )
    return grad_logL


def hess_logL(m, b, xy, covs):
    """
    While doing this analytically is straightforward, it is a lot more
    work. Use finite difference approximation for now, and only use this
    function to estimate the error / cov on the likelihood function.

    # finite difference for hess
    # https://pages.mtu.edu/~msgocken/ma5630spring2003/lectures/diff/diff/node6.html
    """
    # displacements used
    em = np.array([1, 0])
    eb = np.array([0, 1])
    h = [m * 1e-6, b * 1e-6]
    d = [em * h[0], eb * h[1]]

    # 2d function
    def f(r):
        return logL(r[0], r[1], xy, covs)

    r0 = np.array([m, b])
    fr0 = f(r0)

    def d2_d1d2(i, j):
        return (f(r0 + d[i] + d[j]) - f(r0 + d[i]) - f(r0 + d[j]) + fr0) / h[i] * h[j]

    return np.array([[d2_d1d2(0, 0), d2_d1d2(0, 1)], [d2_d1d2(0, 1), d2_d1d2(1, 1)]])


def linear_ortho_maxlh(
    data_x, data_y, cov_xy, ax=None, basic_print=True, debug_print=False
):
    """Do a linear fit based on orthogonal distance, to data where each
    point can have a different covariance between x and y. Uses the
    likelihood function from Hogg et al. (2010). The likelihood is
    maximized using the default scipy.optmize minizer, and the errors
    are estimated using the inverse Hessian of the likelihood at its
    maximum, as provided by the optimized result object.

    IMPORTANT NOTE: the chi2 analog used is not the typical Delta.T *
    C-1 * Delta, with Delta being the distance between the model point
    and the observed point. In fact, the measurment could come from any
    point on the line. Instead, a 1D model is used, which treats the
    orthogonal displacement from the line as the random variable.

    The equation used here is Delta.perp^2 / (v^T * C * v), where
    Delta.perp and v are the orthogonal distance and direction (unit
    vector), with respect to the mx + b line. (v^T * C * v) is the
    correct expression for Delta.perp. In the eigen basis of the
    covariance ellipse, Delta.perp = Delta.x * cos(theta) + Delta.y *
    sin(theta), and Delta.x and Delta.y are independent. Therefore,
    V(Delta.perp) = cos^2(theta) V(Delta.x) + sin^2(theta) V(Delta.y),
    with theta the angle between the normal and the y-eigenvector.

    Returns
    -------
    m: slope
    b_perp: perpendicular intercept (b / sqrt(1 + m^2))
    sigma_m: estimate of error on slop
    sigma_b: estimate of error on intercept
    rho_mb: estimate of pearson coefficient between m and b

    """
    xy = np.column_stack((data_x, data_y))
    # choose a scale factor to avoid the problems that come with the
    # huge dynamic range difference between certain x and y data.
    factor_x = 1 / np.std(data_x)
    factor_y = 1 / np.std(data_y)
    xy, covs = rescale_data(xy, cov_xy, factor_x, factor_y)

    def to_minimize(v):
        m, b_perp = v
        return -logL(m, b_perp, xy, covs)

    def jac(v):
        m, b_perp = v
        return -grad_logL(m, b_perp, xy, covs)

    # use naive result as initial guess
    line_init = models.Linear1D()
    fit = fitting.LinearLSQFitter()
    fitted_model_weights = fit(
        line_init, xy[:, 0], xy[:, 1], weights=1.0 / np.sqrt(covs[:, 1, 1])
    )

    initial_guess = np.array(
        [fitted_model_weights.slope.value, fitted_model_weights.intercept.value,]
    )
    initial_guess[1] = b_to_b_perp(*initial_guess)

    if debug_print:
        eps = np.abs(1.0e-6 * initial_guess)
        grad_approx = optimize.approx_fprime(initial_guess, to_minimize, eps)
        grad_exact = jac(initial_guess)
        print("initial guess: ", initial_guess)
        print("grad approx", grad_approx, "grad exact", grad_exact)

    gtol = np.linalg.norm(jac(initial_guess)) * 1e-6
    res = optimize.minimize(
        to_minimize,
        initial_guess,
        method="BFGS",
        jac=jac,
        options={"disp": False, "gtol": gtol},
    )
    m, b_perp = res.x

    if debug_print:
        print(res)
        # manual check for convergence
        logL0 = logL(m, b_perp, xy, covs)
        logL_up = logL(m + 0.1 * abs(m), b_perp, xy, covs)
        logL_down = logL(m - 0.1 * abs(m), b_perp, xy, covs)
        logL_right = logL(m, b_perp + 0.1 * abs(b_perp), xy, covs)
        logL_left = logL(m, b_perp - 0.1 * abs(b_perp), xy, covs)

        string = """ local logL:
up {}
left {} middle {} right {}
down {}
""".format(
            logL_up, logL_left, logL0, logL_right, logL_down
        )
        print(string)

    # some attempts at estimating the covariance using the hessian
    # not all minimize methods compute the inverse hessian
    # if hasattr(res, "hess_inv"):
    #     hess_inv = res.hess_inv
    # else:
    #     hess = -hess_logL(m, b_perp, xy, covs)
    #     hess_inv = np.linalg.inv(hess)

    # sigma_m, sigma_b_perp = np.sqrt(np.diag(hess_inv))
    # rho_mb_perp = hess_inv[0, 1] / (sigma_m * sigma_b_perp)

    b = b_perp_to_b(m, b_perp)
    if debug_print:
        print("Scaled solution")
        print("m, b_perp:", m, b_perp)
        print("m, b:", m, b)

    # undo the scale factors to get the m and b for the real data
    m_real, b_real = unscale_mb(m, b, factor_x, factor_y)
    b_perp_real = b_real / np.sqrt(1 + m_real * m_real)

    if basic_print:
        print("De-scaled solution")
        print("m, b_perp:", m_real, b_perp_real)
        print("m, b:", m_real, b_real)

    return m_real, b_perp_real


def bootstrap_fit_errors(data_x, data_y, cov_xy):
    """
    Simple bootstrap of linear_ortho_maxlh. Runs the fitting a set
    number of times and prints out empirical covariances between the
    parameters.

    Returns
    -------

    cov: estimate of covariance matrix of m, b

    """
    N = len(data_x)
    M = 100
    ms = np.zeros(M)
    bs = np.zeros(M)
    for m in range(M):
        idxs = np.random.choice(range(N), size=N)
        boot_x = data_x[idxs]
        boot_y = data_y[idxs]
        boot_cov = cov_xy[idxs]
        (ms[m], bs[m]) = linear_ortho_maxlh(boot_x, boot_y, boot_cov, basic_print=False)

    print("Bootstrap: m = {} ; b = {}".format(np.average(ms), np.average(bs)))
    print("Bootstrap: sm = {} ; sb = {}".format(np.std(ms), np.std(bs)))
    print("Bootstrap: corr(m, b) = {}".format(np.corrcoef(ms, bs)[0, 1]))
    cov = np.cov(ms, bs)
    return cov


def plot_solution_linescatter(ax, m, b_perp, cov_mb, num_lines, **plot_kwargs):
    """Visualize the spread on the linear fits using transparent plots.

    Currently approximated as gaussian. Should properly sample the likelihood
    function later.

    """
    mb_samples = np.random.multivariate_normal((m, b_perp), cov_mb, num_lines)
    x = np.array(ax.get_xlim())
    for (m, b_perp) in mb_samples:
        y = m * x + b_perp_to_b(m, b_perp)
        ax.plot(x, y, **plot_kwargs)


def plot_solution_neighborhood(
    ax, m, b, xs, ys, covs, cov_mb=None, area=None, extra_points=None, what="logL"
):
    """
    Color plot of the 2D likelihood function around the given point (m,
    b)

    cov_mb: covariance matrix used to draw an ellipse, so we can see if
    it makes sense

    area: the area over which the function should be plotted, defined as
    [mmin, mmax, bmin, bmax]

    extra_points: points to be plotted on the image, in addition to m, b

    what: choose 'logL', 'L'

    """
    if area is None:
        f = 16
        mmin = m - f * abs(m)
        mmax = m + f * abs(m)
        bmin = b - f * abs(b)
        bmax = b + f * abs(b)
    else:
        mmin, mmax, bmin, bmax = area

    xy = np.column_stack([xs, ys])
    res = 400
    grid_m = np.linspace(mmin, mmax, res)
    grid_b = np.linspace(bmin, bmax, res)
    image = np.zeros((len(grid_m), len(grid_b)))
    for ((i, mi), (j, bj)) in itertools.product(enumerate(grid_m), enumerate(grid_b)):
        image[i, j] = logL(mi, bj, xy, covs)

    imshow_kwargs = dict(
        extent=[bmin, bmax, mmin, mmax], origin="lower", aspect="auto", cmap="viridis"
    )

    if what == "logL":
        im = ax.imshow(image, **imshow_kwargs)
    elif what == "L":
        im = ax.imshow(np.exp(image), **imshow_kwargs)
    else:
        print("Wrong string, no plot made")

    ax.figure.colorbar(im, ax=ax)
    ax.set_ylabel("m")
    ax.set_xlabel("b")
    ax.plot(b, m, "kx")

    if extra_points is not None:
        for (bi, mi) in extra_points:
            ax.plot(b, m, "k+")

    if cov_mb is not None:
        ax.add_patch(
            cov_ellipse(b, m, cov_mb[::-1, ::-1], facecolor="none", edgecolor="k")
        )


def rescale_data(xy, covs, factor_x, factor_y):
    S = np.array([[factor_x, 0], [0, factor_y]])
    xyr = np.einsum("ij,dj", S, xy)
    covr = np.einsum("ij,djk,kl", S.T, covs, S)
    return xyr, covr


def unscale_mb(m, b, factor_x, factor_y):
    # x * factor_x * m = y * factor_y --> y / x = ...
    m_real = m * factor_x / factor_y
    # b = y * factor_y --> y = ...
    b_real = b / factor_y
    return m_real, b_real


def b_perp_to_b(m, b_perp):
    # reminder: b_perp = b cos(theta) = b / sqrt(1 + m^2)
    return b_perp * np.sqrt(1 + m * m)


def b_to_b_perp(m, b):
    return b / np.sqrt(1 + m * m)
