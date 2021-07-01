import numpy as np
from scipy import optimize
from scipy import stats
import itertools


def perp(m):
    """Vector perpendicular to a line with slope m"""
    return np.array([-m, 1]) / np.sqrt(1 + m * m)


def dperp(m):
    """Derivative of perp"""
    m2 = m * m
    diag2 = 1 + m2
    root_m12 = 1 / np.sqrt(1 + m2)  # ^-1/2
    root_m32 = root_m12 / diag2  # ^-3/2
    return np.array([-root_m12 + m * m * root_m32, m * root_m32])


def deltas(xy, m, b_perp):
    """Perpendicular distance for each point, to the line y = mx + b"""
    # reminder: b_perp = b cos(theta) = b / sqrt(1 + m^2)
    return xy.dot(perp(m)) - b_perp


def grad_deltas(xy, m, b_perp):
    """Gradient of Delta_i with respect to m, b_perp, indexed on i, p"""
    # vT . Zj
    dD_dm = np.einsum("i,ji->j", dperp(m), xy)
    dD_dbperp = np.ones(len(xy))

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


def linear_ortho_maxlh(data_x, data_y, cov_xy, ax=None, print_on=True):
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

    def to_minimize(v):
        m, b_perp = v
        return -logL(m, b_perp, xy, cov_xy)

    def jac(v):
        m, b_perp = v
        return -grad_logL(m, b_perp, xy, cov_xy)

    reg = stats.linregress(data_x, data_y)
    initial_guess = [reg.slope, reg.intercept]
    initial_guess[1] *= 1 / np.sqrt(1 + initial_guess[0] ** 2)

    # res = optimize.minimize(to_minimize, initial_guess, method="Powell")
    # res = optimize.minimize(to_minimize, initial_guess, method="Newton-CG", jac=jac)
    # res = optimize.minimize(to_minimize, initial_guess, method="SLSQP", jac=jac, options={'ftol':1e-9})
    res = optimize.minimize(to_minimize, initial_guess, method="Nelder-Mead")

    m, b_perp = res.x

    # manual check for convergence
    logL0 = logL(m, b_perp, xy, cov_xy)

    logL_up = logL(m + 0.1 * abs(m), b_perp, xy, cov_xy)
    logL_down = logL(m - 0.1 * abs(m), b_perp, xy, cov_xy)
    logL_right = logL(m, b_perp + 0.1 * abs(b_perp), xy, cov_xy)
    logL_left = logL(m, b_perp - 0.1 * abs(b_perp), xy, cov_xy)

    if print_on:
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
    #     hess = -hess_logL(m, b_perp, xy, cov_xy)
    #     hess_inv = np.linalg.inv(hess)

    # sigma_m, sigma_b_perp = np.sqrt(np.diag(hess_inv))
    # rho_mb_perp = hess_inv[0, 1] / (sigma_m * sigma_b_perp)

    # reminder: b_perp = b cos(theta) = b / sqrt(1 + m^2)
    b = b_perp * np.sqrt(1 + m * m)

    if print_on:
        print(res)
        print("m, b_perp:", m, b_perp)
        # print("err, err, rho:", sigma_m, sigma_b_perp, rho_mb_perp)
        print("m, b:", m, b)
    # plot result if desired
    if ax is not None:
        xlim = ax.get_xlim()
        xs = np.linspace(xlim[0], xlim[1], 100)
        ys = m * xs + b
        ax.plot(xs, ys, color="k")

    return m, b_perp


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
        (ms[m], bs[m]) = linear_ortho_maxlh(boot_x, boot_y, boot_cov, print_on=False)

    print("Bootstrap: m = {} ; b = {}".format(np.average(ms), np.average(bs)))
    print("Bootstrap: sm = {} ; sb = {}".format(np.std(ms), np.std(bs)))
    print("Bootstrap: corr(m, b) = {}".format(np.corrcoef(ms, bs)[0, 1]))
    cov = np.cov(ms, bs)
    covm1 = np.linalg.inv(cov)

    # get the actual solution again, and calc mahalanobis distance from (0, b)
    m, b = linear_ortho_maxlh(data_x, data_y, cov_xy, print_on=False)
    r = np.array([m, 0])
    m_distance_2 = r.dot(covm1).dot(r)
    print("Bootstrap: m-dist from m = 0: ", np.sqrt(m_distance_2))
    return cov


def plot_solution_neighborhood(ax, m, b, xs, ys, covs, cov_mb=None, area=None):
    """
    Color plot of the 2D likelihood function around the given point (m,
    b)

    cov_mb: covariance matrix used to draw an ellipse, so we can see if
    it makes sense

    area: the area over which the function should be plotted, defined as
    [mmin, mmax, bmin, bmax]

    """
    if area is None:
        f = 1.1
        mr = [m / f, m * f]
        br = [b / f, b * f]
        mmin = min(mr)
        mmax = max(mr)
        bmin = min(br)
        bmax = max(br)
    else:
        mmin, mmax, bmin, bmax = area

    xy = np.column_stack([xs, ys])

    grid_m = np.linspace(mmin, mmax, 100)
    grid_b = np.linspace(bmin, bmax, 100)
    image = np.zeros((len(grid_m), len(grid_b)))
    for ((i, mi), (j, bj)) in itertools.product(enumerate(grid_m), enumerate(grid_b)):
        image[i, j] = logL(mi, bj, xy, covs)

    im = ax.imshow(
        image, extent=[bmin, bmax, mmin, mmax], origin="lower", aspect="auto"
    )
    ax.set_ylabel("m")
    ax.set_xlabel("b")
    ax.plot(b, m, "kx")
