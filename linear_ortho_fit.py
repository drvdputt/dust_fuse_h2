import numpy as np
from scipy import optimize
import itertools
from astropy.modeling import models, fitting
from covariance import cov_ellipse
from rescale import rescale_data, unscale_mb

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
    S2 = np.einsum("i,kij,j->k", v, covs, v)
    if (S2 == 0).any():
        print("Sigma^2 zero for some reason!")
    return S2


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
    D = deltas(xy, m, b)
    S2 = sigma2s(covs, m)
    square_devs = np.square(D) / S2
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
    eps = 1e-5
    em = np.array([1, 0])
    eb = np.array([0, 1])
    h = [m * eps, b * eps]
    d = [em * h[0], eb * h[1]]

    # 2d function
    def f(r):
        return logL(r[0], r[1], xy, covs)

    r0 = np.array([m, b])
    fr0 = f(r0)

    def d2_d1d2(i, j):
        return (f(r0 + d[i] + d[j]) - f(r0 + d[i]) - f(r0 + d[j]) + fr0) / h[i] * h[j]

    return np.array([[d2_d1d2(0, 0), d2_d1d2(0, 1)], [d2_d1d2(0, 1), d2_d1d2(1, 1)]])


def find_outliers(xy, covs, m, b_perp):
    """Given a best fit line (m, b_perp), return a set {} of indices
    indicating the outliers in the given data.

    """
    D = deltas(xy, m, b_perp)
    S = np.sqrt(sigma2s(covs, m))
    devs = D / S
    # print("devs", devs)
    return {i for i in np.where(np.abs(devs) > 3)[0]}
    # devwidth = np.std(devs)
    # devrms = np.sqrt(np.average(np.square(devs)))
    # print("deviations of the order", devrms, devwidth)
    # return {i for i in np.where(np.abs(devs) > 2 * devwidth)[0]}


def linear_ortho_maxlh(
    data_x,
    data_y,
    cov_xy,
    ax=None,
    basic_print=True,
    debug_print=False,
    sigma_hess=False,
    manual_outliers=None,
    auto_outliers=False,
    fit_includes_outliers=False,
):
    """Do a linear fit based on orthogonal distance, to data where each
    point can have a different covariance between x and y. Uses the
    likelihood function from Hogg et al. (2010). The likelihood is
    maximized using the default scipy.optmize minizer, and the errors
    are estimated using the inverse Hessian of the likelihood at its
    maximum, as provided by the optimized result object.

    IMPORTANT NOTE: the chi2 analog used is not the typical Delta.T *
    C-1 * Delta, with Delta being the distance between the model point
    and the observed point. In fact, the measurement could come from any
    point on the line. Instead, a 1D model is used which treats the
    orthogonal displacement from the line as the random variable.

    The equation used here is Delta.perp^2 / (v^T * C * v), where
    Delta.perp and v are the orthogonal distance and direction (unit
    vector), with respect to the mx + b line. (v^T * C * v) is the
    correct expression for Delta.perp. In the eigen basis of the
    covariance ellipse, Delta.perp = Delta.x * cos(theta) + Delta.y *
    sin(theta), and Delta.x and Delta.y are independent. Therefore,
    V(Delta.perp) = cos^2(theta) V(Delta.x) + sin^2(theta) V(Delta.y),
    with theta the angle between the normal and the y-eigenvector.

    Parameters
    ----------
    manual_outliers : list of int
        manually marked outliers for the first outlier detection
        iteration. Default is None, which just follows auto_outliers.

    auto_outliers : bool
        mark outliers and rerun the fitting step iteratively until no
        new outliers are found

    fit_includes_outliers : bool
        Use the detected outliers in the fitting, despite them being
        outliers.

    Returns
    -------
    dict :
        Contains
        m: slope
        b_perp: perpendicular intercept (b / sqrt(1 + m^2))
        m_unc: estimate of error on slop
        b_perp_unc: estimate of error on b_perp
        outlier_idxs: list of indices of outliers
    """
    xy = np.column_stack((data_x, data_y))
    # choose a scale factor to avoid the problems that come with the
    # huge dynamic range difference between certain x and y data.
    factor_x = 1 / np.std(data_x)
    factor_y = 1 / np.std(data_y)
    xy, covs = rescale_data(xy, cov_xy, factor_x, factor_y)

    # if manual outliers, add them here. Will be deleted after first
    # iteration
    if manual_outliers is not None:
        outliers = {i for i in manual_outliers}
    else:
        outliers = set()

    def no_outliers(data):
        if fit_includes_outliers:
            return data
        else:
            return np.delete(data, list(outliers), axis=0)

    def to_minimize(v):
        m, b_perp = v
        return -logL(m, b_perp, no_outliers(xy), no_outliers(covs))

    def jac(v):
        m, b_perp = v
        return -grad_logL(m, b_perp, no_outliers(xy), no_outliers(covs))

    # use naive result as initial guess
    line_init = models.Linear1D()
    fit = fitting.LinearLSQFitter()
    naive_fit = fit(line_init, xy[:, 0], xy[:, 1], weights=1.0 / np.sqrt(covs[:, 1, 1]))
    initial_guess = np.array(
        [
            naive_fit.slope.value,
            naive_fit.intercept.value,
        ]
    )
    initial_guess[1] = b_to_b_perp(*initial_guess)

    if debug_print:
        eps = np.abs(1.0e-6 * initial_guess)
        grad_approx = optimize.approx_fprime(initial_guess, to_minimize, eps)
        grad_exact = jac(initial_guess)
        print("initial guess: ", initial_guess)
        print("grad approx", grad_approx, "grad exact", grad_exact)

    iter_done = False
    counter = 0
    while not iter_done:
        # find maximum, and loop
        gtol = np.linalg.norm(jac(initial_guess)) * 1e-6
        res = optimize.minimize(
            to_minimize,
            initial_guess,
            method="BFGS",
            jac=jac,
            options={"disp": False, "gtol": gtol},
        )
        m, b_perp = res.x
        chi2min = to_minimize(res.x)

        if auto_outliers:
            print("outlier iteration ", counter)
            # clear any manual outliers
            if counter == 0:
                outliers.clear()
            # new outliers is previous + current
            new_outliers = outliers or find_outliers(xy, covs, m, b_perp)
            # if new set = old set, we're done
            if outliers == new_outliers:
                iter_done = True
            # update outlier set declared at top of this function
            outliers = new_outliers
        else:
            # if we're not doing auto_outliers, 1 iteration is enough
            iter_done = True

        counter += 1

    outlier_output = sorted(list(outliers))

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

    # undo the scale factors to get the m and b for the real data
    b = b_perp_to_b(m, b_perp)
    m_real, b_real = unscale_mb(m, b, factor_x, factor_y)
    b_perp_real = b_to_b_perp(m_real, b_real)

    if debug_print:
        print("Scaled solution")
        print("m, b_perp:", m, b_perp)
        print("m, b:", m, b)
    if basic_print:
        print("Solution")
        print("m, b_perp:", m_real, b_perp_real)
        print("m, b:", m_real, b_real)
        print(f"chi2min: {chi2min} or {chi2min/(len(xy) - 2)} per DOF")

    result = {"m": m_real, "b_perp": b_perp_real, "outlier_idxs": outlier_output}

    if sigma_hess:
        # estimate error using hessian. Useful for choosing area around max
        # likelihood.
        hess_inv = res.hess_inv
        sigma_m, sigma_b_perp = np.sqrt(np.diag(hess_inv))
        rho_mb_perp = hess_inv[0, 1] / (sigma_m * sigma_b_perp)
        if debug_print:
            print("Scaled solution:")
            print(f"Hess: sm = {sigma_m} ; sb = {sigma_b_perp}")
            print(f"Hess: corr(m, b) = {rho_mb_perp}")

        # use relative error to unscale
        sigma_m_frac = sigma_m / m
        sigma_b_frac = sigma_b_perp / b_perp

        sigma_m_real = sigma_m_frac * m_real
        sigma_b_perp_real = sigma_b_frac * b_perp_real

        result["m_unc"] = sigma_m_real
        result["b_perp_unc"] = sigma_b_perp_real

    return result


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
        r = linear_ortho_maxlh(boot_x, boot_y, boot_cov, basic_print=False)
        ms[m] = r["m"]
        bs[m] = r["b_perp"]

    print("Bootstrap: m = {} ; b = {}".format(np.average(ms), np.average(bs)))
    print("Bootstrap: sm = {} ; sb = {}".format(np.std(ms), np.std(bs)))
    print("Bootstrap: corr(m, b) = {}".format(np.corrcoef(ms, bs)[0, 1]))
    cov = np.cov(ms, bs)
    return cov


def plot_solution_linescatter(ax, m_samples, b_samples, **plot_kwargs):
    """Visualize the spread on the linear fits using transparent plots.

    Requires samples of m and b_perp properly sampled from the
    likelihood.

    """
    x = np.array(ax.get_xlim())
    for (m, b_perp) in zip(m_samples, b_samples):
        y = m * x + b_perp_to_b(m, b_perp)
        ax.plot(x, y, **plot_kwargs)


def sample_likelihood(m, b_perp, m_grid, b_perp_grid, logL_grid):
    """
    Analyze the likelihood around the maximum by sampling from it.

    Parameters
    ----------
    m, b_perp: the fit result

    xs, ys, covs: the data

    m_grid, b_perp_grid, logL_grid : logL, discretized on a grid

    Returns
    -------
    random_m, random_b: array, array
        random samples of m and b_perp, drawn from exp(logL_grid)

    """
    L_grid = np.exp(logL_grid)
    L_grid /= np.sum(L_grid)

    # sample m and b, using L as weight
    L_flat = L_grid.flatten()
    random_flat_i = np.random.default_rng().choice(len(L_flat), size=333, p=L_flat)
    # translate from flat indices, to 0th and 1st indices
    random_m_i, random_b_i = np.unravel_index(random_flat_i, L_grid.shape)
    random_m = m_grid[random_m_i]
    random_b = b_perp_grid[random_b_i]
    return random_m, random_b


def calc_logL_grid(m_min, m_max, b_perp_min, b_perp_max, xs, ys, covs, res=400):
    """
    Calculate logL on a grid.

    Extent of the grid needs to be given. Is best guessed using some
    estimate of the sigmas.

    m is on the y-axis (index 0), b is on the x-axis (index 1)

    """
    xy = np.column_stack([xs, ys])
    grid_m = np.linspace(m_min, m_max, res)
    grid_b = np.linspace(b_perp_min, b_perp_max, res)
    grid_logL = np.zeros((len(grid_m), len(grid_b)))
    for ((i, mi), (j, bj)) in itertools.product(enumerate(grid_m), enumerate(grid_b)):
        grid_logL[i, j] = logL(mi, bj, xy, covs)
    return grid_m, grid_b, grid_logL


def plot_solution_neighborhood(
    ax, image, extent, m, b_perp, cov_mb=None, extra_points=None, what="logL"
):
    """
    Color plot of the 2D likelihood function, with optional markings.

    b_perp is on the x-axis, m on the y axis

    image: 2D array

    extent: [b_perp_min, b_perp_max, m_min, m_max]

    cov_mb: covariance matrix used to draw an ellipse, so we can see if
    it makes sense

    m, bperp: the fit solution

    extra_points: points [(b, m), ...] to be plotted on the image

    what: choose 'logL', 'L'

    """
    imshow_kwargs = dict(extent=extent, origin="lower", aspect="auto", cmap="viridis")

    if what == "logL":
        im = ax.imshow(image, **imshow_kwargs)
    elif what == "L":
        im = ax.imshow(np.exp(image), **imshow_kwargs)
    else:
        print("Wrong string, no plot made")

    ax.figure.colorbar(im, ax=ax)
    ax.set_ylabel("m")
    ax.set_xlabel("b$\\perp$")
    ax.plot(b_perp, m, "kx")

    if extra_points is not None:
        for (bi, mi) in extra_points:
            ax.plot(bi, mi, "k+")

    if cov_mb is not None:
        ax.add_patch(
            cov_ellipse(b_perp, m, cov_mb[::-1, ::-1], facecolor="none", edgecolor="k")
        )

def b_perp_to_b(m, b_perp):
    # reminder: b_perp = b cos(theta) = b / sqrt(1 + m^2)
    return b_perp * np.sqrt(1 + m * m)


def b_to_b_perp(m, b):
    return b / np.sqrt(1 + m * m)
