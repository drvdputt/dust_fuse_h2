import numpy as np
import math
from scipy import optimize


def perp(m):
    """Vector perpendicular to a line with slope m"""
    return np.array([-m, 1]) / math.sqrt(1 + m * m)


def deltas(xy, m, b):
    """Perpendicular distance for each point, to the line y = mx + b"""
    v = perp(m)
    return xy.dot(v) - b * v[1]


def sigmas(covs, m):
    """Projection of the covariance matrix, perpendicular to y = mx + b"""
    v = perp(m)
    return np.einsum("i,kij,j->k", v, covs, v)


def logL(m, b, xy, covs):
    """Log likelihood function from Hogg et al. (2010), using the
    perpendicular distance and covariance."""
    return -0.5 * np.square(deltas(xy, m, b) / sigmas(covs, m)).sum()


def linear_ortho_maxlh(data_x, data_y, cov_xy):
    """
    Do a linear fit based on orthogonal distance, to data where each
    point can have a different covariance between x and y. Uses the
    likelihood function from Hogg et al. (2010). The likelihood is
    maximized using the default scipy.optmize minizer, and the errors
    are estimated using the inverse Hessian of the likelihood at its
    maximum, as provided by the optimized result object.

    Returns
    -------
    m: slope
    b: intercept
    sigma_m: estimate of error on slop
    sigma_b: estimate of error on intercept
    rho_mb: estimate of pearson coefficient between m and b
    """

    xy = np.column_stack((data_x, data_y))

    def to_minimize(v):
        m = v[0]
        b = v[1]
        return -logL(m, b, xy)

    res = optimize.minimize(to_minimize, (1, 1))
    m, b = res.x
    sigma_m, sigma_b = np.sqrt(np.diag(res.hess_inv))
    rho_mb = res.hess_inv[0, 1] / (sigma_m * sigma_b)
    print(res)
    print("mx + b,", m, b)
    print("err, err, rho:", sigma_m, sigma_b, rho_mb)
    return m, b, sigma_m, sigma_b, rho_mb
