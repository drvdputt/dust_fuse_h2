"""Pearson correlation coefficient for data that have cov(x,y) for each point.

A Monte Carlo method is used to generate a set of possible pearson
coefficients. Maybe it's possible to do this analytically too, but I'll
start with Monte Carlo just to be sure.

This coefficient should be a bit more meaningful than just a slope and
an uncertainty, because is tells you something about the DATA directly,
instead of some best fitting model which has underlying assumptions.

"""
import numpy as np


def pearson_mc(xs, ys, covs):
    """Calculate Pearson correlation coefficient and uncertainty on it in a MC way.

    Repeatedly resample xs, ys using 2D gaussian described by covs."""

    # store the samples as follows
    # col0 = all resamplings of x0
    # -> each row is a different realization of our 75 sightlines
    N = len(xs)  # number of sightlines
    M = 100  # number of resamples
    x_samples = np.zeros((M, N))
    y_samples = np.zeros((M, N))
    for i in range(N):
        samples = np.random.multivariate_normal(
            mean=(xs[i], ys[i]), cov=covs[i], size=M
        )
        x_samples[:, i] = samples[:, 0]
        y_samples[:, i] = samples[:, 1]

    corrcoefs = np.array([np.corrcoef(x_samples[j], y_samples[j]) for j in range(M)])
    rhos = corrcoefs[:, 0, 1]
    avg = np.average(rhos)
    std = np.std(rhos)
    rho_naive = np.corrcoef(xs, ys)[0, 1]
    sigmas = rho_naive / std
    print(f"+++ MC pearson result +++\n rho = {rho_naive:.2f} +- {std:.2f} ({sigmas:.2f} sigma)")
