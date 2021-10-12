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

    Repeatedly resample xs, ys using 2D gaussian described by covs.

    Returns
    -------
    rho : correlation coefficient

    std : standard deviation of the rho samples
    """

    # store the samples as follows
    # col0 = all resamplings of x0
    # -> each row is a different realization of our 75 sightlines
    N = len(xs)  # number of sightlines
    M = 2000  # number of resamples
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
    std = np.std(rhos)

    def rho_sigma_message(rho):
        sigmas = rho / std
        return f"rho = {rho:.2f} +- {std:.2f} ({sigmas:.2f} sigma)"

    print("+++ MC pearson result +++")
    rho_naive = np.corrcoef(xs, ys)[0, 1]
    print("data: ", rho_sigma_message(rho_naive))
    avg = np.average(rhos)
    print("avg: ", rho_sigma_message(avg))
    # med = np.median(rhos)
    # print("median: ", rho_sigma_message(med))
    return rho_naive, std
