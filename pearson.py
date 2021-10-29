"""Pearson correlation coefficient for data that have cov(x,y) for each point.

A Monte Carlo method is used to generate a set of possible pearson
coefficients. Maybe it's possible to do this analytically too, but I'll
start with Monte Carlo just to be sure.

This coefficient should be a bit more meaningful than just a slope and
an uncertainty, because is tells you something about the DATA directly,
instead of some best fitting model which has underlying assumptions.

"""
import numpy as np
from matplotlib import pyplot as plt


def draw_points(xs, ys, covs, M):
    # store the samples as follows
    # col0 = all resamplings of x0
    # -> each row is a different realization of our 75 sightlines
    N = len(xs)
    x_samples = np.zeros((M, N))
    y_samples = np.zeros((M, N))
    for j in range(N):
        samples = np.random.multivariate_normal(
            mean=(xs[j], ys[j]), cov=covs[j], size=M
        )
        x_samples[:, j] = samples[:, 0]
        y_samples[:, j] = samples[:, 1]
    return x_samples, y_samples


def pearson_mc(xs, ys, covs, save_hist=None):
    """Calculate Pearson correlation coefficient and uncertainty on it in a MC way.

    Repeatedly resample xs, ys using 2D gaussian described by covs.

    Returns
    -------
    rho : correlation coefficient

    std : standard deviation of the rho samples
    """

    M = 6000  # number of resamples
    # scramble test to create null hypothesis distribution of rho.
    # Technically, only the y samples need to be scrambled, but I'm
    # going to do both just to be sure.
    x_samples, y_samples = draw_points(xs, ys, covs, M)
    x_samples_scrambled, y_samples_scrambled = draw_points(xs, ys, covs, M)
    for i in range(M):
        # np.random.shuffle(x_samples_scrambled[i])
        np.random.shuffle(y_samples_scrambled[i])

    corrcoefs_null = np.array(
        [np.corrcoef(x_samples_scrambled[i], y_samples_scrambled[i]) for i in range(M)]
    )
    rhos_null = corrcoefs_null[:, 0, 1]
    # p16_null = np.percentile(rhos_null, 16)
    # p84_null = np.percentile(rhos_null, 84)
    # std_null = (p84_null - p16_null) / 2
    std_null = np.std(rhos_null)

    corrcoefs = np.array([np.corrcoef(x_samples[i], y_samples[i]) for i in range(M)])
    rhos = corrcoefs[:, 0, 1]

    p16 = np.percentile(rhos, 16)
    p84 = np.percentile(rhos, 84)

    std = np.std(rhos)
    # std = (p84 - p16) / 2

    def rho_sigma_message(rho):
        num_sigmas = rho / std_null
        num_sigmas_lo = p16 / std_null
        num_sigmas_hi = p84 / std_null
        return f"rho = {rho:.2f} +- {std:.2f} ({num_sigmas:.2f} sigma0)\n sigmas range = {num_sigmas_lo:.2f} - {num_sigmas_hi:.2f}"

    print("+++ MC pearson result +++")
    rho_naive = np.corrcoef(xs, ys)[0, 1]
    print("raw: ", rho_sigma_message(rho_naive))
    avg = np.average(rhos)
    print("avg: ", rho_sigma_message(avg))
    med = np.median(rhos)
    print("median: ", rho_sigma_message(med))

    # for debugging purposes
    if save_hist is not None:
        fig, ax = plt.subplots()
        ax.hist(rhos_null, bins=50)
        ax.hist(rhos, bins=50)
        fig.savefig(save_hist)

    return med, std_null
