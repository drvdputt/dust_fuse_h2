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
import rescale
from pathlib import Path

RNG = np.random.default_rng(4321)


def draw_points(xs, ys, covs, M):
    """
    Resample a set of points M times, adding noise according to their covariance matrices.

    Returns
    -------
    x_samples, y_samples : np.array
        Every column j, is x[j] redrawn M times.
        Has M rows, and every row is a realization of xs or ys.
    """
    # store the samples as follows
    # col0 = all resamplings of x0
    # -> each row is a different realization of our 75 sightlines

    # rescale data to avoid numerical problems
    factor_x = 1 / np.std(xs)
    factor_y = 1 / np.std(ys)
    xyr, covr = rescale.rescale_data(
        np.column_stack((xs, ys)), covs, factor_x, factor_y
    )
    N = len(xyr)
    x_samples = np.zeros((M, N))
    y_samples = np.zeros((M, N))
    for j in range(N):
        samples = RNG.multivariate_normal(mean=xyr[j], cov=covr[j], size=M)
        x_samples[:, j] = samples[:, 0]
        y_samples[:, j] = samples[:, 1]

    # unscale the data again before returning
    return x_samples / factor_x, y_samples / factor_y


def pearson_mc(xs, ys, covs, save_hist=None, hist_ax=None):
    """Calculate Pearson correlation coefficient and uncertainty on it in a MC way.

    Repeatedly resample xs, ys using 2D gaussian described by covs.

    Parameters
    ----------

    save_hist : string
        File name for figure of histogram for rho and rho0.

    hist_ax : figure, axes
        Plot histogram on this axes. If None, a new figure will be made.

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
        RNG.shuffle(y_samples_scrambled[i])

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

    outputs = [med, std_null]

    # any of these this implies that we have to plot
    if save_hist is not None or hist_ax is not None:
        # make new fig, ax if none was given
        if hist_ax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = hist_ax.get_figure(), hist_ax

        bins = 64
        ax.hist(rhos_null, bins=bins, label="null data", color="xkcd:gray", alpha=0.5)
        ax.hist(
            rhos, bins=bins, label="resampled data", color="xkcd:bright blue", alpha=0.5
        )

        # save hist to file if requested
        if save_hist is not None:
            d = Path(save_hist).parent
            d.mkdir(exist_ok=True)
            fig.savefig("rho_histograms/" + save_hist)

    return outputs
