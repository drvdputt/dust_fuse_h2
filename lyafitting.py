import get_spectrum
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import minimize, bracket, brute
import numpy as np


def prepare_axes(ax):
    ax.set_xlabel("wavelength")
    ax.set_ylabel("flux (erg cm$^{-2}$s$^{-1}\\AA^{-1}$)")


def wavs_in_ranges(wavs, ranges):
    """Determine if wavelength is in one of the ranges given.

    ranges: list of (lower limit, upper limit) pairs

    Returns
    -------
    wavs: wavelengths to check
    in_range: np.array containing True if the wavelength at the same
    index was in one of the ranges

    """
    in_range = np.full(len(wavs), False)
    for (lo, hi) in ranges:
        in_range = np.logical_or(in_range, np.logical_and(wavs > lo, wavs < hi))
    return in_range


def estimate_continuum(wavs, flux):
    """Estimate the continuum using a linear fit"""

    # use only these points for continuum estimation
    wav_ranges = [[1262, 1275], [1165, 1170], [1179, 1181], [1185, 1188]]
    use = wavs_in_ranges(wavs, wav_ranges)

    # simple linear regression
    x = wavs[use]
    y = flux[use]
    linregress_result = linregress(x, y)

    # estimation of spectroscopic noise
    m = linregress_result.slope
    b = linregress_result.intercept
    sigma = np.sqrt(np.average(np.square(y - m * x + b)))

    return linregress_result.slope, linregress_result.intercept, sigma


def cross(l):
    l0 = 1215.67
    return 4.26e-20 / (6.04e-10 + np.square(l - l0))


def chi2(NHI, fc, sigma_c, wavs, flux):
    extinctions = np.exp(NHI * cross(wavs))
    deltas = fc(wavs) - flux * extinctions
    sigmas = sigma_c * extinctions
    chi2 = np.square((deltas / sigmas)).sum() / (len(deltas) - 1)
    # print("chi2 = ", chi2)
    return chi2


def plot_fit(ax, wavs, flux, fc, NHI):
    fcs = fc(wavs)
    fms = fc(wavs) * np.exp(-NHI * cross(wavs))
    ax.plot(wavs, flux, label="data", color="k")
    ax.plot(wavs, fcs, label="continuum fit")
    ax.plot(wavs, fms, label="profile fit")
    prepare_axes(ax)


# -----------------------------------------------------------------------


def lya_fit(target, ax=None):
    """Fit NHI using lya.

    target: string referring to any of the targets (see directories in
    ./data/)

    ax: axes to plot the result on

    """

    wavs, flux = get_spectrum.processed(target)

    # avoid nans (move to get_spectrum later. First I need to know if
    # this will work for all data.)
    safe = np.isfinite(flux)
    safewavs = wavs[safe]
    safeflux = flux[safe]

    # continuum
    slope, intercept, sigma_c = estimate_continuum(safewavs, safeflux)

    def fc(x):
        return slope * x + intercept

    # for lya, avoid wavelengths where the cross section is large (maybe
    # better to choose this range explicitly in wavelength numbers)
    cross_eval = cross(safewavs)
    safe_cross = cross_eval < 0.1 * np.amax(cross_eval)
    safewavs = safewavs[safe_cross]
    safeflux = safeflux[safe_cross]
    if safe.sum() == 0:
        print("no safe data points!")
        raise

    # avoid datapoints where the extinction is too strong (flux too low)
    use = safeflux > np.amax(safeflux) / 6

    # the fitting itself
    NHI_init = 1e20
    fargs = (fc, sigma_c, safewavs[use], safeflux[use])
    # result = minimize(chi2, NHI_init, args=fargs)
    # NHI = result.x[0]
    # result = bracket(chi2, 1e19, 1e22, args=fargs)
    # print(result)
    result = brute(chi2, [(1e19, 1e22)], args=fargs, Ns=1000)
    print(result)
    NHI = result[0]

    if ax is not None:
        plot_fit(ax, wavs, flux, fc, NHI)


def main():
    # test for one specific target for now
    target = "HD094493"
    lya_fit(target, ax=plt.gca())
    plt.show()


main()
