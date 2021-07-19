import get_spectrum
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import minimize, bracket, brute
import numpy as np
import argparse

LYA = 1215.67


def prepare_axes(ax):
    ax.set_xlabel("wavelength")
    ax.set_ylabel("flux (erg cm$^{-2}$s$^{-1}\\AA^{-1}$)")


def wavs_in_ranges(wavs, ranges):
    """Determine if wavelength is in one of the ranges given.

    wavs: wavelengths to check

    ranges: list of (lower limit, upper limit) pairs

    Returns
    -------
    in_range: np.array containing True if the wavelength at the same
    index was in one of the ranges

    """
    in_range = np.full(len(wavs), False)
    for (lo, hi) in ranges:
        in_range = np.logical_or(in_range, np.logical_and(wavs > lo, wavs < hi))
    return in_range


def safe_for_cont(wavs):
    # use only these points for continuum estimation
    cont_wav_ranges = [[1262, 1275], [1165, 1170], [1179, 1181], [1185, 1188]]
    return wavs_in_ranges(wavs, cont_wav_ranges)


def safe_for_lya(wavs, flux):
    lya_exclude_wav_ranges = [[1171, 1177]]
    safe_range = np.logical_not(wavs_in_ranges(wavs, lya_exclude_wav_ranges))

    # for lya, avoid wavelengths where the cross section is large (maybe
    # better to choose this range explicitly in wavelength numbers)
    cross_eval = cross(wavs)
    safe_cross = cross_eval < 0.1 * np.amax(cross_eval)

    # avoid datapoints where the extinction is too strong (flux too low)
    safe_flux = flux > np.amax(flux) / 6
    return np.logical_and.reduce((safe_cross, safe_flux, safe_range))


def estimate_continuum(wavs, flux):
    """Estimate the continuum using a linear fit"""
    use = safe_for_cont(wavs)

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


def extinction_factor(NHI, l):
    return np.exp(-NHI * cross(wavs))


def chi2(NHI, fc, sigma_c, wavs, flux):
    extinctions = np.exp(NHI * cross(wavs))
    deltas = fc(wavs) - flux * extinctions
    sigmas = sigma_c * extinctions
    chi2 = np.square((deltas / sigmas)).sum() / (len(deltas) - 1)
    # print("chi2 = ", chi2)
    return chi2


def plot_fit(ax, wavs, flux, fc, NHI):
    cont_color = "m"
    lya_color = "xkcd:sky blue"

    # continuum fit
    fcs = fc(wavs)
    ax.plot(wavs, fcs, label="continuum fit", color=cont_color)

    # lya fit
    fms = fc(wavs) * extinction_factor(NHI, wavs)
    ax.plot(wavs, fms, label="profile fit", color=lya_color)

    # data / used for cont / used for lya
    used_for_cont = safe_for_cont(wavs)
    used_for_lya = safe_for_lya(wavs, flux)
    ax.plot(wavs, flux, label="data", color="k")
    ax.plot(
        wavs[used_for_cont],
        flux[used_for_cont],
        label="used for continuum",
        color=cont_color,
        linestyle="none",
        marker="x",
        alpha=0.5,
    )
    ax.plot(
        wavs[used_for_lya],
        flux[used_for_lya],
        label="used for lya",
        color=lya_color,
        linestyle="none",
        marker=".",
        alpha=0.5,
    )

    # fcrec = flux / factor
    # ax.plot(wavs, fcrec, label="reconstructed")

    ax.text(
        LYA, ax.get_ylim()[1] * 0.8, "logNHI = {:2f}".format(np.log10(NHI)), ha="center"
    )
    prepare_axes(ax)


def lya_fit(target, ax=None):
    """Fit NHI using lya.

    target: string referring to any of the targets (see directories in
    ./data/)

    ax: axes to plot the result on

    """

    wavs, flux = get_spectrum.processed(target)

    # continuum
    slope, intercept, sigma_c = estimate_continuum(wavs, flux)

    def fc(x):
        return slope * x + intercept

    use = safe_for_lya(wavs, flux)

    # the fitting itself

    NHI_init = 1e20
    fargs = (fc, sigma_c, wavs[use], flux[use])
    # result = minimize(chi2, NHI_init, args=fargs)
    # NHI = result.x[0]
    # result = bracket(chi2, 1e19, 1e22, args=fargs)
    # print(result)
    result = brute(chi2, [(1e19, 1e22)], args=fargs, Ns=1000)
    print(result)
    NHI = result[0]

    if ax is not None:
        plot_fit(ax, wavs, flux, fc, NHI)

    return NHI


def main():
    #    default_target = "HD094493"
    default_target = "HD037525"
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, default=default_target)
    args = ap.parse_args()

    if args.target == "all":
        for target in get_spectrum.target_use_which_spectrum:
            NHI = lya_fit(target)
            print(target, NHI)
    else:
        lya_fit(args.target, ax=plt.gca())
        plt.title(args.target, loc="right")
        plt.show()


main()
