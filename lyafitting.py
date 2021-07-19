import get_spectrum
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import minimize, bracket, brute
import numpy as np
import argparse
import warnings
import astropy

warnings.filterwarnings("ignore", category=astropy.units.UnitsWarning)

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
    # at these wavelengths, 1e22 * cross is about 100
    # exp(100) ~ e43 is still relatively safe
    lya_exclude_wav_ranges = [[1212.67, 1218.67]]
    safe_range = np.logical_not(wavs_in_ranges(wavs, lya_exclude_wav_ranges))

    # for lya, avoid wavelengths where the cross section is large (maybe
    # better to choose this range explicitly in wavelength numbers)
    # cross_eval = cross(wavs)
    # safe_cross = cross_eval < 0.05 * np.amax(cross_eval)

    # avoid datapoints where the extinction is too strong (flux too low)
    safe_flux = flux > np.average(flux) / 10
    return np.logical_and.reduce((safe_flux, safe_range))


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
    return np.exp(-NHI * cross(l))


def chi2(NHI, fc, sigma_c, wavs, flux):
    # overflows easily. Ignore fit points where overflow occurs
    extinctions = np.exp(NHI * cross(wavs))
    deltas = fc(wavs) - flux * extinctions
    sigmas = sigma_c * extinctions
    square_devs = np.square((deltas / sigmas)).sum() / (len(deltas) - 1)
    chi2 = square_devs[np.isfinite(square_devs)].sum()
    # print("chi2 = ", chi2)
    return chi2


def plot_profile(ax, fc, NHI):
    """Plot an extra profile of user-specified NHI.

    Ax needs to have been prepared properly (xlims already in their
    final form), and fc is the continuum fit.

    """
    extra_color = "g"
    x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 500)
    y = fc(x) * extinction_factor(NHI, x)
    ax.plot(x, y, color=extra_color, label="user")


def plot_fit(ax, wavs, flux, fc, NHI):
    cont_color = "m"
    lya_color = "xkcd:sky blue"

    # continuum fit
    fcs = fc(wavs)
    ax.plot(wavs, fcs, label="continuum fit", color=cont_color, zorder=30)

    # lya fit
    fms = fc(wavs) * extinction_factor(NHI, wavs)
    ax.plot(wavs, fms, label="profile fit", color=lya_color, zorder=40)

    # data / used for cont / used for lya
    used_for_cont = safe_for_cont(wavs)
    used_for_lya = safe_for_lya(wavs, flux)
    ax.plot(wavs, flux, label="data", color="k", zorder=10)
    # change this to an axvspan plot
    ax.plot(
        wavs[used_for_cont],
        flux[used_for_cont],
        label="used for continuum",
        color=cont_color,
        linestyle="none",
        marker="x",
        alpha=0.5,
        zorder=50,
    )
    # use red x for rejected points
    ax.plot(
        wavs[np.logical_not(used_for_lya)],
        flux[np.logical_not(used_for_lya)],
        label="rejected",
        color='r',
        linestyle="none",
        marker="x",
        alpha=0.5,
        zorder=45,
    )

    # fcrec = flux / factor
    # ax.plot(wavs, fcrec, label="reconstructed")

    # ax.text(
    #     LYA, ax.get_ylim()[1] * 0.8, , ha="center"
    # )
    prepare_axes(ax)
    # ax.set_ylim(None, np.amax(fcs) * 1.05)


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

    return NHI, fc


def run_all():
    for target in get_spectrum.target_use_which_spectrum:
        fig, ax = plt.subplots()
        NHI, fc = lya_fit(target, ax=ax)
        ax.set_title(target + "\nlogNHI = {:2f}".format(np.log10(NHI)))
        fig.savefig(f"./lya-plots/{target}.pdf")
        print(target, NHI)


def run_one(target, compare=None):
    ax = plt.gca()
    NHI, fc = lya_fit(target, ax=ax)
    plt.title(target, loc="right")
    if compare is not None:
        NHIc = np.power(10.0, compare)
        plot_profile(ax, fc, NHIc)

    plt.show()


def main():
    # STIS example
    # default_target = "HD094493"
    # IUE H example
    # default_target = "HD037525"
    # IUE L example
    default_target = "HD097471"
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, default=default_target)
    ap.add_argument(
        "--compare",
        type=float,
        default=None,
        help="Plot extra profile using this logNHI value",
    )
    args = ap.parse_args()

    if args.target == "all":
        run_all()
    else:
        run_one(args.target, args.compare)


main()
