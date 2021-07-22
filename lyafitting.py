import get_spectrum
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import brute
import numpy as np
import argparse
import warnings
import astropy
from astropy import convolution, stats

warnings.filterwarnings("ignore", category=astropy.units.UnitsWarning)

LYA = 1215.67


def prepare_axes(ax):
    ax.set_xlabel("wavelength ($\\AA$)")
    ax.set_ylabel("flux (erg cm$^{-2}$s$^{-1}\\AA^{-1}$)")


def boxcar_smooth(wavs, flux):
    """
    Convolve with a boxcar.
    """
    pix_per_ang = len(wavs) / (wavs[-1] - wavs[0])
    pix_per_o25 = 0.50 * pix_per_ang
    if pix_per_o25 < 2:
        # do nothing if spectrum too low res for smoothing
        return flux

    kernel = convolution.Box1DKernel(pix_per_o25)
    smoothed = convolution.convolve(flux, kernel)
    return smoothed


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


def not_peak(wavs, flux):
    """Return mask to avoid peaks in the spectrum."""
    masked_array = stats.sigma_clip(flux)  # uses 5 iterations by default
    return np.logical_not(masked_array.mask)


def safe_for_cont(wavs, flux):
    """Return mask that indicates wavelengths for continuum fit."""
    # use only these points for continuum estimation
    cont_wav_ranges = [[1262, 1275], [1165, 1170], [1179, 1181], [1185, 1188]]
    safe = np.logical_and(wavs_in_ranges(wavs, cont_wav_ranges), not_peak(wavs, flux))
    return safe


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

    return np.logical_and.reduce((safe_flux, safe_range, not_peak(wavs, flux)))


def estimate_continuum(wavs, flux):
    """Estimate the continuum using a linear fit"""
    use = safe_for_cont(wavs, flux)

    # simple linear regression
    x = wavs[use]
    y = flux[use]
    linregress_result = linregress(x, y)

    # estimation of spectroscopic noise
    m = linregress_result.slope
    b = linregress_result.intercept

    def fc(x):
        return m * x + b

    return fc


def estimate_noise(wavs, flux, fc):
    # see figure 1 in DS94
    wav_range_DS94 = [1263, 1269]
    use = wavs_in_ranges(wavs, [wav_range_DS94])
    # DS94 use sum instead of average. Not sure if that is the correct
    # way.
    sigma = np.sqrt(np.average(np.square(flux[use] - fc(wavs[use]))))
    return sigma


def cross(l):
    l0 = 1215.67
    return 4.26e-20 / (6.04e-10 + np.square(l - l0))


def extinction_factor(logNHI, l):
    return np.exp(-np.power(10.0, logNHI) * cross(l))


def chi2(logNHI, fc, sigma_c, wavs, flux):
    # overflows easily. Ignore fit points where overflow occurs
    extinction = 1 / extinction_factor(logNHI, wavs)
    deltas = fc(wavs) - flux * extinction
    sigmas = sigma_c * extinction
    square_devs = np.square(deltas / sigmas)

    # filter out infinities and nans
    square_devs = square_devs[np.isfinite(square_devs)]

    # DS94 divide by n - 1, where n is the number of points used, so we
    # do that here too
    chi2 = square_devs.sum() / (len(square_devs) - 1)
    return chi2


def plot_profile(ax, fc, logNHI):
    """Plot an extra profile of user-specified NHI.

    Ax needs to have been prepared properly (xlims already in their
    final form), and fc is the continuum fit.

    """
    extra_color = "g"
    x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 500)
    y = fc(x) * extinction_factor(logNHI, x)
    ax.plot(x, y, color=extra_color, label="user")


def plot_fit(ax, wavs, flux, fc, logNHI):
    cont_color = "m"
    lya_color = "xkcd:sky blue"

    # continuum fit
    fcs = fc(wavs)
    ax.plot(wavs, fcs, label="continuum fit", color=cont_color, zorder=30)

    # lya fit
    fms = fc(wavs) * extinction_factor(logNHI, wavs)
    ax.plot(wavs, fms, label="profile fit", color=lya_color, zorder=40)

    # data / used for cont / used for lya
    used_for_cont = safe_for_cont(wavs, flux)
    used_for_lya = safe_for_lya(wavs, flux)
    # change this to an axvspan plot
    ax.plot(
        wavs[used_for_cont],
        flux[used_for_cont],
        label="used for continuum",
        color=cont_color,
        linestyle="none",
        marker="o",
        markerfacecolor="none",
        alpha=0.5,
        zorder=50,
    )

    # finally, plot the spectrum and the rejected points
    ax.plot(wavs, flux, label="data", color="k", zorder=10)
    # use red x for rejected points
    ax.plot(
        wavs[np.logical_not(used_for_lya)],
        flux[np.logical_not(used_for_lya)],
        label="rejected",
        color="r",
        linestyle="none",
        marker="x",
        alpha=0.5,
        zorder=45,
    )

    ax.set_ylim([0, 1.1 * np.amax(flux[used_for_lya])])
    prepare_axes(ax)


def lya_fit(target, ax_fit=None, ax_chi2=None):
    """Fit logNHI using lya.

    target: string referring to any of the targets (see directories in
    ./data/)

    ax: axes to plot the result on

    """
    print("-" * 80)
    print(f"Fitting {target}".center(80))

    # obtain data
    wavs, flux, filename = get_spectrum.processed(target)
    # smooth (experimental)
    flux = boxcar_smooth(wavs, flux)
    # estimate continuum
    fc = estimate_continuum(wavs, flux)
    # sigma to use in chi2 equation
    sigma_c = estimate_noise(wavs, flux, fc)
    # choose clean parts of spectrum
    use = safe_for_lya(wavs, flux)

    # the fitting itself
    fargs = (fc, sigma_c, wavs[use], flux[use])
    result, chi2_min, NHIgrid, chi2grid = brute(
        chi2, [(19.0, 23.0)], args=fargs, Ns=2000, full_output=True
    )
    logNHI = result[0]

    # error estimation: sigma is where chi2 = chi2_min + 1. Don't use
    # bisect here because we don't know how well chi2 behaves.
    middle = np.argmin(np.abs(NHIgrid - logNHI))

    if np.amax(chi2grid[:middle]) < chi2_min + 1:
        print("Lower bound not found")
        lower = NHIgrid[np.argmax(chi2grid[:middle])]
    else:
        for i in reversed(range(middle)):
            if chi2grid[i] > chi2_min + 1:
                lower = NHIgrid[i]
                break

    if np.amax(chi2grid[middle:]) < chi2_min + 1:
        print("Upper bound not found")
        upper = NHIgrid[middle + np.argmax(chi2grid[middle:])]
    else:
        for i in range(middle, len(chi2grid)):
            if chi2grid[i] > chi2_min + 1:
                upper = NHIgrid[i]
                break

    print(
        f"logNHI={logNHI:.2f}, lower={lower:.2f}, upper={upper:.2f}, chi2={chi2_min:.2f}"
    )

    if ax_fit is not None:
        plot_fit(ax_fit, wavs, flux, fc, logNHI)

    if ax_chi2 is not None:
        ax_chi2.plot(NHIgrid, chi2grid, color="k")
        # ax_chi2.plot(NHIgrid, np.exp(-chi2grid), color="k")
        ax_chi2.set_xlabel("$\\log N(\\mathrm{H I})$ [cm$^{-2}$]")
        ax_chi2.axvspan(lower, upper, color="y", alpha=0.3)
        ax_chi2.axvline(logNHI, color="k", linestyle=":")
        # ax_chi2.set_ylabel("$\\exp(-\\chi^2)$")
        ax_chi2.set_ylabel("$\\chi^2$")

    info = dict(filename=filename, chi2=chi2)
    return logNHI, fc, info


def run_all():
    targets = []
    infos = []
    logNHIs = []

    for target in get_spectrum.target_use_which_spectrum:
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        logNHI, fc, info = lya_fit(target, ax_fit=ax1, ax_chi2=ax2)
        ax1.set_title(target + f"\nlogNHI = {logNHI:2f}")
        fig1.savefig(f"./lya-plots/{target}.pdf")
        fig2.savefig(f"./lya-plots/{target}_chi2.pdf")
        targets.append(target)
        infos.append(info)
        logNHIs.append(logNHI)

    col_names = ["target", "logNHI"]
    col_data = [targets, logNHIs]
    # add extra output from ly_fit like chi2, file names used, ...
    for k in infos[0]:
        col_names.append(k)
        col_data.append([info[k] for info in infos])

    overview = astropy.table.QTable(col_data, names=col_names)
    print(overview)
    overview.write("lyafitting_overview.dat", format="ascii", overwrite=True)


def run_one(target, compare=None):
    fig, [ax_fit, ax_chi2] = plt.subplots(1, 2)
    logNHI, fc, filename = lya_fit(target, ax_fit=ax_fit, ax_chi2=ax_chi2)
    plt.title(target, loc="right")
    if compare is not None:
        logNHIc = compare
        plot_profile(ax_fit, fc, logNHIc)

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
