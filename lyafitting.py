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

# see figure 1 in DS94
default_noise_wav_range_DS94 = [1265, 1271]

# use a different value for certain targets
target_noise_wav_range = {"BD+56d524": [1243, 1299], "HD062542": [1243, 1282]}

# per-target wavelength exclusions (due to spectroscopic features seen
# by eye)
default_exclude_wav_ranges = [
    [1171, 1178],
    [1181, 1185],
    [1190, 1191],
    [1192, 1195],
    [1198, 1201.4],
    [1205.5, 1207],
    [1227.6, 1239],
    [1249, 1252],
    [1258.5, 1262],
]
target_exclude_wav_ranges = {
    "BD+52d3210": [
        [1209, 1221],  # geocoronal
        [1247, 1266],
        [1236.5, 1240],
        [1171, 1180],
        [1194, 1199],
    ],
    "BD+56d524": [
        [1206, 1223],  # geocoronal
        [1247, 1266],
        [1236.5, 1240],
        [1171, 1180],
    ],
    "HD047129": default_exclude_wav_ranges + [[1242, 1250]],
    "HD062542": [
        [1206, 1223],  # geocoronal
        [1247, 1266],
        [1236.5, 1240],
        [1171, 1180],
    ],
    "HD152248": default_exclude_wav_ranges + [[1242, 1250], [1165.6, 1177]],
}


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
        in_range = in_range | ((wavs > lo) & (wavs < hi))
    return in_range


def is_good_data(wavs, flux, target):
    peak = is_peak(wavs, flux)

    exclude_wav_ranges = target_exclude_wav_ranges.get(
        target, default_exclude_wav_ranges
    )
    excluded = wavs_in_ranges(wavs, exclude_wav_ranges)

    return np.logical_not(peak | excluded)


def is_peak(wavs, flux):
    """Return mask to avoid peaks in the spectrum."""
    masked_array = stats.sigma_clip(flux)  # uses 5 iterations by default
    return masked_array.mask


def safe_for_cont(wavs, flux, target):
    """Return mask that indicates wavelengths for continuum fit."""
    # use only these points for continuum estimation

    noise_wav_range = target_noise_wav_range.get(target, default_noise_wav_range_DS94)
    cont_wav_ranges = [
        noise_wav_range,
        [1165, 1170],
        [1179, 1181],
        [1185, 1188],
    ]
    use = wavs_in_ranges(wavs, cont_wav_ranges)
    good = is_good_data(wavs, flux, target)
    return use & good


def safe_for_lya(wavs, flux, target):
    # at these wavelengths, 1e22 * cross is about 100
    # exp(100) ~ e43 is still relatively safe
    center = wavs_in_ranges(wavs, [[1212.67, 1218.67]])
    good = is_good_data(wavs, flux, target)
    return np.logical_not(center) & good


def estimate_continuum(wavs, flux, target):
    """Estimate the continuum using a linear fit"""
    use = safe_for_cont(wavs, flux, target)

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


def estimate_noise(wavs, flux, fc, target):
    use = wavs_in_ranges(wavs, [default_noise_wav_range_DS94]) & safe_for_cont(
        wavs, flux, target
    )
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


def plot_fit(target, ax, wavs, flux, fc, logNHI, lower_upper=None):
    """Plot lya model, continuum model, and data

    target: target name

    ax: Axes object to use

    wavs: wavelengths to use for x

    flux: flux data to which the fit was performed

    fc: function of wavelength, representing the continuum model

    logNHI: the fit result

    upper_lower: tuple (upper, lower), used to visualize the uncertainty on logNHI

    """

    cont_color = "m"
    lya_color = "xkcd:sky blue"

    # continuum fit
    fcs = fc(wavs)
    ax.plot(wavs, fcs, label="continuum fit", color=cont_color, zorder=30)

    # lya fit
    fms = fcs * extinction_factor(logNHI, wavs)
    ax.plot(wavs, fms, label="profile fit", color=lya_color, zorder=40)

    # data / used for cont / used for lya
    used_for_cont = safe_for_cont(wavs, flux, target)
    used_for_lya = safe_for_lya(wavs, flux, target)
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

    if lower_upper is not None:
        lower_fms = fcs * extinction_factor(lower_upper[0], wavs)
        upper_fms = fcs * extinction_factor(lower_upper[1], wavs)
        ax.fill_between(wavs, lower_fms, upper_fms, alpha=0.3, color=lya_color)

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
    # flux = boxcar_smooth(wavs, flux)
    # estimate continuum
    fc = estimate_continuum(wavs, flux, target)
    # sigma to use in chi2 equation
    sigma_c = estimate_noise(wavs, flux, fc, target)
    # choose clean parts of spectrum
    use = safe_for_lya(wavs, flux, target)

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
        plot_fit(target, ax_fit, wavs, flux, fc, logNHI, (lower, upper))

    if ax_chi2 is not None:
        ax_chi2.plot(NHIgrid, chi2grid, color="k")
        # ax_chi2.plot(NHIgrid, np.exp(-chi2grid), color="k")
        ax_chi2.set_xlabel("$\\log N(\\mathrm{H I})$ [cm$^{-2}$]")
        ax_chi2.axvspan(lower, upper, color="y", alpha=0.3)
        ax_chi2.axvline(logNHI, color="k", linestyle=":")
        # ax_chi2.set_ylabel("$\\exp(-\\chi^2)$")
        ax_chi2.set_ylabel("$\\chi^2$")

    info = dict(filename=filename, chi2=chi2_min)
    return logNHI, fc, info


def run_all():
    targets = []
    infos = []
    logNHIs = []

    for target in get_spectrum.target_use_which_spectrum:
        fig1, [ax1, ax2] = plt.subplots(1, 2, figsize=(8, 4))
        logNHI, fc, info = lya_fit(target, ax_fit=ax1, ax_chi2=ax2)
        data_filename = get_spectrum.target_use_which_spectrum[target]
        data_type = None
        if "x1d" in data_filename:
            data_type = "STIS"
        elif "mxhi" in data_filename:
            data_type = "IUE H"
        elif "mxlo" in data_filename:
            data_type = "IUE L"

        ax1.set_title(target + f"\nlogNHI = {logNHI:2f} ({data_type})")
        fig1.tight_layout()
        fig1.savefig(f"./lya-plots/{target}.pdf")
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
