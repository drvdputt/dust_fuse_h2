import get_spectrum
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import brute
import numpy as np
import argparse
import warnings
import astropy
from astropy import stats
from astropy.table import Table

warnings.filterwarnings("ignore", category=astropy.units.UnitsWarning)

LYA = 1215.67

# see figure 1 in DS94
default_noise_wav_range_DS94 = [1265, 1271]

# None is specified, these are used. Useful for making a test plots of
# sources for which the good ranges are unknown yet.
default_continuum_wav_ranges = [[1150, 1175], [1265, 1271]]
default_lya_wav_ranges = [[1175, 1200], [1225, 1250]]

# manually choose range for continuum fit and lya fit for each target
target_continuum_wav_ranges = {
    "BD+52d3210": [[1155, 1168], [1266, 1291], [1311, 1328], [1342, 1372]],
    "BD+56d524": [[1265, 1275], [1282, 1296], [1309, 1321], [1348, 1382]],
    "HD023060": [[1165, 1171], [1266, 1292], [1315, 1323]],
    "HD037332": [
        [1150, 1153],
        [1159, 1162],
        [1163, 1167],
        [1169, 1170],
        [1171, 1173],
        [1177, 1183],
        [1254, 1258],
        [1313, 1323],
        [1338, 1344],
        [1355, 1361],
    ],
    "HD037525": [
        [1151, 1152.5],
        [1254, 1259],
        [1163, 1173],
        [1281, 1293],
        [1312.5, 1323],
        [1348, 1361],
        [1372, 1368],
        [1366, 1388],
    ],
    "HD046202": [
        [1159, 1174],
        [1177.5, 1182],
        [1309, 1315],
        [1349, 1354],
        [1266, 1276],
    ],
    "HD047129": [[1155, 1168], [1177.5, 1181], [1283.4, 1294.2], [1309, 1317]],
    "HD051013": [
        [1170, 1174.5],
        [1253.5, 1258.5],
        [1268.1, 1274.5],
        [1281.5, 1286.7],
        [1338.3, 1360.5],
        [1371, 1377.5],
        [1381, 1384],
    ],
    "HD062542": [
        [1150, 1160],
        [1142.5, 1158.5],
        [1280.5, 1292.6],
        [1317.5, 1331],
        [1365.5, 1391],
    ],
    "HD093028": [[1162.5, 1167.8], [1283.5, 1295.5], [1272, 1276.5], [1320, 1327.5]],
    "HD093827": [
        [1178, 1186],
        # [1264, 1275],
        [1287.5, 1292],
        [1313, 1323.2],
        [1336.5, 1341.5],
    ],
    "HD094493": [
        [1162, 1172],
        [1178, 1183],
        [1185.5, 1188.5],
        [1270, 1275],
        [1317.5, 1320.5],
    ],
    "HD096675": [[1254.5, 1258.5], [1281.3, 1291], [1314, 1323], [1338, 1344]],
    "HD097471": [[1158, 1165], [1308, 1332], [1350, 1371]],
    "HD099872": [
        [1146, 1157.7],
        [1160, 1166],
        [1169, 1172],
        [1178.5, 1187],
        [1281, 1293],
        [1312, 1323.5],
    ],
    "HD152248": [
        [1155, 1162],
        [1177, 1182],
        [1262.5, 1276],
        [1288, 1295.5],
        [1310, 1317.5],
    ],
    "HD179406": [
        [1159, 1162],
        [1254.5, 1258],
        [1266.8, 1272],
        [1281, 1291],
        [1313, 1323],
    ],
    "HD190603": [  # choosing these based on low res data
        [1158, 1170],
        # [1176.5, 1188.5],
        # [1254.5, 1257.5],
        [1266.4, 1283],
        [1308, 1323],
        [1348, 1373],
    ],
    "HD197770": [  # this one is very fiddly
        # [1168, 1171.8],
        [1251, 1256],
        [1272, 1292],
        [1315, 1323],
        [1239.4, 1242.9],
        [1330.5, 1333.7],
        # [1353.5, 1372],
    ],
    "HD209339": [
        [1150, 1174],
        [1177.3, 1188],
        [1263, 1276],
        [1282, 1294],
        [1305.5, 1323],
    ],
    "HD216898": [[1272, 1276], [1282, 1299], [1308, 1327], [1350, 1377]],
    "HD235874": [[1153, 1168], [1265.5, 1291], [1308, 1329], [1350, 1371]],
    "HD326329": [[1177, 1182], [1262.3, 1270], [1308, 1324.5]],
    "HD045314": [[1288, 1294], [1177.5, 1182], [1167.5, 1171.5], [1262, 1272.5]],
}

target_lya_wav_ranges = {
    "BD+52d3210": [[1155, 1168], [1266, 1291], [1180, 1210], [1222, 1249]],
    "BD+56d524": [[1192, 1207], [1224, 1234], [1241, 1254]],
    "HD023060": [[1194, 1210], [1220, 1245]],
    "HD037332": [
        [1179, 1183],
        [1184, 1187],
        [1196, 1197],
        [1198, 1199],
        [1202, 1205.2],
        [1221.3, 1232],
        [1235, 1241],
        [1244, 1246],
    ],
    "HD037525": [
        [1177, 1187],
        [1191, 1192.5],
        [1195.4, 1197],
        [1201.5, 1206],
        [1220, 1233],
        [1234.5, 1241.5],
        [1242.5, 1249.5],
    ],
    "HD046202": [
        [1177.5, 1182],
        [1185, 1188.5],
        [1195, 1197],
        [1202, 1206],
        [1207.5, 1210],
        [1222, 1236],
        [1243.5, 1246.5],
    ],
    "HD047129": [
        [1177.5, 1181],
        [1186.6, 1188.7],
        [1194.5, 1198.3],
        [1201.5, 1206],
        [1208, 1211],
        [1222, 1227.5],
        [1255.5, 1259],
    ],
    "HD051013": [
        [1179, 1187],
        [1195.5, 1199],
        [1191.4, 1192.6],
        [1202, 1203.7],
        [1209, 1212],
        [1219, 1228.5],
        [1234.5, 1241.5],
    ],
    "HD062542": [[1184.5, 1211.5], [1223, 1258.5]],
    "HD093028": [
        [1180, 1189],
        [1201, 1206],
        [1207, 1212],
        [1221, 1229.5],
        [1243.3, 1246.5],
    ],
    "HD093827": [
        [1202, 1206],
        [1198, 1199],
        [1195.5, 1196.5],
        [1187.4, 1189.5],
        [1221, 1233.4],
        [1240, 1242],
    ],
    "HD094493": [
        [1190.7, 1193],
        [1195, 1199],
        [1202.5, 1205.5],
        [1207.5, 1212],
        [1220, 1235],
        [1238, 1240],
        [1251, 1253],
    ],
    "HD096675": [
        [1195, 1200],
        [1191, 1192],
        [1228, 1233],
        [1237.8, 1242],
        [1254.5, 1258.5],
    ],
    "HD097471": [[1179.5, 1188], [1202, 1212], [1221, 1286]],
    "HD099872": [
        [1195, 1197],
        [1198, 1199.3],
        [1202, 1203.5],
        [1208, 1211.5],
        [1221, 1247],
    ],
    "HD152248": [
        [1185.5, 1190],
        [1195.5, 1198],
        [1201.5, 1205.5],
        [1207, 1212],
        [1221, 1228],
    ],
    "HD179406": [
        [1184.5, 1189.5],
        [1190.8, 1192.3],
        [1197.8, 1199.3],
        [1201, 1202.5],
        [1209, 1212],
        [1222, 1232.8],
        [1234.3, 1246.5],
    ],
    "HD190603": [
        [1201.3, 1203.3],
        [1206.3, 1207],
        [1195.7, 1197.8],
        [1226.3, 1240.8],
        [1241.5, 1245],
    ],
    "HD197770": [
        # [1177, 1192],
        # [1195, 1200],
        [1208.5, 1212],
        [1218, 1223.5],
        [1224.4, 1228.1],
        [1234, 1238],
    ],
    "HD209339": [
        [1194.5, 1199],
        [1201, 1206],
        [1207, 1212],
        [1220, 1236.5],
        [1239, 1246.5],
    ],
    "HD216898": [
        # [1184, 1192],
        [1194.5, 1199],
        [1201, 1202.5],
        [1226, 1233],
        [1237, 1246],
        [1254, 1256.5],
        [1262, 1268],
    ],
    "HD235874": [[1178.5, 1205], [1223, 1258]],
    "HD326329": [
        [1187.5, 1190.3],
        [1195.5, 1199],
        [1204.5, 1207],
        [1221, 1236],
        [1243, 1247],
    ],
    "HD045314": [[1196, 1199], [1201.5, 1206], [1207.6, 1212.5], [1218, 1225]],
}


# keeping these old settings as a comment, as the number might be useful
# default_exclude_wav_ranges = [
#     [1171, 1178],
#     [1181, 1185],
#     [1190, 1191],
#     [1192, 1195],
#     [1198, 1201.4],
#     [1205.5, 1207],
#     [1227.6, 1239],
#     [1249, 1252],
#     [1258.5, 1262],
# ]
# target_exclude_wav_ranges = {
#     "BD+52d3210": [
#         [1209, 1221],  # geocoronal
#         [1247, 1266],
#         [1236.5, 1240],
#         [1171, 1180],
#         [1194, 1199],
#     ],
#     "BD+56d524": [
#         [1206, 1223],  # geocoronal
#         [1247, 1266],
#         [1236.5, 1240],
#         [1171, 1180],
#     ],
#     "HD047129": default_exclude_wav_ranges + [[1242, 1250]],
#     "HD062542": [
#         [1206, 1223],  # geocoronal
#         [1247, 1266],
#         [1236.5, 1240],
#         [1171, 1180],
#     ],
#     "HD152248": default_exclude_wav_ranges + [[1242, 1250], [1165.6, 1177]],
# }


def prepare_axes(ax):
    ax.set_xlabel("wavelength ($\\AA$)")
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
        in_range = in_range | ((wavs > lo) & (wavs < hi))
    return in_range


def is_good_data(wavs, flux, target):
    peak = is_peak(wavs, flux)
    return np.logical_not(peak)


def is_peak(wavs, flux):
    """Return mask to avoid peaks in the spectrum."""
    masked_array = stats.sigma_clip(flux, sigma=4)  # uses 5 iterations by default
    return masked_array.mask


def use_for_cont(wavs, flux, target):
    """Return mask indicating wavelengths for continuum fit."""
    # use only these points for continuum estimation
    use = wavs_in_ranges(wavs, target_continuum_wav_ranges[target])
    good = is_good_data(wavs, flux, target)
    return use & good


def use_for_lya(wavs, flux, target):
    """Return mask indicating wavelengths for lya fit."""
    # at these wavelengths, 1e22 * cross is about 100
    # exp(100) ~ e43 is still relatively safe
    # center = wavs_in_ranges(wavs, [[1212.67, 1218.67]])
    use = wavs_in_ranges(wavs, target_lya_wav_ranges[target])
    good = is_good_data(wavs, flux, target)
    return use & good


def estimate_continuum(wavs, flux, target):
    """Estimate the continuum using a linear fit.

    Specific wavelength ranges are used.

    """
    use = use_for_cont(wavs, flux, target)

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
    """Estimate noise as RMS of (data - continuum).

    Wavelength ranges used to fit continuum are used.

    """
    use = use_for_cont(wavs, flux, target)
    # DS94 use sum instead of average. Not sure if that is the correct
    # way.
    sigma = np.sqrt(np.average(np.square(flux[use] - fc(wavs[use]))))
    return sigma


def cross(l):
    l0 = 1215.67
    return 4.26e-20 / (6.04e-10 + np.square(l - l0))


def extinction_factor(lognhi, l):
    return np.exp(-np.power(10.0, lognhi) * cross(l))


def chi2(lognhi, fc, sigma_c, wavs, flux):
    # overflows easily. Ignore fit points where overflow occurs
    extf = extinction_factor(lognhi, wavs)
    physical_model = fc(wavs) * extf
    # not sure if we want to scale noise with extinction
    # noise_model = sigma_c * extf
    noise_model = sigma_c
    square_devs = np.square((physical_model - flux) / noise_model)

    # filter out infinities and nans
    square_devs = square_devs[np.isfinite(square_devs)]

    # DS94 divide by n - 1, where n is the number of points used
    chi2 = square_devs.sum() / (len(square_devs) - 1)
    return chi2


def plot_profile(ax, fc, lognhi):
    """Plot an extra profile of user-specified nhi.

    Ax needs to have been prepared properly (xlims already in their
    final form), and fc is the continuum fit.

    """
    extra_color = "g"
    x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 500)
    y = fc(x) * extinction_factor(lognhi, x)
    ax.plot(x, y, color=extra_color, label="user")


def plot_fit(target, ax, wavs, flux, fc, lognhi, lower_upper=None):
    """Plot lya model, continuum model, and data

    target: target name

    ax: Axes object to use

    wavs: wavelengths to use for x

    flux: flux data to which the fit was performed

    fc: function of wavelength, representing the continuum model

    lognhi: the fit result

    upper_lower: tuple (upper, lower), used to visualize the uncertainty on lognhi

    """

    cont_color = "m"
    lya_color = "xkcd:sky blue"

    # continuum fit
    fcs = fc(wavs)
    ax.plot(wavs, fcs, label="cont. model", color=cont_color, zorder=30)

    # lya extinction
    e = extinction_factor(lognhi, wavs)

    # lya fit
    fms = fcs * e
    ax.plot(wavs, fms, label="model", color=lya_color, zorder=40)

    # data
    ax.plot(wavs, flux, label="data", color="k", zorder=10, linewidth=1)
    # reconstructed continuum
    rec = flux / e
    safe = np.isfinite(rec) & (rec <= np.amax(flux))
    ax.plot(wavs[safe], rec[safe], label="reconstructed", color="r", alpha=0.5)

    used_for_cont = use_for_cont(wavs, flux, target)
    ax.plot(
        wavs[used_for_cont],
        flux[used_for_cont],
        label="cont. points",
        color=cont_color,
        linestyle="none",
        marker="o",
        markerfacecolor="none",
        alpha=0.5,
        zorder=50,
    )
    used_for_lya = use_for_lya(wavs, flux, target)
    ax.plot(
        wavs[used_for_lya],
        flux[used_for_lya],
        label="fit points",
        color="b",
        linestyle="none",
        marker="+",
        markerfacecolor="none",
        zorder=50,
    )
    # rejections
    # bad = np.logical_not(is_good_data(wavs, flux, target))
    # ax.plot(
    #     wavs[bad],
    #     flux[bad],
    #     label="bad",
    #     color="r",
    #     linestyle="none",
    #     marker="x",
    #     zorder=60,
    # )

    # uncertainty
    if lower_upper is not None:
        lower_fms = fcs * extinction_factor(lower_upper[0], wavs)
        upper_fms = fcs * extinction_factor(lower_upper[1], wavs)
        ax.fill_between(wavs, lower_fms, upper_fms, alpha=0.3, color=lya_color)

    ax.set_ylim([0, 1.1 * np.percentile(flux, 99)])
    prepare_axes(ax)


def lya_fit(target, ax_fit=None, ax_chi2=None, ax_p=None):
    """Fit lognhi using lya.

    target: string referring to any of the targets (see directories in
    ./data/)

    ax: axes to plot the result on

    """
    print("-" * 80)
    print(f"Fitting {target}".center(80))

    # obtain data
    wavs, flux, filename = get_spectrum.processed(target, wmax=2000)
    # estimate continuum
    fc = estimate_continuum(wavs, flux, target)
    # sigma to use in chi2 equation
    sigma_c = estimate_noise(wavs, flux, fc, target)
    # choose clean parts of spectrum
    use = use_for_lya(wavs, flux, target)

    # the fitting itself
    fargs = (fc, sigma_c, wavs[use], flux[use])
    result, chi2_min, nhigrid, chi2grid = brute(
        chi2, [(19.0, 23.0)], args=fargs, Ns=2000, full_output=True
    )
    lognhi = result[0]

    # normalized pdf for error estimation
    P = np.exp(-chi2grid)
    P /= sum(P)
    Pcumul = np.cumsum(P)

    # find where cumulative P becomes 16 and 84
    # find percentiles by interpolating inverse of cumulative (= nhi as
    # a function of percentile)
    q84 = 0.8413447460685429
    q16 = 0.15865525393145707
    [p16, p84] = np.interp([q16, q84], Pcumul, nhigrid)
    # Estimate for 1 sigma. Is correct if shape of P is gaussian.
    unc = (p84 - p16) / 2

    if ax_fit is not None:
        plot_fit(target, ax_fit, wavs, flux, fc, lognhi, (p16, p84))

    if ax_chi2 is not None:
        color = "xkcd:dark red"
        ax_chi2.plot(nhigrid, chi2grid, color=color)
        ax_chi2.axvline(lognhi, color=color, linestyle=":")
        ax_chi2.set_xlabel("$\\log N(\\mathrm{H I})$ [cm$^{-2}$]")
        ax_chi2.set_ylabel("$\\chi^2$", color=color)
        ax_chi2.set_title(f"$\\chi^2_{{\\mathrm{{min}}}} = {chi2_min:.2f}$")

    if ax_p is not None:
        color = "b"
        plt.plot(nhigrid, P, color=color)
        ax_p.set_ylabel("PDF", color=color, zorder=2)
        ax_p.axvspan(p16, p84, color=color, alpha=0.3, zorder=1)
        # zorder problem: axvspan should be below the other plot
        if ax_chi2 is not None:
            # set zorder of axes
            ax_p.set_zorder(ax_chi2.get_zorder() - 1)
            # make canvas of top axes transparent
            ax_chi2.patch.set_visible(False)

    info = dict(lognhi_unc=unc, chi2=chi2_min)
    return lognhi, fc, info


def run_all():
    targets = []
    infos = []
    lognhis = []

    for target in get_spectrum.target_use_which_spectrum:
        if target_continuum_wav_ranges[target] is None:
            target_continuum_wav_ranges[target] = default_continuum_wav_ranges
        if target_lya_wav_ranges[target] is None:
            target_lya_wav_ranges[target] = default_lya_wav_ranges

        fig1, [ax_left, ax_right_l] = plt.subplots(1, 2, figsize=(8, 4))
        ax_right_r = ax_right_l.twinx()
        lognhi, fc, info = lya_fit(
            target, ax_fit=ax_left, ax_chi2=ax_right_l, ax_p=ax_right_r
        )

        set_title(ax_left, target, lognhi)
        fig1.tight_layout()
        fig1.savefig(f"./lya-plots/{target}.pdf")
        targets.append(target)
        infos.append(info)
        lognhis.append(lognhi)

    col_names = ["target", "lognhi"]
    col_data = [targets, lognhis]
    # add extra output from ly_fit like chi2, file names used, ...
    for k in infos[0]:
        col_names.append(k)
        col_data.append([info[k] for info in infos])

    overview = astropy.table.QTable(col_data, names=col_names)
    print(overview)

    print("5 largest uncertainties:")
    unc_sort = np.argsort(overview["lognhi_unc"])
    largest_unc_names = overview[unc_sort][-5:]
    print(largest_unc_names)

    overview.write("lyafitting_overview.dat", format="ascii", overwrite=True)
    return overview


def run_one(target, compare=None):
    if target_continuum_wav_ranges.get(target, None) is None:
        target_continuum_wav_ranges[target] = default_continuum_wav_ranges
    if target_lya_wav_ranges.get(target, None) is None:
        target_lya_wav_ranges[target] = default_lya_wav_ranges

    fig, [ax_fit, ax_chi2] = plt.subplots(1, 2, figsize=(9, 6))
    ax_p = ax_chi2.twinx()
    lognhi, fc, info = lya_fit(target, ax_fit=ax_fit, ax_chi2=ax_chi2, ax_p=ax_p)
    print(f"Fit result: lognhi = {lognhi} ; unc = {info['lognhi_unc']}")
    set_title(ax_fit, target, lognhi)
    if compare is not None:
        lognhic = compare
        plot_profile(ax_fit, fc, lognhic)

    plt.show()


def set_title(ax, target, lognhi):
    data_filename = get_spectrum.target_use_which_spectrum[target]
    data_type = None
    if "x1d" in data_filename:
        data_type = "STIS"
    elif "mxhi" in data_filename:
        data_type = "IUE H"
    elif "mxlo" in data_filename:
        data_type = "IUE L"

    ax.set_title(target + f"\nlognhi = {lognhi:2f} ({data_type})")


def update_catalog(overview_table, original_file):
    """Write the fit results into updated data table."""
    # do not overwrite the recent data from jenkins (ref nr 15)
    jenkins2019_index = 15

    old_data = Table.read(original_file, format="ascii.commented_header")
    for row in overview_table:
        old_index = np.where(old_data["name"].data == row["target"])[0][0]

        if old_data[old_index]["hiref"] != jenkins2019_index:
            old_data[old_index]["hiref"] = 1
            old_data[old_index]["lognhi"] = row["lognhi"]
            old_data[old_index]["lognhi_unc"] = row["lognhi_unc"]

    two_decimals = "{:.2f}"
    formats = {"lognhi": two_decimals, "lognhi_unc": two_decimals}
    old_data.write(
        "data/fuse_h1_h2_with_lyafitting.dat",
        format="ascii.commented_header",
        overwrite=True,
        formats=formats,
    )
    # also write in fits format, so we can open in topcat
    old_data.write("data/fuse_h1_h2_with_lyafitting.fits", overwrite=True)


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
        help="Plot extra profile using this lognhi value",
    )
    ap.add_argument("--update_catalog", default=None, help="File name")
    args = ap.parse_args()

    if args.target == "all":
        overview = run_all()
        if args.update_catalog is not None:
            update_catalog(overview, args.update_catalog)
    else:
        run_one(args.target, args.compare)


if __name__ == "__main__":
    main()
