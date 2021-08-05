"""Tools for getting spectra for lya fitting.

Includes choosing a data file for each star, reading the files, and
processing the spectral data (from either IUE, STIS, ...) into a format
that can be used directly for the fitting.

The variable target_use_which_spectrum indicates which data to use for
each star. It can be customized by editing this file. Running this
module directly will print out the default value for this dictionary.

"""
from astropy.table import Table
from astropy.io import fits
import numpy as np
from pathlib import Path
from warnings import warn
from scipy.interpolate import interp1d

# can be manually tweaked. TODO: If the value is a list or contains *, the spectra will be coadded
target_use_which_spectrum = {
    "HD097471": "data/HD097471/swp19375mxlo_vo.fits",
    "HD094493": "data/HD094493/mastDownload/HST/o54306010/o54306010_x1d.fits",
    "HD037525": "data/HD037525/swp27579.mxhi.gz",
    "HD093827": "data/HD093827/swp50536.mxhi.gz",
    "HD051013": "data/HD051013/swp22860.mxhi.gz",
    "HD096675": "data/HD096675/swp41717.mxhi.gz",
    "HD023060": "data/HD023060/swp11151mxlo_vo.fits",
    "HD099872": "data/HD099872/mastDownload/HST/o6lj0i020/o6lj0i020_x1d.fits",
    # "HD152248": "data/HD152248/swp54576.mxhi.gz",
    "HD152248": "data/HD152248/*.mxhi.gz",
    "HD209339": "data/HD209339/mastDownload/HST/o5lh0b010/o5lh0b010_x1d.fits",
    # "HD197770": "data/HD197770/mastDownload/HST/oedl04010/oedl04010_x1d.fits",
    "HD197770": "data/HD197770/swp49267.mxhi.gz",
    "HD037332": "data/HD037332/swp32289.mxhi.gz",
    "HD093028": "data/HD093028/swp05521.mxhi.gz",
    # "HD062542": "data/HD062542/mastDownload/HST/obik01020/obik01020_x1d.fits",
    # "HD062542": "data/HD062542/swp36353.mxhi.gz",
    "HD062542": "data/HD062542/swp29213mxlo_vo.fits",
    "HD190603": "data/HD190603/swp01822.mxhi.gz",
    # "HD046202": "data/HD046202/swp08845.mxhi.gz",
    # "HD046202": "data/HD046202/mastDownload/HST/ocb6e0030/ocb6e0030_x1d.fits",
    "HD046202": "data/HD046202/mastDownload/HST/**/*_x1d.fits",
    "HD047129": "data/HD047129/swp07077.mxhi.gz",
    "HD235874": "data/HD235874/swp34158mxlo_vo.fits",
    "HD216898": "data/HD216898/swp43934.mxhi.gz",
    "HD326329": "data/HD326329/swp48698.mxhi.gz",
    "HD179406": "data/HD179406/swp36939.mxhi.gz",
    "BD+52d3210": "data/BD+52d3210/swp34153mxlo_vo.fits",
    "BD+56d524": "data/BD+56d524/swp20330mxlo_vo.fits",
}


def processed(target):
    """Get spectrum data ready for fitting Lya for the given target.

    Tweak the variable get_spectrum.target_use_which_spectrum to choose
    the right data. Depending on whether a IUE or STIS spectrum was
    chosen, different steps will be taken. The end result is the
    spectral data in a common format, processed with different steps
    depending on the source of the data.

    Returns
    -------
    wav, flux: ndarray of wavelengths (angstrom) and fluxes (erg s-1 cm-2 angstrom-1)

    """
    # choose data
    filename = target_use_which_spectrum[target]
    print("Getting data from ", filename)
    wavs, flux, errs, rebin = auto_wavs_flux_errs(filename)

    if rebin:
        binnedwavs, binnedflux = bin_spectrum_around_lya(wavs, flux, errs)
    else:
        wavmin = 1150
        wavmax = 1300
        use = np.logical_and(wavmin < wavs, wavs < wavmax)
        binnedwavs, binnedflux = wavs[use], flux[use]

    # remove nans (these are very annoying when they propagate, e.g.
    # max([array with nan]) = nan).
    safe = np.isfinite(binnedflux)
    safewavs = binnedwavs[safe]
    safeflux = binnedflux[safe]
    return safewavs, safeflux, filename


def auto_wavs_flux_errs(filename):
    """Load spectrum or multiple spectra based on file name."""

    # determine if multiple files were provided. If a glob pattern was provided, this counts as
    if isinstance(filename, list):
        multiple_files = filename
    elif isinstance(filename, str):
        if "*" in filename:
            multiple_files = [str(p) for p in Path(".").glob(filename)]
        else:
            multiple_files = None
    else:
        warn("filename should be str or list!")
        raise

    if multiple_files is None:
        if "x1d" in filename:
            wavs, flux, errs = merged_stis_data(filename)
            rebin = True
        elif "mxhi" in filename:
            wavs, flux, errs = merged_iue_h_data(filename)
            rebin = True
        elif "mxlo" in filename:
            wavs, flux, errs = iue_l_data(filename)
            rebin = False
        else:
            warn("File {} not supported yet, exiting".format(filename))
            exit()
    else:
        if "mxhi" not in multiple_files[0]:
            warn("Only coadding of mxhi is supported")
            raise

        wavs, flux, errs = coadd_iue_h(multiple_files)
        rebin = False

    return wavs, flux, errs, rebin


def merged_stis_data(filename, get_net=False):
    """Get wavelengths, fluxes and errors from all STIS spectral orders.

    get_net : add net column to output

    Returns
    -------

    wavs: numpy array, all wavelengths, sorted

    flux: all fluxes at these wavelengths

    errs: all errors at these wavelengths

    """
    t = Table.read(filename)
    output_columns = ["WAVELENGTH", "FLUX", "ERROR"]
    if get_net:
        output_columns.append("NET")

    output = [np.concatenate(t[c]) for c in output_columns]
    # sort by wavelength
    idxs = np.argsort(output[0])
    return [array[idxs] for array in output]


def merged_iue_h_data(filename, extra_columns=None):
    """
    Get wavelengths, fluxes and errors from IUE high res data.

    Returns
    -------
    column_values : list containing wavelengths, "ABS_CAL", "NOISE" plus
        any of the extra_columns
    """
    t = Table.read(filename)

    def iue_wavs(i):
        return t[i]["WAVELENGTH"] + t[i]["DELTAW"] * np.arange(t[i]["NPOINTS"])

    def pixrange(i):
        return slice(t[i]["STARTPIX"], t[i]["STARTPIX"] + t[i]["NPOINTS"])

    def all_of_column(colname):
        return np.concatenate([t[i][colname][pixrange(i)] for i in range(len(t))])

    allwavs = np.concatenate([iue_wavs(i) for i in range(len(t))])
    idxs = np.argsort(allwavs)

    colnames = ["WAVELENGTH", "ABS_CAL", "NOISE"]
    if extra_columns is not None:
        colnames += extra_columns

    column_values = [allwavs[idxs]]
    for colname in colnames[1:]:
        array = all_of_column(colname)
        # already sorted by wavelength here
        column_values.append(array[idxs])

    # clean up using DQ
    alldq = all_of_column("QUALITY")
    goodDQ = alldq == 0
    for array in column_values:
        array = array[goodDQ]

    return column_values


def coadd_iue_h(filenames):
    return coadd_general(
        filenames, lambda x: merged_iue_h_data(x, extra_columns=["NET"])
    )


def coadd_general(filenames, wavs_flux_errs_net_get_function):
    """General function for coadding spectra in filenames

    The second argument should be a function that takes a file name, and
    returns [wavs, flux, errs, net].

    Returns
    -------
        coadded wavs, flux, errs
    """
    num_files = len(filenames)

    # get all the per-wavelength data
    all_wavs = []
    all_flux = []
    all_errs = []
    all_net = []
    for i in range(num_files):
        wavs, flux, errs, net = wavs_flux_errs_net_get_function(filenames[i])
        all_wavs.append(wavs)
        all_flux.append(flux)
        all_errs.append(errs)
        all_net.append(net)

    # determine new wavelength grid, using max of median of wavelength
    # increment as step size
    maxwav = np.amax(all_wavs)
    minwav = np.amin(all_wavs)
    disp = np.amax([np.median(np.diff(w)) for w in all_wavs])
    newwavs = np.arange(minwav, maxwav, disp)

    # instead of binning, we're just going to do nearest neighbour on a
    # slightly coarser wavelength grid. It worked for Julia, so...
    flux_sum = np.zeros(len(newwavs))
    weight_sum = np.zeros(len(newwavs))
    variance_sum = np.zeros(len(newwavs))
    for i in range(num_files):
        # get exposure time from the primary hdu. The keyword can have
        # different names sometimes.
        header = fits.getheader(filenames[i], ext=0)
        for exptime_key in ("LEXPTIME", "SEXPTIME"):
            if exptime_key in header:
                exptime = float(header[exptime_key])
                break

        # nearest neighbour interpolation of all relevant quantities
        def do_interp1d(quantity):
            return interp1d(
                all_wavs[i],
                quantity,
                kind="nearest",
                fill_value=np.nan,
                bounds_error=False,
            )(newwavs)

        fi = do_interp1d(all_flux[i])
        ei = do_interp1d(all_errs[i])
        ni = do_interp1d(all_net[i])

        # total_counts = flux * sensitivity * exptime
        # --> flux = total_counts / (sensitivity * exptime)
        #
        # V(flux) = V(total_counts) / (sensitivity * exptime)**2
        #         = total_counts / (sensitivity * exptime)**2 (poisson)
        #         = flux * sensitivity * exptime / (sensitivity * exptime)**2
        #         = flux / (sensitivity * exptime)

        # counts per flux unit
        sensitivity = ni / fi
        weights = sensitivity * exptime
        weight_sum += weights
        flux_sum += weights * fi
        variance_sum += np.square(ei * weights)

    flux_result = flux_sum / weight_sum
    errs_result = np.sqrt(variance_sum) / weight_sum
    return newwavs, flux_result, errs_result


def iue_l_data(filename):
    t = Table.read(filename)
    wavs = t["WAVE"]
    flux = t["FLUX"]
    sigma = t["SIGMA"]
    return wavs, flux, sigma


def bin_spectrum_around_lya(wavs, flux, errs):
    """
    Rebin spectrum to for lya fitting, and reject certain points.

    A rebinning of the spectrum to make it more useful for lya fitting.
    Every new point is the weighted average of all data within the range
    of a bin. The weights are 1 / errs**2. The bins are chosen as 1000
    equally spaced intervals, from 1150 to 1280 angstrom. **subject to
    change**

    Additionally, only the points that satisfy some basic data rejection
    criteria are used. E.g flux > 0.

    Returns
    -------
    newwavs: average wavelength in each bin
    newflux: average flux in each bin

    """
    # the bin details are hardcoded here
    numbins = 1000
    wavmin = 1150
    wavmax = 1280
    wavbins = np.linspace(wavmin, wavmax, numbins, endpoint=True)

    # np.digitize returns list of indices. b = 1 means that the data point
    # is between wav[0] (first) and wav[1]. b = n-1 means between wav[n-2]
    # and wav[n-1] (last). b = 0 or n mean out of range.
    bs = np.digitize(wavs, wavbins)
    newwavs = np.zeros(len(wavbins) - 1)
    newflux = np.zeros(len(wavbins) - 1)
    for i in range(0, len(wavbins) - 1):
        in_bin = bs == i + 1  # b runs from 1 to n-1
        use = np.logical_and(in_bin, flux > 0)
        weights = 1 / np.square(errs[use])

        # if a bin is empty or something else is wrong, the nans will be
        # filtered out later
        if not use.any() or weights.sum() == 0:
            newwavs[i] = 0
            newflux[i] = np.nan
        else:
            newwavs[i] = np.average(wavs[use], weights=weights)
            newflux[i] = np.average(flux[use], weights=weights)

    return newwavs, newflux


def get_exptime(header):
    """Tries a couple of keywords to find the exposure time in a FITS header"""
    for exptime_key in ("EXPTIME", "LEXPTIME", "SEXPTIME"):
        if exptime_key in header:
            exptime = float(header[exptime_key])
            return exptime


# Some code to generate the above dict from scratch. Manual tweaking can
# occur after.
if __name__ == "__main__":
    gen_dict = {}
    here = Path(".")
    for d in list(here.glob("./data/HD*")) + list(here.glob("./data/BD*")):
        has_iue_h = False
        has_iue_l = False
        has_hst_stis = False
        # has_hst_cos = False

        # lower in this list of ifs is higher priority
        target = Path(d).name

        # def set_if_exists(glob_pattern):
        #     files = d.glob(glob_pattern)
        #     if len(files) > 0:
        #         spectrum_file = files[0]

        iue_l_files = list(d.glob("*mxlo_vo.fits"))
        if len(iue_l_files) > 0:
            spectrum_file = str(iue_l_files[0])

        iue_h_files = list(d.glob("*mxhi.gz"))
        if len(iue_h_files) > 0:
            spectrum_file = str(iue_h_files[0])

        hst_stis_files = list(d.glob("**/*x1d.fits"))
        if len(hst_stis_files) > 0:
            spectrum_file = str(hst_stis_files[0])

        gen_dict[target] = spectrum_file

    print(gen_dict)
