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
import collections

# can be manually tweaked. TODO: If the value is a list or contains *, the spectra will be coadded
target_use_which_spectrum = {
    "HD097471": "data/HD097471/swp19375mxlo_vo.fits",
    "HD094493": "data/HD094493/mastDownload/HST/o54306010/o54306010_x1d.fits",
    "HD037525": "data/HD037525/swp27579.mxhi.gz",
    "HD093827": "data/HD093827/swp50536.mxhi.gz",
    "HD051013": "data/HD051013/swp22860.mxhi.gz",
    "HD096675": "data/HD096675/swp41717.mxhi.gz",
    "HD023060": "data/HD023060/swp11151mxlo_vo.fits",
    "HD099872": "data/HD099872/mastDownload/HST/**/*_x1d.fits",
    # "HD152248": "data/HD152248/swp54576.mxhi.gz",
    "HD152248": "data/HD152248/*.mxhi.gz",
    "HD209339": "data/HD209339/mastDownload/HST/o5lh0b010/o5lh0b010_x1d.fits",
    # "HD197770": "data/HD197770/mastDownload/HST/oedl04010/oedl04010_x1d.fits",
    "HD197770": "data/HD197770/swp49267.mxhi.gz",
    "HD037332": "data/HD037332/swp32289.mxhi.gz",
    "HD093028": "data/HD093028/swp05521.mxhi.gz",
    # "HD062542": "data/HD062542/mastDownload/HST/obik01020/obik01020_x1d.fits", # wavelength range
    # "HD062542": "data/HD062542/*.mxhi.gz", # way too noisy
    "HD062542": "data/HD062542/*mxlo_vo.fits",
    "HD190603": "data/HD190603/swp01822.mxhi.gz",
    # "HD046202": "data/HD046202/swp08845.mxhi.gz",
    # "HD046202": "data/HD046202/mastDownload/HST/ocb6e0030/ocb6e0030_x1d.fits",
    # "HD046202": "data/HD046202/mastDownload/HST/ocb6e1030/ocb6e1030_x1d.fits",
    "HD046202": "data/HD046202/mastDownload/HST/**/*_x1d.fits",
    # "HD047129": "data/HD047129/swp07077.mxhi.gz",
    "HD047129": "data/HD047129/*.mxhi.gz",
    "HD235874": "data/HD235874/swp34158mxlo_vo.fits",
    "HD216898": "data/HD216898/swp43934.mxhi.gz",
    "HD326329": "data/HD326329/swp48698.mxhi.gz",
    "HD179406": [
        "data/HD179406/swp08974.mxhi.gz",
        "data/HD179406/swp08976.mxhi.gz",
        "data/HD179406/swp13865.mxhi.gz",
        "data/HD179406/swp36939.mxhi.gz",
        "data/HD179406/swp36940.mxhi.gz",
    ],
    "BD+52d3210": "data/BD+52d3210/swp34153mxlo_vo.fits",
    "BD+56d524": "data/BD+56d524/swp20330mxlo_vo.fits",
}


# namedtuple defines a simple class
Spectrum = collections.namedtuple(
    "Spectrum", ["wavs", "flux", "errs", "net", "exptime"]
)


def processed(target, wmin=0, wmax=1400, disp=0.25):
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
    spectrum, rebin = auto_wavs_flux_errs(filename)
    if rebin:
        binnedwavs, binnedflux = rebin_spectrum_around_lya(spectrum, wmin, wmax, disp)
    else:
        wavs, flux = spectrum.wavs, spectrum.flux
        use = np.logical_and(wmin < wavs, wavs < wmax)
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
        to_be_coadded = filename
    elif isinstance(filename, str):
        if "*" in filename:
            to_be_coadded = [str(p) for p in Path(".").glob(filename)]
        elif "x1d" in filename:
            # a single x1d file can contain multiple extensions, which
            # need to be coadded
            to_be_coadded = [filename]
        else:
            to_be_coadded = None
    else:
        warn("filename should be str or list!")
        raise

    if to_be_coadded is None:
        if "x1d" in filename:
            spectrum = merged_stis_data(filename)
            rebin = True
        elif "mxhi" in filename:
            spectrum = merged_iue_h_data(filename)
            rebin = True
        elif "mxlo" in filename:
            spectrum = iue_l_data(filename)
            rebin = False
        else:
            warn("File {} not supported yet, exiting".format(filename))
            exit()
    else:
        if "x1d" in to_be_coadded[0]:
            spectrum = coadd_hst_stis(to_be_coadded)
            rebin = True
        elif "mxhi" in to_be_coadded[0]:
            spectrum = coadd_iue_h(to_be_coadded)
            rebin = True
        elif "mxlo" in to_be_coadded[0]:
            spectrum = coadd_iue_l(to_be_coadded)
            rebin = False

    return spectrum, rebin


def merged_stis_data(filename, extension=1):
    """Get spectrum data from all STIS spectral orders.

    If only filename is given, use SCI extension.

    Returns
    -------

    wavs: numpy array, all wavelengths, sorted

    flux: all fluxes at these wavelengths

    errs: all errors at these wavelengths

    """
    with fits.open(filename) as f:
        t = f[extension].data
        exptime = get_exptime(f[extension].header)

    output_columns = ["WAVELENGTH", "FLUX", "ERROR", "NET"]
    fields = [np.concatenate(t[c]) for c in output_columns]

    # clean up by dq
    dq = np.concatenate(t["DQ"])
    good = dq == 0
    print(f"STIS: {good.sum()} out of {len(good)} wavelength points are good")
    fields = [c[good] for c in fields]

    # sort by wavelength
    idxs = np.argsort(fields[0])
    fields = [c[idxs] for c in fields]

    # add exptime and create Spectrum (namedtuple) object (* unpacks,
    # should be in right order)
    fields.append(exptime)
    return Spectrum(*fields)


def merged_iue_h_data(filename):
    """
    Get Spectrumn info over all orders of high res IUE data.

    Returns
    -------
    Spectrum
    """
    t = Table.read(filename)

    def iue_wavs(i):
        return t[i]["WAVELENGTH"] + t[i]["DELTAW"] * np.arange(t[i]["NPOINTS"])

    def pixrange(i):
        return slice(t[i]["STARTPIX"], t[i]["STARTPIX"] + t[i]["NPOINTS"])

    def all_of_column(colname):
        return np.concatenate([t[i][colname][pixrange(i)] for i in range(len(t))])

    allwavs = np.concatenate([iue_wavs(i) for i in range(len(t))])

    colnames = ["WAVELENGTH", "ABS_CAL", "NOISE", "NET"]

    column_values = [allwavs]
    for colname in colnames[1:]:
        column_values.append(all_of_column(colname))

    # clean up using DQ
    dq = all_of_column("QUALITY")
    good = dq == 0
    print(f"IUE: {good.sum()} out of {len(good)} wavelength points are good")
    for array in column_values:
        array = array[good]

    # sort by wavelength
    idxs = np.argsort(column_values[0])
    column_values = [c[idxs] for c in column_values]

    # add exptime and create Spectrum
    exptime = get_exptime(fits.getheader(filename, ext=0))

    fields = column_values + [exptime]
    return Spectrum(*fields)


def iue_l_data(filename):
    t = Table.read(filename)
    wavs = t["WAVE"][0]
    flux = t["FLUX"][0]
    sigma = t["SIGMA"][0]
    # net is not available
    net = None
    # exptime is not used (for now)
    exptime = None
    return Spectrum(wavs, flux, sigma, net, exptime)


def coadd_iue_h(filenames):
    print(f"Coadding {len(filenames)} IUE H exposures")
    return coadd_general([merged_iue_h_data(fn) for fn in filenames])


def coadd_iue_l(filenames):
    print(f"Coadding {len(filenames)} IUE L exposures")
    spectrums = [iue_l_data(fn) for fn in filenames]

    if not np.equal.reduce([s.wavs for s in spectrums]).all():
        warn("Not all wavs are equal in IUE L. Implement fix pls.")
        raise

    # Assume that the wavs are always the same. If not, the above error
    # will trigger, and I should reconsider.
    numwavs = len(spectrums[0].wavs)
    flux_sum = np.zeros(numwavs)
    weight_sum = np.zeros(numwavs)
    for s in spectrums:
        good = np.isfinite(s.flux) & (s.errs > 0)
        weight = 1 / s.errs ** 2
        flux_sum[good] += s.flux[good] * weight[good]
        weight_sum[good] += weight[good]

    # simply the 1/sigma2 weighting rule
    new_flux = flux_sum / weight_sum
    new_errs = np.sqrt(1 / weight_sum)
    return Spectrum(spectrums[0].wavs, new_flux, new_errs, None, None)


def coadd_hst_stis(filenames):
    # get all SCI exposures
    spectrums = []

    # remember handles so we can close them later
    for fn in filenames:
        with fits.open(fn) as hdus:
            for extension in range(1, len(hdus)):
                spectrums.append(merged_stis_data(fn, extension))

    print(f"Coadding {len(spectrums)} STIS exposures from {len(filenames)} files")
    return coadd_general(spectrums)


def coadd_general(spectrums):
    """General function for coadding spectra.

    spectrums : list of Spectrum objects

    Returns
    -------
    spectrum : Spectrum object representing the coadded data

    """
    # get all the per-wavelength data
    all_wavs = [s.wavs for s in spectrums]

    # determine new wavelength grid, using max of median of wavelength
    # increment as step size
    maxwav = np.amax(np.concatenate(all_wavs))
    minwav = np.amin(np.concatenate(all_wavs))
    disp = np.amax([np.median(np.diff(w)) for w in all_wavs])
    newwavs = np.arange(minwav, maxwav, disp)

    # instead of binning, we're just going to do nearest neighbour on a
    # slightly coarser wavelength grid. It worked for Julia, so...
    flux_sum = np.zeros(len(newwavs))
    weight_sum = np.zeros(len(newwavs))
    variance_sum = np.zeros(len(newwavs))
    net_sum = np.zeros(len(newwavs))
    total_exptime = np.zeros(len(newwavs))
    for s in spectrums:
        # nearest neighbour interpolation of all relevant quantities
        def do_interp1d(quantity):
            return interp1d(
                s.wavs, quantity, kind="nearest", fill_value=np.nan, bounds_error=False,
            )(newwavs)

        fi = do_interp1d(s.flux)
        ei = do_interp1d(s.errs)
        ni = do_interp1d(s.net)
        exptime = s.exptime

        # weights scale with ni / fi = sensitivity
        good_fi_ni = (fi != 0) & np.isfinite(fi) & (ni != 0) & np.isfinite(ni)
        wi = np.where(good_fi_ni, ni / fi, 0) * exptime
        good_wi = wi > 0

        # total_counts = flux * sensitivity * exptime
        # --> flux = total_counts / (sensitivity * exptime)
        #
        # V(flux) = V(total_counts) / (sensitivity * exptime)**2
        #         = total_counts / (sensitivity * exptime)**2 (poisson)
        #         = flux * sensitivity * exptime / (sensitivity * exptime)**2
        #         = flux / (sensitivity * exptime)

        # sens = counts per flux unit

        weight_sum[good_wi] += wi[good_wi]
        flux_sum[good_wi] += wi[good_wi] * fi[good_wi]
        variance_sum[good_wi] += np.square(ei[good_wi] * wi[good_wi])

        net_sum[good_wi] += ni[good_wi] * exptime
        total_exptime[good_wi] += exptime

    flux_result = flux_sum / weight_sum
    errs_result = np.sqrt(variance_sum) / weight_sum
    net_result = net_sum / total_exptime

    return Spectrum(newwavs, flux_result, errs_result, net_result, total_exptime)


def rebin_spectrum_around_lya(spectrum, wmin=0, wmax=1400, disp=0.25):
    """Rebin spectrum to for lya fitting, and reject certain points.

    A rebinning of the spectrum to make it more useful for lya fitting.
    Every new point is the weighted average of all data within the range
    of a bin. The weights are flux / net * exptime if those are
    available. If not 1 / errs**2 is used. The bins can be specified by
    choosing a minimum, maximum wavelength and a resolution (in
    Angstrom). Additionally, only the points that satisfy some basic
    data rejection criteria are used.

    Returns
    -------
    newwavs: average wavelength in each bin
    newflux: average flux in each bin

    """
    wavs = spectrum.wavs
    flux = spectrum.flux
    wavmin = max(wmin, np.amin(wavs))
    wavmax = min(wmax, np.amax(wavs))
    wavbins = np.arange(wavmin, wavmax, disp)

    if spectrum.net is not None and spectrum.exptime is not None:
        weights = spectrum.net / flux * spectrum.exptime
    else:
        weights = 1 / spectrum.errs ** 2

    # np.digitize returns list of indices. b = 1 means that the data point
    # is between wav[0] (first) and wav[1]. b = n-1 means between wav[n-2]
    # and wav[n-1] (last). b = 0 or n mean out of range.
    bs = np.digitize(wavs, wavbins)
    newwavs = np.zeros(len(wavbins) - 1)
    newflux = np.zeros(len(wavbins) - 1)
    for i in range(0, len(wavbins) - 1):
        in_bin = bs == i + 1  # b runs from 1 to n-1
        use = np.logical_and.reduce(
            [in_bin, np.isfinite(flux), weights > 0, np.isfinite(weights)]
        )
        # if a bin is empty or something else is wrong, the nans will be
        # filtered out later
        if not use.any():
            newwavs[i] = 0
            newflux[i] = np.nan
            continue

        newwavs[i] = np.average(wavs[use], weights=weights[use])
        newflux[i] = np.average(flux[use], weights=weights[use])

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
