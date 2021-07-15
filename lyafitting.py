import get_spectrum
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np


def prepare_axes():
    plt.xlabel("wavelength")
    plt.ylabel("flux (erg cm$^{-2}$s$^{-1}\\AA^{-1}$)")


def wavs_in_ranges(ranges):
    """Determine if wavelength is in one of the ranges given.

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


def estimate_continuum(wavs, flux):
    """Estimate the continuum using a linear fit"""

    # use only these points for continuum estimation
    wav_ranges = [[1262, 1275], [1165, 1170], [1179, 1181], [1185, 1188]]
    use = wavs_in_ranges(wav_ranges)

    linregress_result = linregress(wavs[use], flux[use])
    return linregress_result.slope, linregress_result.intercept


wavs, flux = get_spectrum.processed()
slope, intercept = estimate_continuum(wavs, flux)

plt.plot(wavs, slope * wavs + intercept)
plt.plot(wavs, flux)
prepare_axes()
plt.show()
