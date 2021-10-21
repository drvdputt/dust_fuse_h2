"""Tools for calculating photometric distance.

Uses data from data/ob_mags.dat.
"""

import numpy as np
from astropy.table import Table
import re


def get_abs_magnitudes(sptype):
    """
    Use given spectral type column and ob_mags file to find absolute magnitude.

    Uses table from appendix 3B of Bowen et al. 2008.

    Parameters
    ----------
    sptype : spectral type column from main table

    Returns
    -------
    mv : array
        absolute magnitude for each star

    mv_unc_hi, mv_unc_lo : array
        upper and lower uncertainty on absolute magnitude, using Bowen et al. 2008 eq.
        B10 (quadrature of difference to next spectral type + difference
        to next luminosity class + flat uncertainty of .25)

    """
    mv_table = Table.read("data/ob_mags.dat", format="ascii.commented_header")

    # value, upper
    mv = np.zeros(len(sptype))
    mv_unc = np.zeros(len(sptype))

    for i, s in enumerate(sptype):
        # spectral type (e.g. B0.5)
        match = re.match("[OB][0-9](\.[0-9])?", s)
        # print(match[0])
        # luminosity class (e.g. IV)
        spt = match[0]
        lum = s[match.end() :].lower()

        # retrieve from table
        if spt in mv_table["type"] and lum in mv_table.colnames:
            spt_index = np.where(mv_table["type"] == spt)[0][0]
            mv[i] = mv_table[lum][spt_index]

            # sptype error
            next_spt_index = spt_index + 1
            prev_spt_index = spt_index - 1

            # (look at table: next index = higher (less negative) magnitude)
            if next_spt_index < len(mv_table):
                mv_spt_next = mv_table[lum][next_spt_index]
            else:
                raise ValueError("spT+1 not available (for calculating mv error)")

            if prev_spt_index > 0:
                mv_spt_prev = mv_table[lum][prev_spt_index]
            else:
                raise ValueError("spT-1 not available (for calculating mv error)")

            # the table is not monotonic in the lum direction, so we use
            # the following approach to give a reasonable range.
            # Technically only necessary for lum+-1, but we do it here
            # too for consistency
            mv_vals_when_change_spt = (mv[i], mv_spt_next, mv_spt_prev)
            mv_unc_spt = 0.5 * (
                max(mv_vals_when_change_spt) - min(mv_vals_when_change_spt)
            )

            # lum error
            lum_index = mv_table.colnames.index(lum)
            next_lum_index = lum_index + 1
            prev_lum_index = lum_index - 1

            # (look at table: next index = more negative)
            if next_lum_index < len(mv_table.colnames):
                mv_lum_next = mv_table[lum][next_lum_index]
            else:
                # raise ValueError("lum+1 not available (for calculating mv error)")
                # this happens, here's a workaround
                mv_lum_prev = mv[i]

            if prev_lum_index > 0:
                mv_lum_prev = mv_table[lum][prev_lum_index]
            else:
                # raise ValueError("lum-1 not available (for calculating mv error)")
                mv_lum_prev = mv[i]

            mv_vals_when_change_lum = (mv[i], mv_lum_next, mv_lum_prev)
            mv_unc_lum = 0.5 * (
                max(mv_vals_when_change_lum) - min(mv_vals_when_change_lum)
            )

            mv_unc_flat = 0.25
            mv_unc[i] = np.sqrt(mv_unc_spt ** 2 + mv_unc_lum ** 2 + mv_unc_flat ** 2)
        else:
            mv[i] = np.nan
            mv_unc[i] = np.nan
            print(f"Did not find absolute magnitude for {s}")

    # print(sptype, mv, mv_unc)
    return mv, mv_unc


def main_equation(mv, mv_unc, v, v_unc, av, av_unc):
    """
    Apply magnitude distance equation (divide by 1000 to get kpc).
    """

    def d_smart(mags):
        return np.where(np.isnan(mv), np.nan, 10 * np.power(10, mags / 5) / 1000)

    mag_diff = v - av - mv
    mag_diff_unc = np.sqrt(mv_unc ** 2 + v_unc ** 2 + av_unc ** 2)
    d = d_smart(mag_diff)
    d_hi = d_smart(mag_diff + mag_diff_unc)
    d_lo = d_smart(mag_diff - mag_diff_unc)
    d_unc = 0.5 * (d_hi - d_lo)
    return d, d_unc


def calc_distance(sptype, v, v_unc, av, av_unc):
    """
    Calculate photometric distance for given data (from main table).

    Parameters
    ----------
    sptype : SpType column

    v : V column

    av : AV column

    Returns
    -------

    dphot : array
        photometric distance (kpc)

    dphot_unc : array
        uncertainty on photometric distance

    """
    # get absolute magnitude based on spectral type
    mv, mv_unc = get_abs_magnitudes(sptype)
    d, d_unc = main_equation(mv, mv_unc, v, v_unc, av, av_unc)
    return d, d_unc
