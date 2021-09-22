#!/usr/bin/env python

import numpy as np
from astropy.table import Table, join
import pandas
import re
import astropy.units as u


def add_lin_column_from_log(logcolname, data):
    log = data[logcolname]
    log_unc = data[logcolname + "_unc"]
    lin = np.power(10.0, log)
    lin_unc = 0.5 * (np.power(10.0, log + log_unc) - np.power(10.0, log - log_unc))
    linname = logcolname.replace("log", "")
    data.add_column(lin, name=linname)
    data.add_column(lin_unc, name=linname + "_unc")


def get_fuse_h1_h2():
    """
    Read in the FUSE H1 and H2 column data.

    Returns
    -------
    data : astropy.table object
       Table of the data (h1, h2, htot, etc.)
    """

    data = Table.read("data/fuse_h1_h2_with_lyafitting.dat", format="ascii")

    # rename column to have the same name as other tables
    data.rename_column("name", "Name")

    # remove the ebv column as this is superceded by another table
    #    this ebv value is from the literature and is for each star
    #    the correct E(B-V) value is for the calculated extinction curve
    data.remove_column("ebv")

    # recalculate the lognhtot columns
    #  updated lognhi columns and generally making sure basic math is correct
    nhi = np.power(10.0, data["lognhi"])
    nh2 = np.power(10.0, data["lognh2"])
    nhtot = nhi + 2.0 * nh2
    nhi_unc = 0.5 * (
        np.power(10.0, data["lognhi"] + data["lognhi_unc"])
        - np.power(10.0, data["lognhi"] - data["lognhi_unc"])
    )
    nh2_unc = 0.5 * (
        np.power(10.0, data["lognh2"] + data["lognh2_unc"])
        - np.power(10.0, data["lognh2"] - data["lognh2_unc"])
    )
    nhtot_unc = np.sqrt(np.square(nhi_unc) + np.square(2.0 * nh2_unc))

    # save the new total
    data["lognhtot"] = np.log10(nhtot)
    data["lognhtot_unc"] = 0.5 * (
        np.log10(nhtot + nhtot_unc) - np.log10(nhtot - nhtot_unc)
    )
    # save the linear versions
    data["nhi"] = nhi
    data["nhi_unc"] = nhi_unc
    data["nh2"] = nh2
    data["nh2_unc"] = nh2_unc
    data["nhtot"] = nhtot
    data["nhtot_unc"] = nhtot_unc

    # recalculate the f_H2
    data["fh2"] = 2.0 * data["nh2"] / data["nhtot"]
    data["fh2_unc"] = data["fh2"] * np.sqrt(
        np.square(nh2_unc / nh2) + np.square(nhtot_unc / nhtot)
    )

    data["logfh2"] = np.log10(data["fh2"])

    # H volume density
    #   needs updating when we have new distances
    data["nh"] = np.power(10.0, data["lognh"])

    return data


def get_fuse_h2_details(components=False, comparison=False, stars=None):
    """Read in the results of the H2 fits (level populations and errors).

    Some sightlines have multiple fits, likely due to velocity
    components. Therefore the format is a bit different, and this file
    needs to be loaded separately from the rest (not via
    get_merged_table).

    Parameters
    ----------
    components : bool
        load the data with multiple entries per star. False just sums
        everything, so there's only one row per star.

    comparison : bool
        load the data for the comparison stars instead of the main sample

    stars : list of stars for which to retrieve data

    """
    data = Table.from_pandas(
        pandas.read_csv("data/fuse_h2_details.dat", delim_whitespace=True)
    )

    # convert log to linear
    current_columns = [str(c) for c in data.colnames]
    for c in current_columns:
        if "log" in c and "_unc" not in c:
            add_lin_column_from_log(c, data)

    keep = []
    for s in stars:
        if s in data["Name"]:
            # determine indices for star
            idx = np.where(data["Name"] == s)[0][0]
            keep.append(idx)
            extra_index = idx + 1
            while data["Name"][extra_index] == "idem":
                keep.append(extra_index)
                data["Name"][extra_index] = data["Name"][idx]
                extra_index += 1
    result = data[keep]
    result.write("data/fuse_h2_details_main.dat", format="ascii.commented_header")

    # if the separate components are needed, we can return early
    if components:
        return result
    # if not, continue processing

    # also output summed result per star (only the linear columns to
    # keep this easy). Watch out for name column.
    new_colnames = [c for c in result.colnames if "log" not in c and c != "Name"]
    summed_result = Table(
        names=new_colnames,
    )

    for s in stars:
        rows = result[result["Name"] == s]
        vals = []
        for c in new_colnames:
            if "_unc" in c:
                sum_unc = np.sqrt(np.nansum(np.square(rows[c])))
                vals.append(sum_unc)
            else:
                vals.append(np.nansum(rows[c]))

        summed_result.add_row(vals)

    summed_result.add_column(stars, index=0, name="Name")

    # some rotational temperatures
    def gi(J):
        grot = 2 * J + 1
        if J % 2 == 0:
            return grot
        else:
            return 3 * grot

    def trot(lo, hi):
        return 1 / np.log(
            summed_result[f"nj{lo}"] / gi(lo) * gi(hi) / summed_result[f"nj{hi}"]
        )

    def trot_unc(lo, hi):
        nlo = summed_result[f"nj{lo}"]
        nhi = summed_result[f"nj{hi}"]
        nlo_unc = summed_result[f"nj{lo}_unc"]
        nhi_unc = summed_result[f"nj{hi}_unc"]
        log2_factor = np.square(np.log(nlo / nhi * gi(hi) / gi(lo)))
        return np.sqrt(
            (nlo_unc / nlo / log2_factor) ** 2 + (nhi_unc / nhi / log2_factor) ** 2
        )

    dE01 = 170.48
    t01 = dE01 * trot(0, 1)
    t01_unc = dE01 * trot_unc(0, 1)

    summed_result.add_column(t01, index=0, name="T01")
    summed_result.add_column(t01_unc, index=0, name="T01_unc")

    return summed_result


def get_fuse_ext_details(filename):
    """
    Read in the FUSE extinction details [A(V), R(V), etc.]

    Parameters
    ----------
    filename: str
       name of file with the data

    Returns
    -------
    data : astropy.table object
       Table of the data [A(V), R(V), etc.]
    """

    data = Table.read(filename, format="ascii.commented_header", header_start=-1)

    # create the combined uncertainties
    keys = ["AV", "EBV", "RV"]
    for key in keys:
        if (key in data.colnames) and (not key + "_unc" in data.colnames):
            data[key + "_unc"] = np.sqrt(
                np.square(data[key + "_runc"]) + np.square(data[key + "_sunc"])
            )

    # make EBV column if it does not exist
    if "EBV" not in data.colnames:
        data["EBV"] = data["AV"] / data["RV"]
        data["EBV_unc"] = data["EBV"] * np.sqrt(
            np.square(data["AV_unc"] / data["AV"])
            + np.square(data["RV_unc"] / data["RV"])
        )

    return data


def get_fuse_ext_fm90():
    """
    Read in the FUSE extinction details [A(V), R(V), etc.]

    Returns
    -------
    data : astropy.table object
       Table of the data [A(V), R(V), etc.]
    """

    data = Table.read(
        "data/fuse_ext_fm90.dat", format="ascii.commented_header", header_start=-1
    )

    # create the combined uncertainties
    keys = ["CAV1", "CAV2", "CAV3", "CAV4", "x_o", "gamma"]
    for key in keys:
        data[key + "_unc"] = np.sqrt(
            np.square(data[key + "_runc"]) + np.square(data[key + "_sunc"])
        )

    return data


def get_bohlin78():
    """
    Read in the Bohlin et al. (1978) Copernicus data

    Returns
    -------
    data : astropy.table object
       Table of the data [EBV, etc.]
    """

    data = Table.read(
        "data/bohlin78_copernicus.dat", format="ascii.commented_header", header_start=-1
    )

    # remove sightlines with non-physical EBV values
    (indxs,) = np.where(data["EBV"] > 0.0)
    data = data[indxs]

    # get the units correct
    data["nhi"] = 1e20 * data["nhi"]
    data["nhtot"] = 1e20 * data["nhtot"]

    # convert the uncertainties from % to total
    data["nhi_unc"] = data["nhi"] * data["nhi_unc"] * 0.01

    # make log units for consitency with FUSE work
    data["lognhi"] = np.log10(data["nhi"])
    data["lognhi_unc"] = 0.5 * (
        np.log10(data["nhi"] + data["nhi_unc"])
        - np.log10(data["nhi"] - data["nhi_unc"])
    )

    data["lognhtot"] = np.log10(data["nhtot"])

    # create the fh2 column
    data["fh2"] = 2.0 * data["nh2"] / data["nhtot"]

    # make a AV column assuming RV=3.1
    data["RV"] = np.full((len(data)), 3.1)
    data["AV"] = data["RV"] * data["EBV"]

    # now the NH/AV and NH/EBV
    data["NH_AV"] = data["nhtot"] / data["AV"]
    data["NH_EBV"] = data["nhtot"] / data["EBV"]

    return data


def get_shull2021(drop_duplicates=False):
    """
    Read in the Shull et al. (2021) FUSE-based data.

    Parameters
    ----------
    drop_duplicates : bool
        Remove stars that are already in our sample
    """
    # read in tables 1 to 4 (saved in CDS ascii format from the
    # publisher website)
    tables = [
        Table.read(f"data/shull-2021/table{n}.txt", format="ascii.cds")
        for n in range(1, 5)
    ]

    shull_colnames = ["Name", "E(B-V)", "logNHI", "logNH2", "logNH", "fH2"]
    our_colnames = ["Name", "EBV", "lognhi", "lognh2", "lognhtot", "fh2"]

    # start with empty table
    data = Table()

    for colname, shull_colname in zip(our_colnames, shull_colnames):
        # look for the desired column in one of the 4 tables (we don't
        # use table 3 right now, but later we should watch out 3, as
        # there can be multiple rows per star in that one)
        for t in tables:
            if shull_colname in t.colnames:
                data.add_column(t[shull_colname], name=colname)
                break

    # take exponential if necessary
    for colname in data.colnames:
        if "log" in colname:
            data.add_column(
                np.power(10, data[colname]), name=colname.replace("log", "")
            )

    # add derived columns
    data.add_column(data["nhtot"] / data["EBV"], name="NH_EBV")

    data["Name"] = [edit_shull_name(s) for s in data["Name"]]
    return data


def edit_shull_name(s):
    if "HD" in s:
        number = s.split(" ")[-1]
        newname = "HD" + f"{number:0>6}"
    elif "BD" in s:
        newname = s.replace(" ", "").replace("deg", "d")
    else:
        newname = s
    return newname


def add_distance(table, comp=False):
    """Read or calculate distances from various sources.

    Parallaxes from file created by get_gaia script.

    Also, distances from Table 1 of Shull+21 if available.

    """
    ### GAIA

    fn = "data/gaia/merged{}.dat".format("_comp" if comp else "")
    gaia = Table.read(fn, format="ascii.commented_header")
    # gaia parallaxes often need offset. Most commonly used value is 0.03 e.g. in Shull & Danforth 2019
    offset = 0.03
    plx = gaia["parallax"] + offset
    plx_unc = gaia["parallax_error"]

    def p_to_d(p):
        # return p.to(u.parsec, equivalencies=u.parallax())
        return 1 / p  # p is in milli arcseconds, so this is kpc

    d = p_to_d(plx)
    dplus = p_to_d(plx - plx_unc)
    dmin = p_to_d(plx + plx_unc)
    d_unc = 0.5 * (dplus - dmin)

    # make containing names and distance, to join with main table and to
    # make the names match
    gaia_dist = Table([gaia["Name"], d, d_unc], names=["Name", "d_gaia", "d_gaia_unc"])
    table_edit = join(table, gaia_dist, keys="Name")
    table_edit["d"] = table_edit["d_gaia"]
    table_edit["d_unc"] = table_edit["d_gaia_unc"]

    ### photometric

    # distance in parsec according to magnitude equation
    v = table_edit["V"]
    av = table_edit["AV"]
    mv = get_abs_magnitudes(table_edit["SpType"])
    # apply magnitude distance equation (divide by 1000 to get kpc)
    d = np.where(np.isnan(mv), np.nan, 10 * np.power(10, (v - av - mv) / 5) / 1000)
    # hack for now. Should have real uncertainties here when doing
    # something serious with distance
    d_unc = 0.1 * d
    table_edit.add_column(d, name="dphot")

    # replace the main distance measurement where possible
    replace = np.isfinite(table_edit["dphot"])
    table_edit["d"][replace] = d[replace]
    table_edit["d_unc"][replace] = d_unc[replace]

    ### Shull+21 data. If available, overwrite our value.
    count = 0
    sh = Table.read("data/shull-2021/table1.txt", format="ascii.cds")
    our_names = table_edit["Name"]
    for dphot, name in zip(sh["Dphot"], sh["Name"]):
        our_format = edit_shull_name(name)
        if our_format in our_names:
            our_index = np.where(our_format == table_edit["Name"])[0][0]
            table_edit["d"][our_index] = dphot
            table_edit["d_unc"][our_index] = dphot * 0.1
            count += 1
    print(f"Took {count} distances from Shull+21")

    return table_edit


def get_abs_magnitudes(sptype):
    """
    Use given spectral type column and ob_mags file to find absolute magnitude.

    Uses table from appendix 3B of Bowen et al. 2008.

    Parameters
    ----------
    sptype : spectral type column from main table

    Returns
    -------
    MV : absolute magnitude for each star

    """
    # this works nicely
    t = Table.read("data/ob_mags.dat", format="ascii.commented_header")
    mv = np.zeros(len(sptype))
    for i, s in enumerate(sptype):
        # spectral type (e.g. B0.5)
        match = re.match("[OB][0-9](\.[0-9])?", s)
        # print(match[0])
        # luminosity class (e.g. IV)
        col = s[match.end() :].lower()
        # retrieve from table
        if match[0] in t["type"] and col in t.colnames:
            index = np.where(t["type"] == match[0])[0][0]
            mv[i] = t[col][index]
        else:
            mv[i] = np.nan
            print(f"Did not find absolute magnitude for {s}")

    return mv


def get_merged_table(comp=False):
    """
    Read in the different files and merge them

    Parameters
    ----------
    comp : boolean, optional
       get the comparison data
    """

    # get the three tables to merge
    h1h2_data = get_fuse_h1_h2()
    if comp:
        filename = "data/fuse_comp_details_fm90.dat"
    else:
        filename = "data/fuse_ext_details.dat"
    ext_detail_data = get_fuse_ext_details(filename)

    # merge the tables together
    merged_table = join(h1h2_data, ext_detail_data, keys="Name")

    # add calculated photometric distances or gaia distances
    merged_table = add_distance(merged_table, comp)

    def add_den_column(colname, customname=None):
        # add 3d densities
        if customname is None:
            newname = colname.replace("nh", "denh")
        else:
            newname = customname

        uncname = newname + "_unc"
        merged_table[newname] = merged_table[colname] / merged_table["d"]
        frac = np.sqrt(
            (merged_table[colname + "_unc"] / merged_table[colname]) ** 2
            + (merged_table["d_unc"] / merged_table["d"]) ** 2
        )
        merged_table[uncname] = merged_table[newname] * frac
        # convert to cm-3
        merged_table[newname] = (
            (merged_table[newname] * u.kpc ** -1 * u.cm ** -2).to(u.cm ** -3).value
        )
        merged_table[uncname] = (
            (merged_table[uncname] * u.kpc ** -1 * u.cm ** -2).to(u.cm ** -3).value
        )

    add_den_column("nhtot")
    add_den_column("nh2")
    add_den_column("nhi")
    add_den_column("AV", "AV_d")

    # add 1/RV and uncertainty
    merged_table["1_RV"] = 1 / merged_table["RV"]
    merged_table["1_RV_unc"] = merged_table["RV_unc"] / merged_table["RV"] ** 2

    if not comp:
        h2details = get_fuse_h2_details(stars=merged_table["Name"])
        merged_table = join(merged_table, h2details, keys="Name")

        ext_fm90_data = get_fuse_ext_fm90()
        merged_table1 = join(merged_table, ext_fm90_data, keys="Name")
        merged_table = merged_table1

        # make the "regular" FM90 parameters
        #  normalized by E(B-V) instead of A(V)
        merged_table["C1"] = (merged_table["CAV1"] - 1.0) * merged_table["RV"]
        merged_table["C1_unc"] = merged_table["C1"] * np.sqrt(
            np.square(merged_table["CAV1_unc"] / merged_table["CAV1"])
            + np.square(merged_table["EBV_unc"] / merged_table["EBV"])
        )
        merged_table["C2"] = merged_table["CAV2"] * merged_table["RV"]
        merged_table["C2_unc"] = merged_table["C2"] * np.sqrt(
            np.square(merged_table["CAV2_unc"] / merged_table["CAV2"])
            + np.square(merged_table["EBV_unc"] / merged_table["EBV"])
        )
        merged_table["C3"] = merged_table["CAV3"] * merged_table["RV"]
        merged_table["C3_unc"] = merged_table["C3"] * np.sqrt(
            np.square(merged_table["CAV3_unc"] / merged_table["CAV3"])
            + np.square(merged_table["EBV_unc"] / merged_table["EBV"])
        )
        merged_table["C4"] = merged_table["CAV4"] * merged_table["RV"]
        merged_table["C4_unc"] = merged_table["C4"] * np.sqrt(
            np.square(merged_table["CAV4_unc"] / merged_table["CAV4"])
            + np.square(merged_table["EBV_unc"] / merged_table["EBV"])
        )

    # generate the N(H)/A(V) columns
    merged_table["NH_AV"] = merged_table["nhtot"] / merged_table["AV"]
    merged_table["NH_AV_unc"] = merged_table["NH_AV"] * np.sqrt(
        np.square(merged_table["nhtot_unc"] / merged_table["nhtot"])
        + np.square(merged_table["AV_unc"] / merged_table["AV"])
    )

    # generate the N(H)/E(B-V) columns
    merged_table["NH_EBV"] = merged_table["nhtot"] / merged_table["EBV"]
    merged_table["NH_EBV_unc"] = merged_table["NH_EBV"] * np.sqrt(
        np.square(merged_table["nhtot_unc"] / merged_table["nhtot"])
        + np.square(merged_table["EBV_unc"] / merged_table["EBV"])
    )

    # make the 2175 A bump area
    if ("CAV3" in merged_table.colnames) and ("gamma" in merged_table.colnames):
        # C3 = (merged_table['CAV3'] - 1.0)*merged_table['RV']
        # indxs = np.where(C3 == 0.0)
        # print(C3[indxs], merged_table['CAV3'][indxs])
        # C3_unc = C3*np.sqrt(np.square(merged_table['CAV3_unc']
        #                              / merged_table['CAV3'])
        #                    + np.square(merged_table['RV_unc']
        #                                / merged_table['RV']))
        C3 = merged_table["CAV3"]
        C3_unc = merged_table["CAV3_unc"]
        gamma = merged_table["gamma"]
        gamma_unc = merged_table["gamma_unc"]
        merged_table["bump_area"] = np.pi * C3 / (2.0 * merged_table["gamma"])
        bump_area_unc = np.sqrt(
            np.square(C3_unc / C3)
            + np.square(merged_table["gamma_unc"] / merged_table["gamma"])
        )
        merged_table["bump_area_unc"] = merged_table["bump_area"] * bump_area_unc
        merged_table["bump_amp"] = C3 / gamma ** 2
        bump_amp_unc = np.sqrt(
            (C3_unc / gamma ** 2) ** 2 + (gamma_unc * C3 * -2 / gamma ** 3) ** 2
        )
        merged_table["bump_amp_unc"] = bump_amp_unc

    # add A1000 (within H2 dissociation cross section) and A1300
    # (outside that range)
    params_needed = [f"CAV{n}" for n in range(1, 5)] + ["gamma", "x_o"]
    if all([p in merged_table.colnames for p in params_needed]):
        cav1, cav2, cav3, cav4, gamma, x0 = [merged_table[p] for p in params_needed]
        cav1_unc, cav2_unc, cav3_unc, cav4_unc, gamma_unc, x0_unc = [
            merged_table[p + "_unc"] for p in params_needed
        ]

        def extinction_curve(w):
            """w in angstrom"""
            micron = 10000
            # x is 1/w in micron
            x = 1 / (w / micron)
            # drude
            D = x ** 2 / ((x ** 2 - x0 ** 2) ** 2 + (x * gamma) ** 2)
            F = 0.5392 * (x - 5.9) ** 2 + 0.05644 * (x - 5.9) ** 3
            Aw_AV = cav1 + cav2 * x + cav3 * D + cav4 * F
            return Aw_AV

        def extinction_curve_unc(w):
            x = 1 / w
            # drude
            D = x ** 2 / ((x ** 2 - x0 ** 2) ** 2 + (x * gamma) ** 2)
            F = 0.5392 * (x - 5.9) ** 2 + 0.05644 * (x - 5.9) ** 3

            # dD / dg = -2gx^4 / (g^2 x^2 + (x0^2 + x^2)^2)^2
            # dD / dx0 = -4mx^2(m^2+x^2)/(g^2x^2+(m^2+x^2)^2)^2
            D_denom = (x ** 2 - x0 ** 2) ** 2 + (x * gamma) ** 2
            # multivariate error propagation
            D_unc_gamma = 2 * gamma * x ** 4 / D_denom ** 2 * gamma_unc
            D_unc_x0 = 4 * x0 * x ** 2 * (x0 ** 2 + x ** 2) / D_denom ** 2 * x0_unc
            VD_rel = (D_unc_gamma ** 2 + D_unc_x0 ** 2) / D ** 2
            # sum of relative variances
            Vcav3_rel = (cav3_unc / cav3) ** 2
            Vterm3 = (cav3 * D) ** 2 * (Vcav3_rel + VD_rel)

            VA = cav1_unc ** 2 + (cav2_unc * x) ** 2 + Vterm3 + (cav4_unc * F) ** 2
            return np.sqrt(VA)

        def add_specific_wavelength(w):
            val = f"A{w}_AV"
            unc = val + "_unc"
            merged_table[val] = extinction_curve(w)
            merged_table[unc] = extinction_curve_unc(w)

            # also multiply them by AV
            absval = val.replace("_AV", "")
            merged_table[absval] = merged_table[val] * merged_table["AV"]
            rel_unc2 = (merged_table[unc] / merged_table[val]) ** 2 + (
                merged_table["AV_unc"] / merged_table["AV"]
            ) ** 2
            merged_table[absval + "_unc"] = merged_table[absval] * np.sqrt(rel_unc2)

        add_specific_wavelength(1000)
        add_den_column("A1000", "A1000_d")
        add_specific_wavelength(2175)

    return merged_table


if __name__ == "__main__":

    merged_table = get_merged_table()
    print(merged_table.colnames)
    print(merged_table)
