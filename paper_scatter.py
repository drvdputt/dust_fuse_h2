"""A limited selection of scatter plots to show in the paper.

These might have specific additions or omissions per plot, depending on
what's interesting to show. E.g., only fit a line when relevant.

Some of them are grouped using subplots

"""

from get_data import get_merged_table, get_bohlin78, get_shull2021
from plot_fuse_results import (
    plot_results_scatter,
    plot_results_fit,
    match_comments,
    plot_rho_box,
)
from matplotlib import pyplot as plt
from astropy.table import Column
import numpy as np
from uncertainties import ufloat
import math
import paper_rcparams
import argparse
from misc import delete_from_arrays

OUTPUT_TYPE = "pdf"
MARK4 = True  # switch to enable marking of low NH/AV points
if MARK4:
    MARK_STRING = ["lo_h_av"]
else:
    MARK_STRING = None

# change colors like this
# plot_fuse_results.MAIN_COLOR = "m"


def set_comment(name, s):
    """Set the comment for a specific star to the string s."""
    data["comment"][data["Name"] == name] = s


# main data and comments to help marking some points
data = get_merged_table()
comp = get_merged_table(True)
data.add_column(Column(["none"] * len(data), dtype="<U16", name="comment"))

set_comment("HD096675", "hi_h_av")

# the 4 low outliers
for name in ["HD045314", "HD164906", "HD200775", "HD206773"]:
    set_comment(name, "lo_h_av")

# Same but with two automatically detected low outliers (they are not
# near the same group in the 1/RV plot, so we probably don't want to
# confuse the reader more by indicating these ones too
# for name in ["HD045314", "HD164906", "HD188001", "HD198781", "HD200775", "HD206773"]:
# set_comment(name, "lo_h_av")


bohlin = get_bohlin78()
shull = get_shull2021()


def finalize_double_grid(fig, axs, filename):
    # turn off xlabel for everything but last row
    if axs.shape[0] > 1:
        pass
    #     for ax in axs[:-1].flatten():
    #         ax.set_xlabel("")
    # # turn off ylabel for everything but last column
    if axs.shape[1] > 1:
        for ax in axs[:, 1:].flatten():
            ax.set_ylabel("")
    save(fig, filename)


def save(fig, filename, need_hspace=False, need_wspace=False):
    if not need_hspace:
        fig.subplots_adjust(hspace=0.02)
    if not need_wspace:
        fig.subplots_adjust(wspace=0.02)
    fig.subplots_adjust(right=0.99)
    dpi = None if OUTPUT_TYPE == "pdf" else 300
    fig.savefig(f"paper-plots/{filename}.{OUTPUT_TYPE}", bbox_inches="tight", dpi=dpi)


# Simply a list of lines for the table. It's a global variable, and stuff is added to it when the different plot functions below are executed.
fit_results_table = []


def format_number_with_unc(x, x_unc):
    """This should go in my general toolbox later.

    Uses 'uncertainties' module, but changes the printing to be more
    latexy."""
    # heuristic for the number of figures after the decimal point
    numdecimal = max(1, int(math.floor(abs(math.log10(abs(x_unc / x)))))) + 1

    u = ufloat(x, x_unc)
    fstring = "{u:." + str(numdecimal) + "e}"

    # typical output of ufloat formatter is '(2.21+/-0.02)e+19'
    print_edit = fstring.format(u=u)
    # replace parentheses and +/- by \num command from siunitx latex package
    print_edit = print_edit.replace("(", "\\num{")
    print_edit = print_edit.replace("+/-", " \\pm ")
    print_edit = print_edit.replace(")", " ")
    print_edit += "}"
    return print_edit


def latex_table_line(xparam, yparam, fit_results_dict):
    """
    Format a fit result as an entry for the fit results table in the paper.

    Parameters
    ----------
    xparam: string
        name of parameter on x-axis

    yparam: string
        name of parameter on y-axis

    fit_results_dict: dict
        output of plot_results_fit

    Returns
    -------
    string:
        xparam & yparam & m \pm m_unc & b \pm b_unc
    """
    power = 1e20
    m, m_unc = fit_results_dict["m"] / power, fit_results_dict["m_unc"] / power
    b, b_unc = fit_results_dict["b"] / power, fit_results_dict["b_unc"] / power
    # m_and_unc_str = format_number_with_unc(m, m_unc)
    # the above doesn't look so good in a table, too many ()'s and 'x e21's
    # do something custom here

    def choose_numdecimal(unc):
        """Correct number of decimals for a plain floating point number, not
        power of 10 notation.

        Matches the order of the uncertainty, with two significant
        digits. Only use when all numbers have a power of divided out
        already.

        """
        order = int(math.floor(math.log10(unc)))
        if order > 0:  # just print number without floating point
            f = 0
        else:  # order 0 --> x.x; order -1 --> 0.xx; order -2 --> 0.0xx
            f = -order + 1
        return f

    def format_val_and_unc(val, unc):
        n = choose_numdecimal(unc)
        fstring = "${val:." + str(n) + "f} \\pm {unc:." + str(n) + "f}$"
        return fstring.format(val=val, unc=unc)

    m_and_unc_str = format_val_and_unc(m, m_unc)
    b_and_unc_str = format_val_and_unc(b, b_unc)
    return f"{xparam} & {yparam} & {m_and_unc_str} & {b_and_unc_str}\\\\"


def plot1_column_column():
    """The first plot shows gas columns vs dust columns.

    Main things to show:
    - outliers in AV relation
    - not outliers in EBV
    - A1000 is very correlated with NH2, and not with AV
    - NHI and NHTOT are correlated with AV, but less with A1000
    - Also show NHtot

    """
    fig, axs = plt.subplots(3, 3, sharey="row", sharex="col")
    fig.set_size_inches(paper_rcparams.base_width, paper_rcparams.base_width)

    # use these variables, so we can easily swap column and rows around
    col = {"AV": 0, "EBV": 1, "A1000": 2}
    row = {"nhtot": 0, "nhi": 1, "nh2": 2}

    def choose_ax(x, y):
        return axs[row[y], col[x]]

    ax = choose_ax("AV", "nhtot")
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "AV",
        "nhtot",
        # data_comp=comp,
        data_bohlin=bohlin,
        # ignore_comments=["lo_h_av", "hi_h_av"],
    )
    out = np.where(match_comments(data, ["lo_h_av", "hi_h_av"]))[0]
    r = plot_results_fit(xs, ys, covs, ax, outliers=out, auto_outliers=True)
    fit_results_table.append(latex_table_line("\\av", "\\nh", r))
    out = r["outlier_idxs"]
    plot_rho_box(
        ax,
        *delete_from_arrays((xs, ys, covs), out),
        method="nocov",
    )

    # print("AV vs nhtot outliers: ", data['name'][

    ax = choose_ax("AV", "nhi")
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "AV",
        "nhi",
        # data_comp=comp,
        data_bohlin=bohlin,
        mark_comments=MARK_STRING,
    )
    plot_rho_box(ax, xs, ys, covs)

    ax = choose_ax("AV", "nh2")
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "AV",
        "nh2",
        # data_comp=comp,
        data_bohlin=bohlin,
        mark_comments=MARK_STRING,
    )
    plot_rho_box(ax, xs, ys, covs)

    ax = choose_ax("EBV", "nhtot")
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "EBV",
        "nhtot",
        # data_comp=comp,
        data_bohlin=bohlin,
        mark_comments=MARK_STRING,
        # ignore_comments=["hi_h_av"],
    )
    r = plot_results_fit(xs, ys, covs, ax, auto_outliers=True)
    fit_results_table.append(latex_table_line("\\ebv", "\\nh", r))
    out = r["outlier_idxs"]
    plot_rho_box(
        ax,
        *delete_from_arrays((xs, ys, covs), out),
        method="nocov",
    )

    ax = choose_ax("EBV", "nhi")
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "EBV",
        "nhi",
        # data_comp=comp,
        data_bohlin=bohlin,
        mark_comments=MARK_STRING,
    )
    plot_rho_box(ax, xs, ys, covs)

    ax = choose_ax("EBV", "nh2")
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "EBV",
        "nh2",
        # data_comp=comp,
        data_bohlin=bohlin,
        mark_comments=MARK_STRING,
    )
    plot_rho_box(ax, xs, ys, covs)

    ax = choose_ax("A1000", "nhtot")
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "A1000",
        "nhtot",
        data_bohlin=bohlin,
        mark_comments=MARK_STRING,
    )
    plot_rho_box(ax, xs, ys, covs)

    ax = choose_ax("A1000", "nhi")
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "A1000",
        "nhi",
        data_bohlin=bohlin,
        mark_comments=MARK_STRING,
    )
    plot_rho_box(ax, xs, ys, covs)

    ax = choose_ax("A1000", "nh2")
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "A1000",
        "nh2",
        data_bohlin=bohlin,
        mark_comments=MARK_STRING,
    )
    r = plot_results_fit(
        xs,
        ys,
        covs,
        ax,
        auto_outliers=True,
        fit_includes_outliers=True,
    )
    fit_results_table.append(latex_table_line("\\ak", "\\nhtwo", r))
    plot_rho_box(ax, xs, ys, covs)

    for ax in axs[1:, 0]:
        ax.yaxis.offsetText.set_visible(False)

    axs[0][0].legend(bbox_to_anchor=(1.5, 1), loc="lower center", ncol=4)

    fig.tight_layout()
    finalize_double_grid(fig, axs, "column_vs_column")


def plot1_poster():
    """Poster version of column vs column plot."""
    fig, axs = plt.subplots(3, 2, sharey="row", sharex="col")
    fig.set_size_inches(paper_rcparams.base_width * 2 / 3, paper_rcparams.base_width)

    ax = axs[0, 0]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "AV",
        "nhtot",
        # data_comp=comp,
        data_bohlin=bohlin,
        # ignore_comments=["lo_h_av", "hi_h_av"],
    )
    out = np.where(match_comments(data, ["lo_h_av", "hi_h_av"]))[0]
    r = plot_results_fit(xs, ys, covs, ax, outliers=out, auto_outliers=True)
    fit_results_table.append(latex_table_line("\\av", "\\nh", r))
    # out = r["outlier_idxs"]
    # plot_rho_box(
    #     ax,
    #     np.delete(xs, out),
    #     np.delete(ys, out),
    #     np.delete(covs, out, 0),
    #     method="nocov",
    # )

    # print("AV vs nhtot outliers: ", data['name'][

    ax = axs[1, 0]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "AV",
        "nhi",
        # data_comp=comp,
        data_bohlin=bohlin,
        mark_comments=MARK_STRING,
    )
    # plot_rho_box(ax, xs, ys, covs)

    ax = axs[2, 0]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "AV",
        "nh2",
        # data_comp=comp,
        data_bohlin=bohlin,
        mark_comments=MARK_STRING,
    )
    # plot_rho_box(ax, xs, ys, covs)

    ax = axs[0, 1]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "A1000",
        "nhtot",
        data_bohlin=bohlin,
        mark_comments=MARK_STRING,
    )
    # plot_rho_box(ax, xs, ys, covs)

    ax = axs[1, 1]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "A1000",
        "nhi",
        data_bohlin=bohlin,
        mark_comments=MARK_STRING,
    )
    # plot_rho_box(ax, xs, ys, covs)

    ax = axs[2, 1]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "A1000",
        "nh2",
        data_bohlin=bohlin,
        mark_comments=MARK_STRING,
    )
    r = plot_results_fit(
        xs,
        ys,
        covs,
        ax,
        # auto_outliers=True,
        fit_includes_outliers=True,
    )
    fit_results_table.append(latex_table_line("\\ak", "\\nhtwo", r))
    # plot_rho_box(ax, xs, ys, covs)

    for ax in axs[1:, 0]:
        ax.yaxis.offsetText.set_visible(False)

    axs[0][0].legend(bbox_to_anchor=(1, 1), loc="lower center", ncol=4)

    fig.tight_layout()
    finalize_double_grid(fig, axs, "column_vs_column_poster")


def plot2_ratio_ratio(mark4: bool = True, no_fh2: bool = False, ignore4: bool = False):
    """Ratio vs ratio.

    x: RV and maybe A1000/AV (extinction ratios)
    y: NH/AV and fh2 (column ratios)

    Parameters
    ----------
    no_fh2: bool
        Do not do the bottom 4 plots (for talk)

    mark4: bool
        show the 4 points with low NH/AV on all the plots

    ignore4: bool
        ignore the 4 special points in the calculation of the correlation coefficient
    """
    if no_fh2:
        nrows = 1
        height = paper_rcparams.base_width * 1 / 3
    else:
        nrows = 2
        height = paper_rcparams.base_width * 2 / 3

    fig, axs = plt.subplots(nrows, 3, sharex="col", sharey="row", squeeze=False)
    fig.set_size_inches(paper_rcparams.base_width, height)

    max_nh_av = 5e21

    ax = axs[0, 0]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "1_RV",
        "NH_AV",
        pxrange=[0.1, 0.5],
        pyrange=[0, max_nh_av],
        # data_comp=comp,
        ignore_comments=["hi_h_av"],
        mark_comments=MARK_STRING,
    )
    r = plot_results_fit(xs, ys, covs, ax, auto_outliers=False)
    # add a rho box using the method specialized for rho with covariance
    # out = r["outlier_idxs"]

    plot_rho_box(
        ax,
        xs,
        ys,
        covs,
        method="cov approx",
        optional_plot_fname="rv_nhav_rho.pdf",
    )
    fit_results_table.append(latex_table_line("\\rvi", "\\nhav", r))
    print("Average NH/AV = ", np.average(ys, weights=1 / covs[:, 1, 1]))

    def eval_nhav(rv):
        x = 1 / rv
        dm = x
        db = 1
        y = x * r["m"] + r["b"]
        varterm = (dm * r["m_unc"]) ** 2 + (db * r["b_unc"]) ** 2
        covterm = dm * db * r["mb_cov"]
        return y, np.sqrt(varterm + covterm)

    nhav_3d1, nhav_unc = eval_nhav(3.1)
    nhav_2, _ = eval_nhav(2)
    nhav_6, _ = eval_nhav(6)

    print("Evaluated at galactic average 1/RV = 0.32,  NH/AV() = ", nhav_3d1)
    print("Error with covariance = ", nhav_unc)
    y_samples = 0.32 * r["m_samples"] + r["b_samples"]
    print(y_samples)
    print("Error based on samples = ", np.std(y_samples))

    print("Value at RV = 2: ", nhav_2)
    print("Value at RV = 6: ", nhav_6)

    ax = axs[0, 1]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "A1000_AV",
        "NH_AV",
        pyrange=[0, max_nh_av],
        ignore_comments=["hi_h_av"],
        mark_comments=MARK_STRING,
    )
    r = plot_results_fit(xs, ys, covs, ax, auto_outliers=False)
    fit_results_table.append(latex_table_line("\\akav", "\\nhav", r))
    # out = r["outlier_idxs"]
    plot_rho_box(
        ax, xs, ys, covs, method="cov approx", optional_plot_fname="a1000_nhav_rho.pdf"
    )

    ax = axs[0, 2]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "A2175_AV",
        "NH_AV",
        pyrange=[0, max_nh_av],
        ignore_comments=["hi_h_av"],
        mark_comments=MARK_STRING,
    )
    r = plot_results_fit(xs, ys, covs, ax, auto_outliers=False)
    out = r["outlier_idxs"]
    plot_rho_box(
        ax,
        *delete_from_arrays((xs, ys, covs), out),
        method="cov approx",
    )
    fit_results_table.append(latex_table_line("\\abumpav", "\\nhav", r))

    if not no_fh2:
        if ignore4:
            special4 = np.where(data["comment"] == "lo_h_av")
        else:
            special4 = []

        ax = axs[1, 0]
        xs, ys, covs = plot_results_scatter(
            ax,
            data,
            "1_RV",
            "fh2",
            # data_comp=comp,
            data_bohlin=bohlin,
            # ignore_comments=mark_comments,
            mark_comments=MARK_STRING,
        )
        xs, ys, covs = delete_from_arrays((xs, ys, covs), special4)
        plot_rho_box(ax, xs, ys, covs, method="nocov")

        ax = axs[1, 1]
        xs, ys, covs = plot_results_scatter(
            ax,
            data,
            "A1000_AV",
            "fh2",
            data_bohlin=bohlin,
            # ignore_comments=["hi_h_av"],
            mark_comments=MARK_STRING,
        )
        xs, ys, covs = delete_from_arrays((xs, ys, covs), special4)
        plot_rho_box(ax, xs, ys, covs, method="nocov")

        ax = axs[1, 2]
        xs, ys, covs = plot_results_scatter(
            ax,
            data,
            "A2175_AV",
            "fh2",
            data_bohlin=bohlin,
            # ignore_comments=["hi_h_av"],
            mark_comments=MARK_STRING,
        )
        xs, ys, covs = delete_from_arrays((xs, ys, covs), special4)
        plot_rho_box(ax, xs, ys, covs, method="nocov")

    # plt.show()
    finalize_double_grid(fig, axs, "rv_trends")

    # there is a number that goes with this plot: the fit evaluated at RV = 3.1 or 1/RV = 0.32.

    # fit result from RVI vs NHAV
    m = 1.02e22
    sm = 8.53e20
    b = -1.27e21
    sb = 1.47e20
    x = 0.32
    nhav_eval = m * x + b
    nhav_err = math.sqrt((sm * x) ** 2 + sb**2)
    print("NH_AV evaluated at Galactic average 1_RV=0.32:", nhav_eval, " pm ", nhav_err)


def plot2b_perh():
    """Compares extinction normalized per H

    [done] Needs implementation of covariance due to NH
    """
    fig, axs = plt.subplots(1, 2, sharex="col", sharey="row", squeeze=False)
    fig.set_size_inches(paper_rcparams.base_width, paper_rcparams.base_width * 2 / 3)

    ax = axs[0, 0]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "A1000_NH",
        "AV_NH",
        mark_comments=MARK_STRING,
    )

    ax = axs[0, 1]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "A2175_NH",
        "AV_NH",
        mark_comments=MARK_STRING,
    )

    finalize_double_grid(fig, axs, "nh_normalized")


def plot3_fm90(hide_alternative=False):
    """FM90 vs fh2."""
    if hide_alternative:
        nrows = 3
    else:
        nrows = 4
    fig, axs = plt.subplots(nrows, 2, sharey=True)

    ax = axs[0, 0]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "CAV1",
        "fh2",
        mark_comments=MARK_STRING,
    )
    plot_rho_box(ax, xs, ys, covs)

    ax = axs[0, 1]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "CAV2",
        "fh2",
        mark_comments=MARK_STRING,
    )
    plot_rho_box(ax, xs, ys, covs)

    ax = axs[1, 0]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "CAV3",
        "fh2",
        mark_comments=MARK_STRING,
    )
    plot_rho_box(ax, xs, ys, covs)

    ax = axs[1, 1]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "CAV4",
        "fh2",
        mark_comments=MARK_STRING,
    )
    plot_rho_box(ax, xs, ys, covs)

    ax = axs[2, 0]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "gamma",
        "fh2",
        mark_comments=MARK_STRING,
    )
    plot_rho_box(ax, xs, ys, covs)

    ax = axs[2, 1]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "x_o",
        "fh2",
        mark_comments=MARK_STRING,
    )
    plot_rho_box(ax, xs, ys, covs)

    if not hide_alternative:

        ax = axs[3, 0]
        xs, ys, covs = plot_results_scatter(
            ax,
            data,
            "bump_amp",
            "fh2",
            mark_comments=MARK_STRING,
        )
        plot_rho_box(ax, xs, ys, covs)

        ax = axs[3, 1]
        xs, ys, covs = plot_results_scatter(
            ax,
            data,
            "bump_area",
            "fh2",
            mark_comments=MARK_STRING,
        )
        plot_rho_box(ax, xs, ys, covs)

    fig.set_size_inches(paper_rcparams.base_width, paper_rcparams.base_height)
    for (ax_l, ax_r) in axs:
        ax_r.set_ylabel("")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.03, right=0.99, left=0.1, bottom=0.05, top=0.99)
    save(fig, "fh2_vs_fm90", need_hspace=True)


def plot4(add_t_vs_n=False, mark=False):
    """
    This should show some extra things I discovered in my big corner
    plot. Most important ones: When CAV4 is high, T01 is low and denhtot
    is high! Let's do those first, and then take another look at the
    corner plot.

    Additional things to show: T01 decreases with log(denhtot) (but noisy plot)

    x values: CAV4
    y values: T01 and denhtot
    """
    mark_string = MARK_STRING if mark else None
    cols = 1
    if add_t_vs_n:
        cols += 1
    fig, axs = plt.subplots(2, cols, sharex="col", squeeze=False)
    ax = axs[0, 0]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "CAV4",
        "T01",
        mark_comments=mark_string,
    )
    plot_rho_box(ax, xs, ys, covs)

    ax = axs[1, 0]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "CAV4",
        "denhtot",
        mark_comments=mark_string,
    )
    ax.set_yscale("log")
    plot_rho_box(ax, xs, ys, covs)

    if add_t_vs_n:
        ax = axs[1, 1]
        xs, ys, covs = plot_results_scatter(
            ax,
            data,
            "T01",
            "denhtot",
            # data_comp=comp,
            # mark_comments=MARK_STRING,
        )
        ax.set_yscale("log")
        plot_rho_box(ax, xs, ys, covs)

        ax = axs[0, 1]
        xs, ys, covs = plot_results_scatter(
            ax,
            data,
            "T01",
            "fh2",
            # data_comp=comp,
            # mark_comments=MARK_STRING,
        )
        ax.set_xlim(40, 140)
        plot_rho_box(ax, xs, ys, covs)

    if add_t_vs_n:
        fig.set_size_inches(
            paper_rcparams.base_width, paper_rcparams.base_height * 2 / 3
        )
        fig.subplots_adjust(wspace=0.3)
    else:
        fig.set_size_inches(
            paper_rcparams.column_width, paper_rcparams.base_height * 2 / 3
        )
    save(fig, "temp_dens", need_wspace=True)


def plot_c4_talk():
    """
    Show only the CAV4 results, so I can fit all of them on one slide in my talk

    x values: CAV4
    y values: T01, denhtot, fh2
    """
    fig, axs = plt.subplots(1, 3, sharey="row", squeeze=False)

    ax = axs[0, 0]
    _ = plot_results_scatter(
        ax,
        data,
        "fh2",
        "CAV4",
        mark_comments=MARK_STRING,
    )
    # ax.set_xscale("log")

    ax = axs[0, 1]
    _ = plot_results_scatter(
        ax,
        data,
        "denhtot",
        "CAV4",
        mark_comments=MARK_STRING,
    )
    ax.set_xscale("log")

    ax = axs[0, 2]
    _ = plot_results_scatter(
        ax,
        data,
        "T01",
        "CAV4",
        mark_comments=MARK_STRING,
    )
    # ax.set_xscale("log")
    ax.set_xlim(40, 140)

    fig.set_size_inches(paper_rcparams.base_width, paper_rcparams.base_height * 1 / 3)
    fig.subplots_adjust(wspace=0.3)
    # save(fig, "c4", need_wspace=False)
    finalize_double_grid(fig, axs, "c4")


def plot5_null():
    """Some interesting null results.

    Most important one: NH/AV vs fh2"""
    fig, axs = plt.subplots()
    ax = axs
    max_nh_av = 5e21
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "fh2",
        "NH_AV",
        pyrange=[0, max_nh_av],
        # data_comp=comp,
        ignore_comments=["hi_h_av"],
        mark_comments=MARK_STRING,
    )
    fig.set_size_inches(paper_rcparams.column_width, paper_rcparams.column_width)
    save(fig, "null")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputtype", default=None)
    args = ap.parse_args()
    if args.outputtype is not None:
        OUTPUT_TYPE = args.outputtype
    # for presentations, we clean up the plots a bit with some of the
    # parameters given here
    # plot1_column_column()
    plot2_ratio_ratio(no_fh2=False)
    # plot2b_perh()
    plot3_fm90(hide_alternative=True)
    # plot4()
    # plot5_null()
    # plot_c4_talk()
    # for line in fit_results_table:
    # print(line)

    plot1_poster()
