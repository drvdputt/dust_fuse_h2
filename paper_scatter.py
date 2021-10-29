"""A limited selection of scatter plots to show in the paper.

These might have specific additions or omissions per plot, depending on
what's interesting to show. E.g., only fit a line when relevant.

Some of them are grouped using subplots

"""

from get_data import get_merged_table, get_bohlin78, get_shull2021
import plot_fuse_results
from plot_fuse_results import (
    plot_results_scatter,
    plot_results_fit,
    match_comments,
)
from matplotlib import pyplot as plt
from astropy.table import Column
import numpy as np
from uncertainties import ufloat
import math

plt.rcParams.update({"font.family": "times"})

# change colors like this
plot_fuse_results.MAIN_COLOR = "k"

# base width for figures. Is equal to the text width over two columns
base_width = 7.1
base_height = 9.3


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


def finalize_single(fig, filename):
    fig.set_size_inches(base_width / 2, 2 / 3 * base_width)
    save(fig, filename)


def finalize_double(fig, filename):
    for ax in fig.axes[1:]:
        ax.set_ylabel("")
    fig.set_size_inches(base_width, 2 / 3 * base_width)
    save(fig, filename)


def finalize_vertical(fig, filename):
    for ax in fig.axs[:-1]:
        ax.set_xlabel("")
    fig.set_size_inches(base_width / 2, base_width)
    save(fig, filename)


def finalize_double_grid(fig, axs, filename):
    # turn off xlabel for everything but last row
    for ax in axs[:-1].flatten():
        ax.set_xlabel("")
    # turn off ylabel for everything but last column
    for ax in axs[:, 1:].flatten():
        ax.set_ylabel("")
    save(fig, filename)


def save(fig, filename, need_hspace=False, need_wspace=False):
    if not need_hspace:
        fig.subplots_adjust(hspace=0.02)
    if not need_wspace:
        fig.subplots_adjust(wspace=0.02)
    fig.subplots_adjust(right=0.99)
    fig.savefig("paper-plots/" + filename)


# Simply a list of lines for the table. It's a global variable, and stuff is added to it when the different plot functions below are executed.
fit_results_table = []


def format_number_with_unc(x, x_unc):
    """This should go in my general toolbox later.

    Uses 'uncertainties' module, but changes the printing to be more
    latexy."""
    # heuristic for the number of figures after the decimal point
    numdecimal = max(1, int(math.floor(math.log10(abs(x_unc / x)))))

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
    m_and_unc_str = format_number_with_unc(
        fit_results_dict["m"], fit_results_dict["m_unc"]
    )
    b_and_b_unc_str = format_number_with_unc(
        fit_results_dict["b"], fit_results_dict["b_unc"]
    )
    return f"{xparam} & {yparam} & {m_and_unc_str} & {b_and_b_unc_str}\\\\"


def plot1():
    """The first plot shows gas columns vs dust columns.

    Main things to show:
    - outliers in AV relation
    - not outliers in EBV
    - A1000 is very correlated with NH2, and not with AV
    - NHI and NHTOT are correlated with AV, but less with A1000
    - Also show NHtot
    """

    fig, axs = plt.subplots(3, 3, sharey="row", sharex="col")
    fig.set_size_inches(base_width, base_width)

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
        report_rho=False,
    )
    out = np.where(match_comments(data, ["lo_h_av", "hi_h_av"]))[0]
    r = plot_results_fit(
        xs, ys, covs, ax, report_rho=True, outliers=out, auto_outliers=True
    )
    fit_results_table.append(latex_table_line("\\av", "\\nh", r))
    # print("AV vs nhtot outliers: ", data['name'][

    ax = choose_ax("AV", "nhi")
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "AV",
        "nhi",
        # data_comp=comp,
        data_bohlin=bohlin,
        mark_comments=["lo_h_av"],
    )
    ax = choose_ax("AV", "nh2")
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "AV",
        "nh2",
        # data_comp=comp,
        data_bohlin=bohlin,
        mark_comments=["lo_h_av"],
    )

    ax = choose_ax("EBV", "nhtot")
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "EBV",
        "nhtot",
        # data_comp=comp,
        data_bohlin=bohlin,
        mark_comments=["lo_h_av"],
        # ignore_comments=["hi_h_av"],
        report_rho=False,
    )
    r = plot_results_fit(xs, ys, covs, ax, auto_outliers=True, report_rho=True)
    fit_results_table.append(latex_table_line("\\ebv", "\\nh", r))

    ax = choose_ax("EBV", "nhi")
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "EBV",
        "nhi",
        # data_comp=comp,
        data_bohlin=bohlin,
        mark_comments=["lo_h_av"],
    )

    ax = choose_ax("EBV", "nh2")
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "EBV",
        "nh2",
        # data_comp=comp,
        data_bohlin=bohlin,
        mark_comments=["lo_h_av"],
    )

    ax = choose_ax("A1000", "nhtot")
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "A1000",
        "nhtot",
        data_bohlin=bohlin,
        mark_comments=["lo_h_av"],
    )

    ax = choose_ax("A1000", "nhi")
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "A1000",
        "nhi",
        data_bohlin=bohlin,
        mark_comments=["lo_h_av"],
    )

    ax = choose_ax("A1000", "nh2")
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "A1000",
        "nh2",
        data_bohlin=bohlin,
        mark_comments=["lo_h_av"],
        report_rho=False,
    )
    r = plot_results_fit(
        xs,
        ys,
        covs,
        ax,
        auto_outliers=True,
        fit_includes_outliers=True,
        report_rho=True,
    )
    fit_results_table.append(latex_table_line("\\ak", "\\nhtwo", r))

    for ax in axs[1:, 0]:
        ax.yaxis.offsetText.set_visible(False)

    axs[0][0].legend(bbox_to_anchor=(1.5, 1), loc="lower center", ncol=4)

    fig.tight_layout()
    finalize_double_grid(fig, axs, "column_vs_column.pdf")


def plot2():
    """Ratio vs ratio.

    x: RV and maybe A1000/AV (extinction ratios)
    y: NH/AV and fh2 (column ratios)

    """
    fig, axs = plt.subplots(2, 3, sharex="col", sharey="row")
    fig.set_size_inches(base_width, base_width * 2 / 3)

    max_nh_av = 5e21

    ax = axs[0, 0]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "1_RV",
        "NH_AV",
        pyrange=[0, max_nh_av],
        # data_comp=comp,
        ignore_comments=["hi_h_av"],
        mark_comments=["lo_h_av"],
        report_rho=False,
    )
    r = plot_results_fit(xs, ys, covs, ax, auto_outliers=False, report_rho=True)
    fit_results_table.append(latex_table_line("\\rvi", "\\nhav", r))
    print("Average NH/AV = ", np.average(ys, weights=1 / covs[:, 1, 1]))

    ax = axs[0, 1]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "A1000_AV",
        "NH_AV",
        pyrange=[0, max_nh_av],
        ignore_comments=["hi_h_av"],
        mark_comments=["lo_h_av"],
    )
    r = plot_results_fit(xs, ys, covs, ax, auto_outliers=False)
    fit_results_table.append(latex_table_line("\\\akav", "\\nhav", r))

    ax = axs[0, 2]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "A2175_AV",
        "NH_AV",
        pyrange=[0, max_nh_av],
        ignore_comments=["hi_h_av"],
        mark_comments=["lo_h_av"],
    )
    r = plot_results_fit(xs, ys, covs, ax, auto_outliers=False)
    fit_results_table.append(latex_table_line("\\abumpav", "\\nhav", r))

    ax = axs[1, 0]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "1_RV",
        "fh2",
        # data_comp=comp,
        data_bohlin=bohlin,
        ignore_comments=["lo_h_av"],
        report_rho=True,
    )
    # plot_results_fit(xs, ys, covs, ax, auto_outliers=True, report_rho=True)

    ax = axs[1, 1]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "A1000_AV",
        "fh2",
        data_bohlin=bohlin,
        # ignore_comments=["hi_h_av"],
        mark_comments=["lo_h_av"],
    )

    ax = axs[1, 2]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "A2175_AV",
        "fh2",
        data_bohlin=bohlin,
        # ignore_comments=["hi_h_av"],
        mark_comments=["lo_h_av"],
    )
    finalize_double_grid(fig, axs, "rv_trends.pdf")

    # there is a number that goes with this plot: the fit evaluated at RV = 3.1 or 1/RV = 0.32.

    # fit result from RVI vs NHAV
    m = 1.02e22
    sm = 8.53e20
    b = -1.27e21
    sb = 1.47e20
    x = 0.32
    nhav_eval = m * x + b
    nhav_err = math.sqrt((sm * x) ** 2 + sb ** 2)
    print("NH_AV evaluated at Galactic average 1_RV=0.32:", nhav_eval, " pm ", nhav_err)


def plot3():
    """FM90 vs fh2."""
    fig, axs = plt.subplots(4, 2, sharey=True)
    _ = plot_results_scatter(
        axs[0, 0],
        data,
        "CAV1",
        "fh2",
        mark_comments=["lo_h_av"],
    )
    _ = plot_results_scatter(
        axs[0, 1],
        data,
        "CAV2",
        "fh2",
        mark_comments=["lo_h_av"],
    )
    _ = plot_results_scatter(
        axs[1, 0],
        data,
        "CAV3",
        "fh2",
        mark_comments=["lo_h_av"],
    )
    _ = plot_results_scatter(
        axs[1, 1],
        data,
        "CAV4",
        "fh2",
        mark_comments=["lo_h_av"],
    )
    _ = plot_results_scatter(
        axs[2, 0],
        data,
        "gamma",
        "fh2",
        mark_comments=["lo_h_av"],
    )
    _ = plot_results_scatter(
        axs[2, 1],
        data,
        "x_o",
        "fh2",
        mark_comments=["lo_h_av"],
    )
    _ = plot_results_scatter(
        axs[3, 0],
        data,
        "bump_amp",
        "fh2",
        mark_comments=["lo_h_av"],
    )
    _ = plot_results_scatter(
        axs[3, 1],
        data,
        "bump_area",
        "fh2",
        mark_comments=["lo_h_av"],
    )

    # this one is already in rv trends plot
    # _ = plot_results_scatter(
    #     axs[3, 1],
    #     data,
    #     "A2175_AV",
    #     "fh2",
    #     mark_comments=["lo_h_av"],
    # )

    fig.set_size_inches(base_width, base_height)
    for (ax_l, ax_r) in axs:
        ax_r.set_ylabel("")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.03, right=0.99, left=0.1, bottom=0.05, top=0.99)
    save(fig, "fh2_vs_fm90.pdf", need_hspace=True)


def plot4():
    """
    This should show some extra things I discovered in my big corner
    plot. Most important ones: When CAV4 is high, T01 is low and denhtot
    is high! Let's do those first, and then take another look at the
    corner plot.

    Additional things to show: T01 decreases with log(denhtot)

    x values: CAV4
    y values: T01 and denhtot
    """
    fig, axs = plt.subplots(2, 2, sharex="col")
    ax = axs[0, 0]
    _ = plot_results_scatter(
        ax,
        data,
        "CAV4",
        "T01",
        mark_comments=["lo_h_av"],
    )

    ax = axs[1, 0]
    _ = plot_results_scatter(
        ax,
        data,
        "CAV4",
        "denhtot",
        mark_comments=["lo_h_av"],
    )
    ax.set_yscale("log")

    ax = axs[1, 1]
    _ = plot_results_scatter(
        ax,
        data,
        "T01",
        "denhtot",
        # data_comp=comp,
        # mark_comments=["lo_h_av"],
    )
    ax.set_yscale("log")
    ax = axs[0, 1]
    _ = plot_results_scatter(
        ax,
        data,
        "T01",
        "fh2",
        # data_comp=comp,
        # mark_comments=["lo_h_av"],
    )
    ax.set_xlim(40, 140)

    fig.set_size_inches(base_width, base_height * 2 / 3)
    fig.subplots_adjust(wspace=0.3)
    save(fig, "temp_dens.pdf", need_wspace=True)


if __name__ == "__main__":
    plot1()
    plot2()
    plot3()
    plot4()
    for line in fit_results_table:
        print(line)
