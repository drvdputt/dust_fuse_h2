"""A limited selection of scatter plots to show in the paper.

These might have specific additions or omissions per plot, depending on
what's interesting to show. E.g., only fit a line when relevant.

Some of them are grouped using subplots

"""

from get_data import get_merged_table, get_bohlin78, get_shull2021
import plot_fuse_results
from plot_fuse_results import plot_results_scatter, plot_results_fit
from matplotlib import pyplot as plt
from astropy.table import Column
import math
import numpy as np

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
for name in ["HD200775", "HD164906", "HD045314", "HD206773"]:
    set_comment(name, "lo_h_av")

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
        data_comp=comp,
        data_bohlin=bohlin,
        ignore_comments=["lo_h_av", "hi_h_av"],
    )
    plot_results_fit(xs, ys, covs, ax)
    ax = choose_ax("AV", "nhi")
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "AV",
        "nhi",
        data_comp=comp,
        data_bohlin=bohlin,
        mark_comments=["lo_h_av", "hi_h_av"],
    )
    ax = choose_ax("AV", "nh2")
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "AV",
        "nh2",
        data_comp=comp,
        data_bohlin=bohlin,
        mark_comments=["lo_h_av", "hi_h_av"],
    )

    ax = choose_ax("EBV", "nhtot")
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "EBV",
        "nhtot",
        data_comp=comp,
        data_bohlin=bohlin,
        mark_comments=["lo_h_av"],
        ignore_comments=["hi_h_av"],
    )
    plot_results_fit(xs, ys, covs, ax)
    ax = choose_ax("EBV", "nhi")
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "EBV",
        "nhi",
        data_comp=comp,
        data_bohlin=bohlin,
        mark_comments=["lo_h_av", "hi_h_av"],
    )
    ax = choose_ax("EBV", "nh2")
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "EBV",
        "nh2",
        data_comp=comp,
        data_bohlin=bohlin,
        mark_comments=["lo_h_av", "hi_h_av"],
    )

    ax = choose_ax("A1000", "nhtot")
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "A1000",
        "nhtot",
        data_bohlin=bohlin,
        mark_comments=["lo_h_av", "hi_h_av"],
    )
    ax = choose_ax("A1000", "nhi")
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "A1000",
        "nhi",
        data_bohlin=bohlin,
        mark_comments=["lo_h_av", "hi_h_av"],
    )
    ax = choose_ax("A1000", "nh2")
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "A1000",
        "nh2",
        data_bohlin=bohlin,
        mark_comments=["lo_h_av", "hi_h_av"],
    )
    plot_results_fit(xs, ys, covs, ax)

    for ax in axs[1:, 0]:
        ax.yaxis.offsetText.set_visible(False)

    fig.tight_layout()
    finalize_double_grid(fig, axs, "column_vs_column.pdf")


def plot2():
    """Ratio vs ratio.

    x: RV and maybe A1000/AV (extinction ratios)
    y: NH/AV and fh2 (column ratios)

    """
    fig, axs = plt.subplots(2, 3, sharex="col", sharey="row")
    fig.set_size_inches(base_width, base_width * 2 / 3)

    ax = axs[0, 0]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "1_RV",
        "NH_AV",
        pyrange=[0, 1.0e22],
        data_comp=comp,
        ignore_comments=["hi_h_av"],
        mark_comments=["lo_h_av"],
    )
    plot_results_fit(xs, ys, covs, ax)
    print("Average NH/AV = ", np.average(ys, weights=1 / covs[:, 1, 1]))

    ax = axs[1, 0]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "1_RV",
        "fh2",
        data_comp=comp,
        data_bohlin=bohlin,
        mark_comments=["lo_h_av", "hi_h_av"],
    )
    plot_results_fit(xs, ys, covs, ax)

    ax = axs[0, 1]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "A1000_AV",
        "NH_AV",
        pyrange=[0, 1.0e22],
        ignore_comments=["hi_h_av"],
        mark_comments=["lo_h_av"],
    )
    plot_results_fit(xs, ys, covs, ax)

    ax = axs[1, 1]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "A1000_AV",
        "fh2",
        data_bohlin=bohlin,
        mark_comments=["lo_h_av", "hi_h_av"],
    )

    ax = axs[0, 2]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "A2175_AV",
        "NH_AV",
        pyrange=[0, 1.0e22],
        ignore_comments=["hi_h_av"],
        mark_comments=["lo_h_av"],
    )
    plot_results_fit(xs, ys, covs, ax)

    ax = axs[1, 2]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "A2175_AV",
        "fh2",
        data_bohlin=bohlin,
        mark_comments=["lo_h_av", "hi_h_av"],
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
    fig, axs = plt.subplots(2, 2, sharey="row", sharex="col")
    ax = axs[1, 0]
    _ = plot_results_scatter(
        ax,
        data,
        "CAV4",
        "T01",
        mark_comments=["lo_h_av"],
    )

    ax = axs[0, 0]
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
        "denhtot",
        "T01",
        mark_comments=["lo_h_av"],
    )
    ax.set_xscale("log")

    fig.set_size_inches(base_width, base_height * 2 / 3)
    finalize_double_grid(fig, axs, "temp_dens.pdf")


# plot4: extinction parameter corner plot? (correlations might be a lot
# of work)
