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


def save(fig, filename):
    fig.subplots_adjust(wspace=0.02)
    fig.subplots_adjust(hspace=0.02)
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

    fig, axs = plt.subplots(3, 2, sharey="row", sharex="col")
    fig.set_size_inches(base_width, 4 / 3 * base_width)

    # do not use loop or other abstractions here, so we can manually
    # adust each plot as needed

    ax = axs[0, 0]
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

    ax = axs[1, 0]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "AV",
        "nhi",
        data_comp=comp,
        data_bohlin=bohlin,
        mark_comments=["lo_h_av", "hi_h_av"],
    )

    ax = axs[2, 0]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "AV",
        "nh2",
        data_comp=comp,
        data_bohlin=bohlin,
        mark_comments=["lo_h_av", "hi_h_av"],
    )

    ax = axs[0, 1]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "A1000",
        "nhtot",
        data_bohlin=bohlin,
        mark_comments=["lo_h_av"],
        ignore_comments=["hi_h_av"],
    )

    ax = axs[1, 1]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "A1000",
        "nhi",
        data_bohlin=bohlin,
        mark_comments=["lo_h_av", "hi_h_av"],
    )

    ax = axs[2, 1]
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
    fig.set_size_inches(base_width, base_width)

    ax = axs[0, 0]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "1_RV",
        "NH_AV",
        pyrange=[0, 1.0e22],
        data_comp=comp,
        mark_comments=["lo_h_av", "hi_h_av"],
    )
    plot_results_fit(xs, ys, covs, ax)

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
        # data_comp=comp,
        mark_comments=["lo_h_av", "hi_h_av"],
    )
    plot_results_fit(xs, ys, covs, ax)

    ax = axs[1, 1]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "A1000_AV",
        "fh2",
        # data_comp=comp,
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
        # data_comp=comp,
        mark_comments=["lo_h_av", "hi_h_av"],
    )
    plot_results_fit(xs, ys, covs, ax)

    ax = axs[1, 2]
    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        "A2175_AV",
        "fh2",
        # data_comp=comp,
        data_bohlin=bohlin,
        mark_comments=["lo_h_av", "hi_h_av"],
    )
    finalize_double_grid(fig, axs, "rv_trends.pdf")


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
        "A2175_AV",
        "fh2",
        mark_comments=["lo_h_av"],
    )

    fig.set_size_inches(base_width, base_height)
    for (ax_l, ax_r) in axs:
        ax_r.set_ylabel("")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.03, right=0.99, left=0.1, bottom=0.05, top=0.99)
    save(fig, "fh2_vs_fm90.pdf")


# plot4: extinction parameter corner plot?

# plot5: something with T01?

plot1()
plot2()
plot3()
