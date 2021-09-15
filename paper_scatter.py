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
    fig.set_size_inches(base_width, 4/3 * base_width)
    save(fig, filename)


def save(fig, filename):
    fig.subplots_adjust(wspace=0.02)
    fig.subplots_adjust(hspace=0.02)
    fig.tight_layout()
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
    plot_results_fit(xs, ys, covs, ax)

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
    fig.tight_layout()
    finalize_double_grid(fig, axs, "column_vs_column.pdf")


def plot2():
    """Gas to dust ratio main trend."""
    fig, (ax_av_nhav, ax_rvi_nhav) = plt.subplots(1, 2, sharey=True)
    print("NH/AV vs AV")
    xs, ys, covs = plot_results_scatter(
        ax_av_nhav,
        data,
        "AV",
        "NH_AV",
        pyrange=[0, 1.0e22],
        data_comp=comp,
        data_bohlin=bohlin,
        mark_comments=["lo_h_av", "hi_h_av"],
    )
    print("NH_AV vs 1/RV")
    xs, ys, covs = plot_results_scatter(
        ax_rvi_nhav,
        data,
        "1_RV",
        "NH_AV",
        # pyrange=[0, 1.6e22],
        data_comp=comp,
        ignore_comments=["lo_h_av", "hi_h_av"],
    )
    plot_results_fit(xs, ys, covs, ax_rvi_nhav)
    finalize_double(fig, "nhav_vs_av.pdf")

    # redo the same thing, but with the outliers. Just need it for the numbers
    print("NH_AV vs 1/RV (with outliers)")
    xs, ys, covs = plot_results_scatter(
        ax_rvi_nhav,
        data,
        "1_RV",
        "NH_AV",
        # pyrange=[0, 1.6e22],
        data_comp=comp,
        ignore_comments=["hi_h_av"],
        mark_comments=["lo_h_av"],
    )
    plot_results_fit(xs, ys, covs, ax_rvi_nhav)


def plot3():
    """fh2 vs extinction."""
    fig, (ax_l, ax_m, ax_r) = plt.subplots(1, 3, sharey=True)
    _ = plot_results_scatter(
        ax_l,
        data,
        "EBV",
        "fh2",
        data_comp=comp,
        data_bohlin=bohlin,
        data_shull=shull,
        mark_comments=["lo_h_av"],
    )
    _ = plot_results_scatter(
        ax_m,
        data,
        "AV",
        "fh2",
        data_comp=comp,
        data_bohlin=bohlin,
        # data_shull=shull,
        mark_comments=["lo_h_av"],
    )
    _ = plot_results_scatter(
        ax_r,
        data,
        "A1000",
        "fh2",
        # data_comp=comp,
        # data_bohlin=bohlin,
        # data_shull=shull,
        mark_comments=["lo_h_av"],
    )
    finalize_double(fig, "fh2_vs_ext.pdf")


def plot4():
    """Some of the parameters describing the extinction curve vs fh2.

    Two imporant results: fh2 vs 1/RV and fh2 vs CAV4.

    C3 and bump size.

    """
    fig, axs = plt.subplots(3, 2, sharey=True)
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
        "bump_area",
        "fh2",
        mark_comments=["lo_h_av"],
    )
    xs, ys, covs = plot_results_scatter(
        axs[2, 1],
        data,
        "1_RV",
        "fh2",
        data_comp=comp,
        # data_bohlin=bohlin,
        # data_shull=shull,
        ignore_comments=["lo_h_av"],
    )
    plot_results_fit(xs, ys, covs, axs[2, 1])
    fig.set_size_inches(base_width, base_height)
    for (ax_l, ax_r) in axs:
        ax_r.set_ylabel("")
    fig.tight_layout()
    save(fig, "fh2_vs_fm90.pdf")


plot1()
