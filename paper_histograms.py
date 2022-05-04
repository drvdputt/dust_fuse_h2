"""Demonstration of rho-histograms"""
import paper_rcparams
import get_data
import pearson
from matplotlib import pyplot as plt
from plot_fuse_results import format_colname


data = get_data.get_merged_table()


def rho_histogram(ax, xparam, yparam, comment, vert_label=False):
    """
    vert_label: bool
        Label the vertical lines
    """
    annotation_size = 9
    xs, ys, covs = get_data.get_xs_ys_covs(data, xparam, yparam)
    rho_median, std_null = pearson.pearson_mc_nocov(xs, ys, covs, hist_ax=ax)
    xlabel = format_colname(xparam).split(" ")[0]
    ylabel = format_colname(yparam).split(" ")[0]
    text = f"$x = ${xlabel}\n$y = ${ylabel}\n{comment}"
    ax.text(0.03, 0.9, text, transform=ax.transAxes, va="top", fontsize=annotation_size)
    ax.tick_params("x", direction="inout")

    def weighted_y(f):
        return (1 - f) * ax.get_ylim()[0] + f * ax.get_ylim()[1]

    x_offset = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01
    y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.04

    # indicate median rho
    ax.axvline(rho_median, color="xkcd:bright blue")
    if vert_label:
        ax.text(rho_median + x_offset, weighted_y(0.8), "median", color="k", ha="left")

    # indicate null sigma
    ax.axvline(0, color="xkcd:gray")
    # if vert_label:
    #     ax.text(0 + x_offset, weighted_y(0.8), "$\\rho_0 = 0$", color="k", ha="left")

    # 1 sigma arrow
    # ax.arrow(0, weighted_y(0.5), std_null, 0)
    height1 = weighted_y(0.6)
    ax.annotate(
        "",
        xy=(std_null, height1),
        xytext=(0, height1),
        xycoords="data",
        arrowprops=dict(arrowstyle="->"),
    )
    ax.text(
        std_null * 0.8,
        height1 + y_offset,
        "$\\sigma(r_0)$",
        ha="center",
        fontsize=annotation_size,
    )
    # n sigmfa arrow
    # ax.arrow(0, weighted_y(0.4), rho_median, 0, length_includes_head=True, head_length=3, head_width=3)
    height2 = weighted_y(0.4)
    ax.annotate(
        "",
        xy=(rho_median, height2),
        xytext=(0, height2),
        xycoords="data",
        arrowprops=dict(arrowstyle="->"),
    )
    ax.text(
        rho_median * 0.8,
        height2 + y_offset,
        f"${rho_median / std_null:.1f}\\sigma$",
        ha="right",
        fontsize=annotation_size,
    )


fig, ax = plt.subplots(2, 1, sharex=True)

rho_histogram(ax[0], "1_RV", "NH_AV", "not significant")
rho_histogram(ax[1], "A1000_AV", "NH_AV", "significant", vert_label=True)

ax[0].legend()
ax[1].set_xlabel("$r$")
ax[1].set_ylabel("number of $r$ samples")
fig.set_size_inches(paper_rcparams.column_width, paper_rcparams.column_width * 1)
fig.subplots_adjust(hspace=0, top=0.996, right=0.996)
fig.savefig("paper-plots/hist.pdf", bbox_inches="tight")
