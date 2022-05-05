"""Demonstration of rho-histograms"""
import paper_rcparams
import get_data
import pearson
from matplotlib import pyplot as plt
from plot_fuse_results import plot_results_scatter
from paper_scatter import data
import numpy as np

# use same data as for the scatter plots


def rho_histogram(ax, xparam, yparam, comment, vert_label=False):
    """
    vert_label: bool
        Label the vertical lines
    """
    annotation_size = 9
    xs, ys, covs = get_data.get_xs_ys_covs(data, xparam, yparam)
    results = pearson.new_rho_method(xs, ys, covs, hist_ax=ax)
    rho_measured, null_std, numsigma = (
        results["measured_rho"],
        results["null_std"],
        results["numsigma"],
    )
    # xlabel = format_colname(xparam).split(" ")[0]
    # ylabel = format_colname(yparam).split(" ")[0]
    # text = f"$x = ${xlabel}\n$y = ${ylabel}\n{comment}"
    # ax.text(0.03, 0.9, text, transform=ax.transAxes, va="top", fontsize=annotation_size)
    ax.tick_params("x", direction="inout")

    def weighted_y(f):
        return (1 - f) * ax.get_ylim()[0] + f * ax.get_ylim()[1]

    x_offset = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.06
    y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.04

    # indicate nullrho
    null_rho = results["null_rho"]
    ax.axvline(null_rho, color="xkcd:bright blue")

    def vertical_line_label(x, text, color):
        if vert_label:
            ax.text(
                x - x_offset,
                weighted_y(0.6),
                text,
                color=color,
                ha="left",
                va="bottom",
                rotation="vertical",
            )

    # indicate measured rho
    vertical_line_label(rho_measured, "measured", 'k')
    # indicate rho intrinsic to null hypothesis
    vertical_line_label(null_rho, "induced offset", "xkcd:bright blue")

    # indicate zero vertical
    ax.axvline(0, color="k", alpha=0.5, ls="dotted", lw=1)


    def numsigma_arrow(ypos, start, end):
        height = weighted_y(ypos)
        ax.annotate(
            "",
            xy=(end, height),
            xytext=(start, height),
            xycoords="data",
            arrowprops=dict(arrowstyle="->"),
        )
        ax.text(
            end - x_offset,
            height,
            f"${np.abs(rho_measured - start) / null_std:.1f}\\sigma$",
            ha="right",
            va="center",
            fontsize=annotation_size,
        )

    numsigma_arrow(0.4, null_rho, rho_measured)
    numsigma_arrow(0.2, 0, rho_measured)


fig, axs = plt.subplots(2, 1)

# rho_histogram(axs[0], "1_RV", "NH_AV", "not significant")

# plot fh2 vs nh_av as demonstration
max_nh_av = 5e21
xs, ys, covs = plot_results_scatter(
    axs[0],
    data,
    "fh2",
    "NH_AV",
    pyrange=[0, max_nh_av],
    # data_comp=comp,
    ignore_comments=["hi_h_av"],
    # mark_comments=MARK_STRING,
)

rho_histogram(axs[1], "fh2", "NH_AV", "", vert_label=True)

axs[1].legend(loc="upper right")
axs[1].set_xlabel("$r$")
axs[1].set_ylabel("number of $r$ samples")
fig.set_size_inches(paper_rcparams.column_width, paper_rcparams.column_width * 2)
# fig.subplots_adjust(hspace=0, top=0.996, right=0.996)
fig.tight_layout()
plt.show()
fig.savefig("paper-plots/hist.pdf", bbox_inches="tight")
