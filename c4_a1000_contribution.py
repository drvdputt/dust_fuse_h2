"""Quick experiment to try to figure out how much c4 contributes to the
total extinction at 1000"""

from get_data import get_merged_table, get_param_and_unc
from extinction_curve_set import ExtinctionCurveSet
from matplotlib import pyplot as plt
import numpy as np
import itertools
import covariance
from plot_fuse_results import (
    MAIN_COLOR,
    MAIN_MARKER,
    MAIN_MARKER_MEW,
    MAIN_MARKER_SIZE,
    plot_rho_box,
    format_colname,
)
from paper_rcparams import column_width

data = get_merged_table()
ecs = ExtinctionCurveSet(data)

w = 1000
x = ecs.w_to_x(w)
a1000_av = ecs.evaluate(w)
a1000_av_rise = ecs.cav4 * ecs.F(x)
a1000_av_rise_unc = ecs.cav4_unc * ecs.F(x)

w = 2175
x = ecs.w_to_x(w)
a2175_av = ecs.evaluate(w)
a2175_av_bump = ecs.cav3 * ecs.D(x)

marker = itertools.cycle(("x", "+", ".", "o", "*"))


def quick_rho_and_scatter(aw_av, label):
    m = next(marker)
    x = data["nh2"]
    y = aw_av * data["AV"]
    rho = np.corrcoef(x, y)[0, 1]
    print(f"{label}: {rho}")
    plt.scatter(x, y, label=label, marker=m)
    plt.xlabel("$N(\\rm{H_2})$")
    plt.ylabel("$A(\\lambda)$")


quick_rho_and_scatter(a1000_av, label="Total")
quick_rho_and_scatter(a1000_av_rise, label="$C_4 F(x)$")
# quick_rho_and_scatter(a1000_av - a1000_av_rise, label="difference")
plt.legend()

show_2175 = False
if show_2175:
    plt.figure()
    quick_rho_and_scatter(a2175_av, label="total")
    quick_rho_and_scatter(a2175_av_bump, label="C_3 D(x)")
    plt.legend()

# copied over the main plot command from. Doing it this way because I
# don't need any of the feature marking / ignoring that are in
# plot_results_scatter()

av, av_unc = get_param_and_unc("AV", data)
av_unc_r = av_unc / av


def paper_style_scatter(
    yparam,
    awlabel,
    aw_av,
    aw_av_unc,
    color=None,
    marker=None,
    s=None,
    do_rho_box=False,
    legend_label=None,
):
    # get one of the x columns
    y, y_unc = get_param_and_unc(yparam, data)
    ylabel = format_colname(yparam)

    # convert to absolute extinction. do not count av uncertainty. It is
    # already included in aw_av. We are simply removing the factor, so
    # it's already overestimated anyway)
    aw = aw_av * av
    aw_unc = aw * (aw_av_unc / aw_av)
    covs = covariance.make_cov_matrix(aw_unc**2, y_unc**2)

    ax = plt.gca()
    covariance.plot_scatter_auto(
        ax,
        aw,
        y,
        covs,
        1,
        color=MAIN_COLOR if color is None else color,
        marker=MAIN_MARKER if marker is None else marker,
        linewidth=MAIN_MARKER_MEW,
        s=MAIN_MARKER_SIZE if s is None else s,
        label=legend_label,
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel(awlabel)

    # assumes that x_unc is not correlated with y_unc
    if do_rho_box:
        plot_rho_box(ax, aw, y, covs, method="nocov")

    plt.gcf().set_size_inches(column_width, column_width)


a1000_av, a1000_av_unc = get_param_and_unc("A1000_AV", data)

plt.figure()
# paper_style_scatter(
#     "nh2",
#     "A(1000)",
#     a1000_av,
#     a1000_av_unc,
#     color="xkcd:grey",
#     marker="x",
#     legend_label="Total",
# )
paper_style_scatter(
    "nh2",
    # "A(1000)",
    "$A_{\\mathrm{rise}}(1000)$",
    a1000_av_rise,
    a1000_av_rise_unc,
    do_rho_box=True,
    legend_label="FUV rise",
    s=10
)
# plt.legend()
plt.tight_layout()
plt.savefig("paper-plots/c4_contribution.pdf", bbox_inches="tight")

# plt.figure()
# a1000_av, a1000_av_unc = get_param_and_unc("A1000_AV", data)
# paper_style_scatter("nhtot", "A(1000)", a1000_av, a1000_av_unc, color="orange")
# paper_style_scatter("nhtot", "A(1000)", a1000_av_rise, a1000_av_rise_unc, do_rho_box=True)
plt.show()

# just out of curiosity, lets also take a look at the ratio
plt.figure()
ratio = a1000_av_rise / a1000_av
plt.scatter(data['nh2'], ratio)
plt.show()
