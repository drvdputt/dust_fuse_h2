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
    plot_results_fit,
)
from paper_rcparams import column_width
from math import pi
from scipy import integrate

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
a2175_av_bump = ecs.cav3 / ecs.gamma**2
# quick check if this matters at 2175 A
a2175_av_rise = ecs.cav4_unc * ecs.F(x)

# try getting bump area (area of lorentian is pi * gamma. Not sure about drude)
bump_area_approx = pi * ecs.gamma * ecs.cav3

wgrid = np.linspace(1000, 3000, 2000)
N = len(ecs.cav1)
y = np.zeros((N, len(wgrid)))
for iw, w in enumerate(wgrid):
    y[:, iw] = ecs.cav3 * ecs.D(ecs.w_to_x(w))
bump_area_exact = integrate.trapezoid(y, x=wgrid, axis=1)

marker = itertools.cycle(("x", "+", ".", "o", "*"))


def quick_rho_and_scatter(aw_av, label, quantity=None):
    if quantity is None:
        xparam = "nh2"
    else:
        xparam = quantity
    m = next(marker)
    x = data[xparam]
    y = aw_av * data["AV"]
    rho = np.corrcoef(x, y)[0, 1]
    print(f"{label}: {rho:.2f}")
    plt.scatter(x, y, label=label + f"; r = {rho:.2f}", marker=m)
    plt.xlabel(xparam)
    plt.ylabel("$A(\\lambda)$")


quick_rho_and_scatter(a1000_av, label="Total")
quick_rho_and_scatter(a1000_av_rise, label="$A(V) C_4 F(x)$")
quick_rho_and_scatter(a1000_av - a1000_av_rise, label="Rest")
plt.legend()

show_nhi = False
if show_nhi:
    plt.figure()
    quick_rho_and_scatter(a1000_av_rise, label="Rise", quantity="nhtot")
    quick_rho_and_scatter(a1000_av - a1000_av_rise, label="Rest", quantity="nhtot")
    plt.legend()

show_2175 = True
if show_2175:
    plt.figure()
    # quick_rho_and_scatter(a2175_av, label="Total")
    quick_rho_and_scatter(a2175_av_bump, label="$A(V) C_3 / \\gamma^2$")
    quick_rho_and_scatter(a2175_av - a2175_av_bump, label="Rest")
    quick_rho_and_scatter(bump_area_exact, label="area")
    quick_rho_and_scatter(bump_area_approx, label="area approx")
    plt.legend()

# experiment: assume that the "baseline" of the extinction cruve
# flattens at a certain wavelength. Then the FUV rise contribution is
# the difference between A(1000) and A(w).
show_uvdiff = False
if show_uvdiff:
    plt.figure()
    for w_compare in [1250, 1500, 1750]:
        uvdiff = a1000_av - ecs.evaluate(w_compare)
        quick_rho_and_scatter(uvdiff, label=f"A1000 - A{w_compare}")
# conclusion: these work amost as well as the FUV rise (about 0.88
# for w_compare = 1250). But the shorter we go in wavelength, the
# more this difference is dominated by C4, so this is nothing new.

plt.legend()

# plt.show()
# exit()

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
    aw_av_diff=None,
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

    # add the difference, but with less details
    if aw_av_diff is not None:
        ax.scatter(
            aw_av_diff * av, y, color="orange", alpha=0.5, label="Rest", marker="*"
        )

    # assumes that x_unc is not correlated with y_unc
    if do_rho_box:
        plot_rho_box(ax, aw, y, covs, method="nocov")

    # fit
    plot_results_fit(aw, y, covs, ax)

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
    "$A(1000)$",
    # "$A_{\\mathrm{rise}}(1000)$",
    a1000_av_rise,
    a1000_av_rise_unc,
    do_rho_box=True,
    legend_label="Rise",
    s=10,
    aw_av_diff=a1000_av - a1000_av_rise,
)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("paper-plots/c4_contribution.pdf", bbox_inches="tight")

# plt.figure()
# paper_style_scatter(
#     "nhtot",
#     "$A(1000)$",
#     # "$A_{\\mathrm{rise}}(1000)$",
#     a1000_av_rise,
#     a1000_av_rise_unc,
#     do_rho_box=True,
#     legend_label="Rise",
#     s=10,
#     aw_av_diff=a1000_av - a1000_av_rise,
# )
# plt.legend(loc="lower right")
# plt.tight_layout()


# plt.figure()
# a1000_av, a1000_av_unc = get_param_and_unc("A1000_AV", data)
# paper_style_scatter("nhtot", "A(1000)", a1000_av, a1000_av_unc, color="orange")
# paper_style_scatter("nhtot", "A(1000)", a1000_av_rise, a1000_av_rise_unc, do_rho_box=True)
plt.show()

# just out of curiosity, lets also take a look at the ratio
plt.figure()
ratio = a1000_av_rise / a1000_av
plt.scatter(data["nh2"], ratio)
plt.show()
