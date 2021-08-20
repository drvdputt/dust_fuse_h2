#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from astropy.modeling import models, fitting
from get_data import get_merged_table, get_bohlin78
import covariance
import linear_ortho_fit

# some easily customizable constants
BOHLIN_COLOR = "xkcd:magenta"
SHULL_COLOR = "xkcd:dark yellow"
COMP_COLOR = "xkcd:sky blue"
MAIN_COLOR = "xkcd:gray"
MARK_COLOR = "r"
MARK_MARKER = "s"
FIT_COLOR = "k"
BAD_COLOR = "r"


def set_params(lw=1.5, universal_color="#262626", fontsize=16):
    """Configure some matplotlib rcParams.

    Parameters
    ----------
    lw : scalar
        Linewidth of plot and axis lines. Default is 1.5.
    universal_color : str, matplotlib named color, rgb tuple
        Color of text and axis spines. Default is #262626, off-black
    fontsize : scalar
        Font size in points. Default is 12
    """
    rc("font", size=fontsize)
    rc("lines", linewidth=lw, markeredgewidth=lw * 0.5)
    rc("patch", linewidth=lw, edgecolor="#FAFAFA")
    rc(
        "axes",
        linewidth=lw,
        edgecolor=universal_color,
        labelcolor=universal_color,
        axisbelow=True,
    )
    rc("image", origin="lower")  # fits images
    rc("xtick.major", width=lw * 0.75)
    rc("xtick.minor", width=lw * 0.5)
    rc("xtick", color=universal_color)
    rc("ytick.major", width=lw * 0.75)
    rc("ytick.minor", width=lw * 0.5)
    rc("ytick", color=universal_color)
    rc("grid", linewidth=lw)
    rc(
        "legend",
        loc="best",
        numpoints=1,
        scatterpoints=1,
        handlelength=1.5,
        fontsize=fontsize,
        columnspacing=1,
        handletextpad=0.75,
    )


def initialize_parser():
    """For running from command line, initialize argparse with common args"""
    ftypes = [
        "png",
        "jpg",
        "jpeg",
        "pdf",
        "ps",
        "eps",
        "rgba",
        "svg",
        "tiff",
        "tif",
        "pgf",
        "svgz",
        "raw",
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--savefig",
        action="store",
        default=False,
        choices=ftypes,
        help="Save figure to a file",
    )
    return parser


def format_colname(name):
    """
    Convert the column name to a better formatted name
    """
    colnames = [
        "AV",
        "RV",
        "EBV",
        "CAV1",
        "CAV2",
        "CAV3",
        "CAV4",
        "C1",
        "C2",
        "C3",
        "C4",
        "x_o",
        "gamma",
        "bump_area",
        "fh2",
        "nhtot",
        "nh2",
        "nhi",
        "NH_AV",
        "NH_EBV",
    ]
    plotnames = [
        "$A(V)$",
        "$R(V)$",
        "$E(B-V)$",
        "$C^{A(V)}_1$",
        "$C^{A(V)}_2$",
        "$C^{A(V)}_3$",
        "$C^{A(V)}_4$",
        "$C_1$",
        "$C_2$",
        "$C_3$",
        "$C_4$",
        "$x_o$",
        r"$\gamma$",
        r"$\pi C^{A(V)}_3 / 2 \gamma$",
        "$f(H_2)$",
        "$N(H)$",
        "$N(H_2)$",
        "$N(HI)$",
        "$N(H)/A(V)$",
        "$N(H)/E(B-V)$",
    ]
    dic_pairs = dict(zip(colnames, plotnames))

    out_name = name
    if name[:3] == "log":
        out_name = r"$\log (" + name[3:].upper() + ")$"
    elif name in dic_pairs.keys():
        out_name = dic_pairs[name]

    return out_name


def plot_results2(
    data,
    xparam,
    yparam,
    pxrange=None,
    pyrange=None,
    data_comp=None,
    data_bohlin=None,
    data_shull=None,
    figsize=None,
    alpha=0.5,
    ignore_comments=None,
    mark_comments=None,
):
    """Plot the fuse results with specificed x and y axes

    Parameters
    ----------
    data: astropy.table
        Table of the data to plot

    xparam: str
        name of column to plot as the x variable

    yparam: str
        name of column to plot as the y variable

    pxrange: float[2]
        min/max x range to plot

    pyrange: float[2]
        min/max y range to plot

    data_comp: astropy.table
        Table of the data to plot for the comparision stars

    data_bohlin: astropy.table 
        Optional, data to plot from Bohlin 1978. Does nothing if xparam
        and yparam are not present.

    data_shull: astropy.table 
        Optional, data to plot from Shull et al. 2021. Does nothing if
        xparam and yparam are not present.

    figsize : float[2]
       x,y size of plot

    ignore_comments : list of str
       exclude points for which data['comment'] equals one of the given
       strings from fitting (they will still be plotted, but in a highlighted
       color)

    mark_comments : list of str
       highlight points for which data['comment'] equals one of the
       given comment strings

    """
    set_params(lw=1, universal_color="#202026", fontsize=10)

    # fig, ax = plt.subplots(figsize=figsize)
    fig, (ax, ax2,) = plt.subplots(figsize=(8, 5), ncols=2)

    # plot bohlin or shull data (not used for fitting)
    def plot_extra_data(extra, label, color):
        if extra is not None and xparam in extra.colnames and yparam in extra.colnames:
            xcol, xcol_unc = get_param_and_unc(xparam, extra)
            ycol, ycol_unc = get_param_and_unc(yparam, extra)
            # errorbar() has no problems with xerr and yerr being None,
            # so don't need to check for uncertainty columns
            ax.errorbar(
                xcol,
                ycol,
                xerr=xcol_unc,
                yerr=ycol_unc,
                label=label,
                color=color,
                linestyle="none",
                marker=".",
                alpha=alpha,
            )

    plot_extra_data(data_bohlin, "Bohlin78", BOHLIN_COLOR)
    plot_extra_data(data_shull, "Shull+21", SHULL_COLOR)

    # plot comparison star data (not used for fitting)
    if data_comp is not None:
        # xs, ys, covs = get_xs_ys_covs(data_comp, xparam, yparam, "AV")
        xs, ys, covs = get_xs_ys_covs_new(data_comp, xparam, yparam)
        covariance.plot_scatter_with_ellipses(
            ax, xs, ys, covs, 1, color=COMP_COLOR, alpha=alpha, label="compar"
        )

    # decide which points to ignore
    if ignore_comments is not None:
        # only use points for which the comment does not match any of
        # the ignore flags
        use = np.logical_and.reduce([c != data["comment"] for c in ignore_comments])
    else:
        use = np.full(len(data), True)

    # decide which points to highlight
    if mark_comments is not None:
        mark = np.logical_or.reduce([c == data["comment"] for c in mark_comments])
    else:
        mark = np.full(len(data), False)

    # choose columns and calculate covariance matrices
    # if yparam[0:3] == "CAV":
    #     cparam = "AV"
    # elif yparam[0:1] == "C":
    #     cparam = "EBV"
    # else:
    #     cparam = "AV"

    # if xparam == "RV" and yparam == "NH_AV":
    #     # WIP: check if these are similar (they aren't, but I think the
    #     # old implementation is incorrect).
    #     old_xs, old_ys, old_covs = get_xs_ys_covs(data[use], xparam, yparam, cparam)
    #     print("old covs", old_covs)
    #     print("new covs", covs)

    # get and plot main data
    xs, ys, covs = get_xs_ys_covs_new(data[use], xparam, yparam)
    covariance.plot_scatter_with_ellipses(
        ax, xs, ys, covs, 1, color=MAIN_COLOR, alpha=alpha, marker="x", label="sample"
    )

    # mark data points, if any
    if mark.any():
        xs_mark, ys_mark, _ = get_xs_ys_covs_new(data[mark], xparam, yparam)
        ax.scatter(
            xs_mark,
            ys_mark,
            facecolors="none",
            edgecolors=MARK_COLOR,
            marker=MARK_MARKER,
        )

    # plot ignored points in different color
    if not use.all():
        # bad_xs, bad_ys, bad_covs = get_xs_ys_covs(
        #     data[np.logical_not(use)], xparam, yparam, cparam
        # )
        bad_xs, bad_ys, bad_covs = get_xs_ys_covs_new(
            data[np.logical_not(use)], xparam, yparam
        )
        covariance.plot_scatter_with_ellipses(
            ax,
            bad_xs,
            bad_ys,
            bad_covs,
            1,
            color=BAD_COLOR,
            alpha=alpha,
            marker="x",
            label="ignore",
        )

    ax.set_xlabel(format_colname(xparam))
    ax.set_ylabel(format_colname(yparam))

    if pxrange is not None:
        ax.set_xlim(pxrange)
    if pyrange is not None:
        ax.set_ylim(pyrange)

    m, b_perp, sm, sb_perp = linear_ortho_fit.linear_ortho_maxlh(
        xs, ys, covs, ax, sigma_hess=True
    )
    b = linear_ortho_fit.b_perp_to_b(m, b_perp)

    boot_cov_mb = linear_ortho_fit.bootstrap_fit_errors(xs, ys, covs)
    boot_sm, boot_sb = np.sqrt(np.diag(boot_cov_mb))

    a = 2
    m_grid, b_perp_grid, logL_grid = linear_ortho_fit.calc_logL_grid(
        m - a * sm, m + a * sm, b_perp - a * sb_perp, b_perp + a * sb_perp, xs, ys, covs
    )
    random_m, random_b_perp = linear_ortho_fit.sample_likelihood(
        m, b_perp, m_grid, b_perp_grid, logL_grid
    )
    sampled_cov_mb = np.cov(random_m, random_b_perp)

    linear_ortho_fit.plot_solution_neighborhood(
        ax2,
        logL_grid,
        [min(b_perp_grid), max(b_perp_grid), min(m_grid), max(m_grid)],
        m,
        b_perp,
        cov_mb=sampled_cov_mb,
        what="L",
        extra_points=zip(random_b_perp, random_m),
    )

    # plot the fitted line
    xlim = ax.get_xlim()
    xp = np.linspace(xlim[0], xlim[1], 3)
    yp = m * xp + b
    ax.plot(xp, yp, color=FIT_COLOR, linewidth=2)

    # plot sampled lines
    linear_ortho_fit.plot_solution_linescatter(
        ax, random_m, random_b_perp, color=FIT_COLOR, alpha=5 / len(random_m)
    )

    # compare to naive regression
    plot_naive_regression(ax, xs, ys, covs)

    return fig


def plot_naive_regression(ax, xs, ys, covs):
    line_init = models.Linear1D()
    fit = fitting.LinearLSQFitter()
    fitted_model_weights = fit(line_init, xs, ys, weights=1.0 / np.sqrt(covs[:, 1, 1]))
    xlim = ax.get_xlim()
    x_mod = np.linspace(xlim[0], xlim[1])
    ax.plot(x_mod, fitted_model_weights(x_mod), linestyle=":", color="y")
    return fitted_model_weights


def get_param_and_unc(param, data):
    """
    Returns the unc column if it is in the table
    """
    d = data[param].data
    unc_key = param + "_unc"
    unc = data[unc_key] if unc_key in data.colnames else None
    return d, unc


def get_xs_ys_covs_new(data, xparam, yparam):
    px, px_unc = get_param_and_unc(xparam, data)
    py, py_unc = get_param_and_unc(yparam, data)

    if xparam == "AV" and yparam == "NH_AV" or xparam == "EBV" and yparam == "NH_EBV":
        covs = covariance.get_cov_x_ydx(px, py, px_unc, py_unc)
    elif xparam == "nhtot" and (yparam == "NH_AV" or yparam == "NH_EBV"):
        covs = covariance.get_cov_x_xdy(px, py, px_unc, py_unc)
    elif xparam == "fh2" and (
        yparam == "NH_AV" or yparam == "NH_EBV" or yparam == "nhtot"
    ):
        x1, x1_unc = get_param_and_unc("nhi", data)
        x2, x2_unc = get_param_and_unc("nh2", data)
        C_fh2_htot = covariance.get_cov_fh2_htot(x1, x2, x1_unc, x2_unc)
        # in case of NH, no extra factor is needed
        if yparam == "NH_AV":
            av, av_unc = get_param_and_unc("AV", data)
            covs = covariance.new_cov_when_divide_y(
                C_fh2_htot, data["nhtot"], av, av_unc
            )
        elif yparam == "NH_EBV":
            ebv, ebv_unc = get_param_and_unc("EBV", data)
            covs = covariance.new_cov_when_divide_y(
                C_fh2_htot, data["nhtot"], ebv, ebv_unc
            )

    elif xparam == "RV" and yparam == "NH_AV":
        # first, get the covariance of AV and NH_AV
        av, av_unc = get_param_and_unc("AV", data)
        C_av_nhav = covariance.get_cov_x_ydx(av, py, av_unc, py_unc)
        # RV = AV / EBV, so adjust x for division by EBV
        ebv, ebv_unc = get_param_and_unc("EBV", data)
        covs = covariance.new_cov_when_divide_x(C_av_nhav, av, ebv, ebv_unc)
    else:
        print(
            "No covariances implemented for this parameter pair. If x and y are uncorrelated, you can dismiss this."
        )
        covs = covariance.make_cov_matrix(px_unc ** 2, py_unc ** 2)

    # Check if cauchy schwarz is satisfied. If not, enforce using fudge
    # factor.
    bad_cov = covs[:, 0, 1] ** 2 > covs[:, 0, 0] * covs[:, 1, 1]
    if (bad_cov).any():
        print("Some covs don't satisfy Cauchy-Schwarz inequality! cov^2 !< Vx * Vy!")
        print("Fudging the correlation to 99% to avoid further problems.")
        covs[bad_cov, 0, 1] = (
            np.sign(covs[bad_cov, 0, 1])
            * 0.99
            * np.sqrt(covs[bad_cov, 0, 0] * covs[bad_cov, 1, 1])
        )
        covs[bad_cov, 1, 0] = covs[bad_cov, 0, 1]

    return px, py, covs


def get_xs_ys_covs(data, xparam, yparam, cparam):
    """
    Return arrays of x, y, and cov(x,y) for a pair of parameters

    """
    x, xerr = get_param_and_unc(xparam, data)
    y, yerr = get_param_and_unc(yparam, data)
    cterm, cterm_unc = get_param_and_unc(cparam, data)

    if (xparam == "AV" and yparam == "NH_AV") or (
        xparam == "EBV" and yparam == "NH_EBV"
    ):
        yfac = yerr / y
        xfac = xerr / x
        corr = -1.0 * xfac / yfac
    elif (
        xparam == "RV"
        and yparam == "NH_AV"
        and cterm is not None
        and cterm_unc is not None
    ):
        avfac = cterm_unc / cterm
        yfac = yerr / y
        corr = -1.0 * avfac / yfac
    elif xparam == "AV" and yparam == "RV":
        yfac = yerr / y
        xfac = xerr / x
        corr = xfac / yfac
    elif (
        ((xparam == "RV") or (xparam == "AV"))
        and ((yparam[0:3] == "CAV") or (yparam == "bump_area"))
        and cterm is not None
        and cterm_unc is not None
    ):
        avfac = cterm_unc / cterm
        yfac = yerr / y
        corr = -1.0 * avfac / yfac
    elif (
        ((xparam == "RV") or (xparam == "EBV"))
        and (yparam[0:1] == "C")
        and cterm is not None
        and cterm_unc is not None
    ):
        ebvfac = cterm_unc / cterm
        yfac = yerr / y
        corr = ebvfac / yfac
    else:
        corr = np.full(len(x), 0.0)

    covs = np.zeros((len(x), 2, 2))
    covs[:, 0, 0] = np.square(xerr)
    covs[:, 1, 1] = np.square(yerr)
    covs[:, 0, 1] = xerr * yerr * corr
    covs[:, 1, 0] = covs[:, 0, 1]
    return x, y, covs


if __name__ == "__main__":

    # get the data table
    data = get_merged_table()
    colnames = data.colnames

    parser = initialize_parser()
    parser.add_argument(
        "--xparam",
        action="store",
        default="AV",
        choices=colnames,
        help="Choose column type to plot",
    )
    parser.add_argument(
        "--yparam",
        action="store",
        default="lognhtot",
        choices=colnames,
        help="Choose column type to plot",
    )
    parser.add_argument(
        "--xrange", action="store", nargs=2, type=float, help="plot x range"
    )
    parser.add_argument(
        "--yrange", action="store", nargs=2, type=float, help="plot y range"
    )
    parser.add_argument(
        "--comps", action="store_true", help="plot the comparision sightlines"
    )
    parser.add_argument(
        "--bohlin", action="store_true", help="plot the Bohlin78 sightlines"
    )
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    # get extra data if desired
    if args.comps:
        data_comp = get_merged_table(comp=True)
    else:
        data_comp = None
    if args.bohlin:
        data_bohlin78 = get_bohlin78()
    else:
        data_bohlin78 = None

    # make the requested plot
    fig = plot_results2(
        data,
        args.xparam,
        args.yparam,
        pxrange=args.xrange,
        pyrange=args.yrange,
        data_comp=data_comp,
        data_bohlin=data_bohlin78,
        figsize=(10, 8),
    )

    fig.tight_layout()

    # save the plot
    basename = "fuse_results_" + args.xparam + "_" + args.yparam
    if args.pdf:
        fig.savefig(basename + ".pdf")
    else:
        plt.show()
