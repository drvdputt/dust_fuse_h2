#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from astropy.modeling import models, fitting
from get_data import get_merged_table, get_bohlin78
from covariance import plot_scatter_with_ellipses
import linear_ortho_fit

# some easily customizable constants
BOHLIN_COLOR = "green"
COMP_COLOR = "purple"
SAMPLE_COLOR = "grey"
BAD_COLOR = "red"


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
    figsize=None,
    alpha=0.5,
    ignore_comments=None,
):
    """
    Plot the fuse results with specificed x and y axes

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

    figsize : float[2]
       x,y size of plot

    ignore_comments : list of str
       exclude points for which data['comment'] equals one of the given
       strings from fitting (they will still be plotted in a highlighted
       color)
    """
    set_params(lw=1, universal_color="#202026", fontsize=10)

    # fig, ax = plt.subplots(figsize=figsize)
    fig, (ax, ax2, ax3) = plt.subplots(figsize=(12, 5), ncols=3)

    # plot bohlin data (not used for fitting)
    if data_bohlin is not None:
        if (xparam in data_bohlin.colnames) and (yparam in data_bohlin.colnames):
            xcol = data_bohlin[xparam]
            xcol_unc = get_unc(xparam, data_bohlin)
            ycol = data_bohlin[yparam]
            ycol_unc = get_unc(yparam, data_bohlin)
            ax.errorbar(
                xcol,
                ycol,
                xerr=xcol_unc,
                yerr=ycol_unc,
                label="Bohlin (1978)",
                color=BOHLIN_COLOR,
                linestyle="none",
                marker=".",
                alpha=alpha,
            )

    # plot comparison star data (not used for fitting)
    if data_comp is not None:
        xs, ys, covs = get_xs_ys_covs(data_comp, xparam, yparam, "AV")
        plot_scatter_with_ellipses(ax, xs, ys, covs, 1, color=COMP_COLOR, alpha=alpha)

    # decide which points to ignore
    if ignore_comments is not None:
        # only use points for which the comment does not match any of
        # the ignore flags
        use = np.logical_and.reduce([c != data["comment"] for c in ignore_comments])
    else:
        use = np.full(len(data), True)

    # choose columns and calculate covariance matrices
    if yparam[0:3] == "CAV":
        cparam = "AV"
    elif yparam[0:1] == "C":
        cparam = "EBV"
    else:
        cparam = "AV"
    xs, ys, covs = get_xs_ys_covs(data[use], xparam, yparam, cparam)
    plot_scatter_with_ellipses(
        ax, xs, ys, covs, 1, color=SAMPLE_COLOR, alpha=alpha, marker="x"
    )

    # plot ignored points in different color
    if not use.all():
        bad_xs, bad_ys, bad_covs = get_xs_ys_covs(
            data[np.logical_not(use)], xparam, yparam, cparam
        )
        plot_scatter_with_ellipses(
            ax, bad_xs, bad_ys, bad_covs, 1, color=BAD_COLOR, alpha=alpha, marker="x"
        )

    ax.set_xlabel(format_colname(xparam))
    ax.set_ylabel(format_colname(yparam))

    if pxrange is not None:
        ax.set_xlim(pxrange)
    if pyrange is not None:
        ax.set_ylim(pyrange)

    m, b = linear_ortho_fit.linear_ortho_maxlh(xs, ys, covs, ax)
    cov_mb = linear_ortho_fit.bootstrap_fit_errors(xs, ys, covs)
    sm, sb = np.sqrt(np.diag(cov_mb))
    a = 2
    area = [m - a * sm, m + a * sm, b - a * sb, b + a * sb]
    linear_ortho_fit.plot_solution_neighborhood(
        ax2, m, b, xs, ys, covs, cov_mb=cov_mb, area=area, what="logL"
    )
    linear_ortho_fit.plot_solution_neighborhood(
        ax3, m, b, xs, ys, covs, cov_mb=cov_mb, area=area, what="L"
    )
    ax3.set_ylabel("")

    # plot the fitted line
    xlim = ax.get_xlim()
    xp = np.linspace(xlim[0], xlim[1], 3)
    yp = m * xp + b * np.sqrt(1 + m * m)
    ax.plot(xp, yp, color="k")

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


def get_unc(param, data):
    """
    Returns the unc column if it is in the table
    """
    if param + "_unc" in data.colnames:
        return data[param + "_unc"].data
    else:
        return None


def get_corr(xparam, yparam, x, y, xerr, yerr, cterm=None, cterm_unc=None):
    """
    Return the correlation coefficient between pairs of parameters
    """
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

    return corr


def get_xs_ys_covs(data, xparam, yparam, cparam):
    """
    Return arrays of x, y, and cov(x,y) for a pair of parameters

    """
    x = data[xparam]
    xerr = get_unc(xparam, data)
    y = data[yparam]
    yerr = get_unc(yparam, data)
    cterm = data[cparam]
    cterm_unc = get_unc(cparam, data)

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
