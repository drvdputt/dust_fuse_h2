#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from astropy.modeling import models, fitting
from get_data import get_merged_table, get_bohlin78, get_param_and_unc, get_xs_ys_covs
import covariance
import linear_ortho_fit
import pearson

# quick workaround to hide sigma number for talk plots 
HIDE_SIGMA = False

# some easily customizable constants
BOHLIN_COLOR = "xkcd:gray"
BOHLIN_MARKER = "D"
SHULL_COLOR = "xkcd:dark yellow"
SHILL_MARKER = "o"
COMP_COLOR = "xkcd:gray"
MAIN_COLOR = "xkcd:bright blue"
MAIN_MARKER = "o"
MAIN_MARKER_MEW = 0.6  # linewidth for scatter command
MAIN_MARKER_SIZE = 20
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
    cmcol = " [cm$^{-2}$]"
    mag = " [mag]"
    col_mag = " [cm$^{-2}$ mag$^{-1}$]"

    dic_pairs = {
        "AV": "$A(V)$" + mag,
        "RV": "$R(V)$",
        "EBV": "$E(B-V)$" + mag,
        "CAV1": "$C^{A(V)}_1$",
        "CAV2": "$C^{A(V)}_2$",
        "CAV3": "$C^{A(V)}_3$",
        "CAV4": "$C^{A(V)}_4$",
        "C1": "$C_1$",
        "C2": "$C_2$",
        "C3": "$C_3$",
        "C4": "$C_4$",
        "x_o": "$x_o$",
        "gamma": r"$\gamma$",
        "bump_area": r"$\pi C^{A(V)}_3 / 2 \gamma$",
        "bump_amp": r"$C^{A(V)}_3 / 2 \gamma^2$",
        "fh2": "$f(\mathrm{H}_2)$",
        "nhtot": "$N(\mathrm{H})$" + cmcol,
        "nh2": "$N(\mathrm{H}_2)$" + cmcol,
        "nhi": "$N(HI)$" + cmcol,
        "NH_AV": "$N(\mathrm{H})/A(V)$" + col_mag,
        "NH_EBV": "$N(\mathrm{H})/E(B-V)$" + col_mag,
        "1_RV": "$1/R(V)$",
        "A1000": "$A(1000)$" + mag,
        "A1000_AV": "$A(1000)/A(V)$",
        "A2175": "$A(2175)$" + mag,
        "A2175_AV": "$A(2175)/A(V)$",
        "T01": "$T_{01}$ [K]",
        "denhtot": "$n(\mathrm{H})$ [cm$^{-3}$]",
    }

    out_name = name
    if name[:3] == "log":
        out_name = r"$\log (" + name[3:].upper() + ")$"
    elif name in dic_pairs.keys():
        out_name = dic_pairs[name]

    return out_name


def plot_rho_box(ax, xs, ys, covs, method='nocov', optional_plot_fname=None):
    """Draw box with correlation coefficient for given data on given ax.

    Method needs to be chosen. See pearson.py for explanation.

    Parameters
    ----------
    ax: axes where a box with rho and a number of sigma will be put

    xs, ys, covs: the data

    method: str
        "nocov" or "cov approx". Nocov is only applicable if the
        covariances are small. Cov approx tries model the offset of
        rho_null caused by the covariance.

    optional_plot_fname: str
        file name for plot of histogram, illustrating the method (if supported)
    """
    if method == 'nocov':
        rho, srho = pearson.pearson_mc_nocov(xs, ys, covs)
    else: # method == 'cov approx'
        rho, srho = pearson.new_rho_method(xs, ys, covs)

    # choose best place to put it
    if rho > 0:
        ha = "left"
        xpos = 0.03
    else:
        ha = "right"
        xpos = 0.98

    text = f"$r = {rho:.2f}$"
    if not HIDE_SIGMA:
        text += f"\n${np.abs(srho):.1f}\\sigma$"

    ax.text(
        xpos,
        0.96,
        text,
        transform=ax.transAxes,
        horizontalalignment=ha,
        # bbox=dict(facecolor="white", edgecolor='none', alpha=0.5),
        verticalalignment="top",
    )


def match_comments(data, comments):
    """
    Return mask which is True where data["comment"] equals any of the given comments.

    Parameters
    ----------

    data : astropy Table
        data set with "comment" column

    comments : list of str
        one string per comment

    Returns
    -------
    match : np.array of booleans
        True at index i if data["comment"][i] == c for any c in comments
    """
    if comments is None:
        match = np.full(len(data), False)
    else:
        match = np.logical_or.reduce([c == data["comment"] for c in comments])
    return match


def plot_results_scatter(
    ax,
    data,
    xparam,
    yparam,
    pxrange=None,
    pyrange=None,
    data_comp=None,
    data_bohlin=None,
    data_shull=None,
    figsize=None,
    ignore_comments=None,
    mark_comments=None,
):
    """Do only the scatter plot of plot_results2, not the fit

    Returns
    -------
    xs, ys, covs: 1D array, 1D array, 3D array with shape (len(data), 2, 2)
        Main data to be used for fitting

    """

    print("-" * 72)
    print(f"{xparam} vs {yparam}")
    print("-" * 72)

    # plot bohlin or shull data (not used for fitting)
    def plot_extra_data(extra, find_dups=False, **kwargs):
        kwargs["markersize"] = 2
        kwargs["markerfacecolor"] = "white"
        kwargs["elinewidth"] = 1
        if extra is not None and xparam in extra.colnames and yparam in extra.colnames:
            # errorbar() has no problems with xerr and yerr being None,
            # so don't need to check for uncertainty columns
            xcol, xcol_unc = get_param_and_unc(xparam, extra)
            ycol, ycol_unc = get_param_and_unc(yparam, extra)
            # avoid duplicates (some stars in shull are also in our sample)
            if find_dups:
                dup = np.isin(extra["Name"], data["Name"])
                not_dup = np.logical_not(dup)
                # plot duplicates as "+"
                # if dup.any():
                #     ax.scatter(
                #         xcol[dup],
                #         ycol[dup],
                #         alpha=alpha ** kwargs,
                #     )
                # discard duplicates
                xcol = xcol[not_dup]
                ycol = ycol[not_dup]
                xcol_unc = None if xcol_unc is None else xcol_unc[not_dup]
                ycol_unc = None if ycol_unc is None else ycol_unc[not_dup]
                print(f"{len(xcol)} stars remaining after removing duplicates")

            ax.errorbar(
                xcol,
                ycol,
                xerr=xcol_unc,
                yerr=ycol_unc,
                linestyle="none",
                # alpha=alpha,
                **kwargs,
            )

    plot_extra_data(
        data_bohlin,
        label="Bohlin78",
        color=BOHLIN_COLOR,
        marker=BOHLIN_MARKER,
        zorder=-10,
    )
    plot_extra_data(
        data_shull, find_dups=True, label="Shull+21", color=SHULL_COLOR, marker="."
    )

    # plot comparison star data (not used for fitting)
    if data_comp is not None:
        # xs, ys, covs = get_xs_ys_covs(data_comp, xparam, yparam, "AV")
        xs, ys, covs = get_xs_ys_covs(data_comp, xparam, yparam)
        covariance.plot_scatter_auto(
            ax, xs, ys, covs, 1, color=COMP_COLOR, label="comp"
        )

    # decide which points to highlight as ignored
    ignore = match_comments(data, ignore_comments)
    not_ignored = np.logical_not(ignore)
    # decide which points to highlight with generic mark
    mark = match_comments(data, mark_comments)

    # get all data, and plot everything except ignored
    xs, ys, covs = get_xs_ys_covs(data, xparam, yparam)
    covariance.plot_scatter_auto(
        ax,
        xs[not_ignored],
        ys[not_ignored],
        covs[not_ignored],
        1,
        color=MAIN_COLOR,
        marker=MAIN_MARKER,
        linewidth=MAIN_MARKER_MEW,
        s=MAIN_MARKER_SIZE,
        label="sample",
        zorder=10,
    )

    # mark data points, if any
    if mark.any():
        xs_mark, ys_mark, _ = get_xs_ys_covs(data[mark], xparam, yparam)
        ax.scatter(
            xs_mark,
            ys_mark,
            facecolors="none",
            edgecolors=MARK_COLOR,
            marker=MARK_MARKER,
            # label="highlight"
            zorder=11,
        )

    # plot ignored points in different color
    if ignore.any():
        bad_xs, bad_ys, bad_covs = get_xs_ys_covs(data[ignore], xparam, yparam)
        covariance.plot_scatter_auto(
            ax,
            bad_xs,
            bad_ys,
            bad_covs,
            1,
            color=BAD_COLOR,
            marker="x",
            label="ignore",
        )

    ax.set_xlabel(format_colname(xparam))
    ax.set_ylabel(format_colname(yparam))
    if pxrange is not None:
        ax.set_xlim(pxrange)
    if pyrange is not None:
        ax.set_ylim(pyrange)

    # return all data, except ignored
    return xs[not_ignored], ys[not_ignored], covs[not_ignored]


def plot_results_fit(
    xs,
    ys,
    covs,
    line_ax,
    lh_ax=None,
    outliers=None,
    auto_outliers=False,
    fit_includes_outliers=False,
):
    """Do the fit and plot the result.

    Parameters
    ----------
    sc_ax : axes to plot the best fit line

    lh_ax : axes to plot the likelihood function

    xs, ys, covs: the data to use (see return value of plot_results_scatter)

    outliers : list of int
        list of indices for which data will be ignored in the fitting.
        If auto_outliers is True, then this data will only be ignored
        for the first iteration. The manual outlier choice positions the
        fit where were we want it. Then, these points are added back in,
        and ideally, the automatic outlier rejection will reject them in
        an objective way. This is to make sure that we are not guilty of
        cherry picking.

    auto_outliers : bool
        Use auto outlier detection in linear_ortho_maxlh, and mark
        outliers on plot (line ax). See outlier detection function for
        criterion.

    fit_includes_outliers : bool
        Use the detected outliers in the fitting, despite them being outliers.

    Returns
    -------
    outlier_idxs : array of int
        Indices of points treated as outliers
    """
    # fix ranges before plotting the fit
    line_ax.set_xlim(line_ax.get_xlim())
    line_ax.set_ylim(line_ax.get_ylim())

    r = linear_ortho_fit.linear_ortho_maxlh(
        xs,
        ys,
        covs,
        line_ax,
        sigma_hess=True,
        manual_outliers=outliers,
        auto_outliers=auto_outliers,
        fit_includes_outliers=fit_includes_outliers,
    )
    m = r["m"]
    b_perp = r["b_perp"]
    sm = r["m_unc"]
    sb_perp = r["b_perp_unc"]
    outlier_idxs = r["outlier_idxs"]

    b = linear_ortho_fit.b_perp_to_b(m, b_perp)

    # The fitting process also indicated some outliers. Do the rest without them.
    if fit_includes_outliers:
        xs_used = xs
        ys_used = ys
        covs_used = covs
    else:
        xs_used = np.delete(xs, outlier_idxs, axis=0)
        ys_used = np.delete(ys, outlier_idxs, axis=0)
        covs_used = np.delete(covs, outlier_idxs, axis=0)

    # Looking at bootstrap with and without outliers might be interesting.
    # boot_cov_mb = linear_ortho_fit.bootstrap_fit_errors(xs_no_out, ys_no_out, covs_no_out)
    # boot_sm, boot_sb = np.sqrt(np.diag(boot_cov_mb))

    # sample the likelihood function to determine statistical properties
    # of m and b
    a = 2
    m_grid, b_perp_grid, logL_grid = linear_ortho_fit.calc_logL_grid(
        m - a * sm,
        m + a * sm,
        b_perp - a * sb_perp,
        b_perp + a * sb_perp,
        xs_used,
        ys_used,
        covs_used,
    )

    # Sample the likelihood of (m, b_perp) and convert to (m, b), so we
    # can properly determine the covariance.
    sampled_m, sampled_b_perp = linear_ortho_fit.sample_likelihood(
        m, b_perp, m_grid, b_perp_grid, logL_grid, N=2000
    )
    sampled_b = linear_ortho_fit.b_perp_to_b(sampled_m, sampled_b_perp)

    sample_cov_mb = np.cov(sampled_m, sampled_b)
    m_unc = np.sqrt(sample_cov_mb[0, 0])
    b_unc = np.sqrt(sample_cov_mb[1, 1])
    mb_corr = sample_cov_mb[0, 1] / (m_unc * b_unc)

    # print out results here
    print("*** FIT RESULT ***")
    print(f"m = {m:.2e} pm {m_unc:.2e}")
    print(f"b = {b:.2e} pm {b_unc:.2e}")
    print(f"correlation = {mb_corr:.2f}")
    if lh_ax is not None:
        linear_ortho_fit.plot_solution_neighborhood(
            lh_ax,
            logL_grid,
            [min(b_perp_grid), max(b_perp_grid), min(m_grid), max(m_grid)],
            m,
            b_perp,
            cov_mb=sample_cov_mb,
            what="L",
            extra_points=zip(sampled_b_perp, sampled_m),
        )

    # plot the fitted line
    xlim = line_ax.get_xlim()
    xp = np.linspace(xlim[0], xlim[1], 3)
    yp = m * xp + b
    line_ax.plot(xp, yp, color=FIT_COLOR, linewidth=1)

    # plot sampled lines
    linear_ortho_fit.plot_solution_linescatter(
        line_ax, sampled_m, sampled_b_perp, color=FIT_COLOR, alpha=4 / 255, zorder=-10
    )

    # if outliers, mark them
    if len(outlier_idxs) > 0:
        line_ax.scatter(
            xs[outlier_idxs],
            ys[outlier_idxs],
            marker="x",
            color="r",
            label="outlier",
            zorder=10,
        )

    # return as dict, in case we want to do more specific things in
    # post. Example: gathering numbers and putting them into a table, in
    # the main plotting script (paper_scatter.py).
    # Also return covariance and samples, useful for determining error on y = mx + b.
    results = {
        "m": m,
        "m_unc": m_unc,
        "b": b,
        "b_unc": b_unc,
        "mb_cov": sample_cov_mb[0, 1],
        "outlier_idxs": outlier_idxs,
        "m_samples": sampled_m,
        "b_samples": sampled_b,
    }
    return results


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
    """Plot the fuse results with specificed x and y axes and do the linear fit.

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
    fig, (
        ax,
        lh_ax,
    ) = plt.subplots(figsize=(8, 5), ncols=2)

    xs, ys, covs = plot_results_scatter(
        ax,
        data,
        xparam,
        yparam,
        pxrange,
        pyrange,
        data_comp,
        data_bohlin,
        data_shull,
        figsize,
        alpha,
        ignore_comments,
        mark_comments,
    )
    out = np.where(match_comments(data, ignore_comments))[0]
    _ = plot_results_fit(
        xs, ys, covs, ax, lh_ax, outliers=out, auto_outliers=True
    )
    # print("Outliers: ", data["Name"][auto_out])
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
