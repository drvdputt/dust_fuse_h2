"""Pearson correlation coefficient for data that have cov(x,y) for each point.

A Monte Carlo method is used to generate a set of possible pearson
coefficients. Maybe it's possible to do this analytically too, but I'll
start with Monte Carlo just to be sure.

This coefficient should be a bit more meaningful than just a slope and
an uncertainty, because is tells you something about the DATA directly,
instead of some best fitting model which has underlying assumptions.

"""
import numpy as np
from matplotlib import pyplot as plt
import rescale
from pathlib import Path
from covariance import cov_common_denominator

RNG = np.random.default_rng(4321)


def draw_points(xs, ys, covs, M):
    """
    Resample a set of points M times, adding noise according to their covariance matrices.

    Returns
    -------
    x_samples, y_samples : np.array
        Every column j, is x[j] redrawn M times.
        Has M rows, and every row is a realization of xs or ys.
    """
    # store the samples as follows
    # col0 = all resamplings of x0
    # -> each row is a different realization of our 75 sightlines

    # rescale data to avoid numerical problems
    rd = rescale.RescaledData(xs, ys, covs)
    N = len(rd.xy)
    x_samples = np.zeros((M, N))
    y_samples = np.zeros((M, N))
    for j in range(N):
        samples = RNG.multivariate_normal(mean=rd.xy[j], cov=rd.covs[j], size=M)
        x_samples[:, j] = samples[:, 0]
        y_samples[:, j] = samples[:, 1]

    # unscale the data again before returning
    return x_samples / rd.factor_x, y_samples / rd.factor_y


def pearson_mc(xs, ys, covs, save_hist=None, hist_ax=None):
    """Calculate Pearson correlation coefficient and uncertainty on it in a MC way.

    Repeatedly resample xs, ys using 2D gaussian described by covs.

    KNOWN DEFICIENCY: does not work with correlated uncertainties.
    Alternate methods to generate a null hypothesis sample, and simulate
    the shift from r=0 due to measurement correlations, will be
    attempted below.

    This function only does a good job of estimating the r-distribution
    under the null hypothesis that the physical rho=0, when the
    uncertainties are uncorrelated. Any correlations induced by the
    covariance, are removed again because of the scrambling.

    Parameters
    ----------

    save_hist : string
        File name for figure of histogram for rho and rho0.

    hist_ax : figure, axes
        Plot histogram on this axes. If None, a new figure will be made.

    Returns
    -------
    rho : correlation coefficient

    std : standard deviation of the rho samples
    """
    M = 6000  # number of resamples
    # scramble test to create null hypothesis distribution of rho.
    # Technically, only the y samples need to be scrambled, but I'm
    # going to do both just to be sure.
    x_samples, y_samples = draw_points(xs, ys, covs, M)
    x_samples_scrambled, y_samples_scrambled = draw_points(xs, ys, covs, M)
    # TODO: current way is actually wrong. Works fine for uncorrelated
    # data, but does not do what it needs to do when there are
    # correlations. Need to shift according to noise AFTER scrambling x
    # with respect to y. To be clear: 1. create uncorrelated data 2.
    # induce correlation due to correlated uncertainties 3. see if
    # observed correlation is significantly higher than the typical
    # induced correlation

    # the hard part: need covariance matrix for each point in the
    # scrambled data set. We can use the covariance equations that we
    # used for the original data for this though. See get_xs_ys_covs in
    # get_data.

    # possible way to go would be choosing the "worst case scenario".
    # Pick the biggest correlation coefficient, and then use the same
    # covariance matrix to sample the shifts of all the scrambled pairs.

    # but that won't work, because often the points with the biggest
    # error bars have smaller covariances.

    # other options: order covariance matrices randomly?
    for i in range(M):
        # np.random.shuffle(x_samples_scrambled[i])
        RNG.shuffle(y_samples_scrambled[i])

    corrcoefs_null = np.array(
        [np.corrcoef(x_samples_scrambled[i], y_samples_scrambled[i]) for i in range(M)]
    )
    rhos_null = corrcoefs_null[:, 0, 1]
    # p16_null = np.percentile(rhos_null, 16)
    # p84_null = np.percentile(rhos_null, 84)
    # std_null = (p84_null - p16_null) / 2
    std_null = np.std(rhos_null)

    corrcoefs = np.array([np.corrcoef(x_samples[i], y_samples[i]) for i in range(M)])
    rhos = corrcoefs[:, 0, 1]

    p16 = np.percentile(rhos, 16)
    p84 = np.percentile(rhos, 84)

    std = np.std(rhos)
    # std = (p84 - p16) / 2

    def rho_sigma_message(rho):
        num_sigmas = rho / std_null
        num_sigmas_lo = p16 / std_null
        num_sigmas_hi = p84 / std_null
        return f"rho = {rho:.2f} +- {std:.2f} ({num_sigmas:.2f} sigma0)\n sigmas range = {num_sigmas_lo:.2f} - {num_sigmas_hi:.2f}"

    print("+++ MC pearson result +++")
    rho_naive = np.corrcoef(xs, ys)[0, 1]
    print("raw: ", rho_sigma_message(rho_naive))
    avg = np.average(rhos)
    print("avg: ", rho_sigma_message(avg))
    med = np.median(rhos)
    print("median: ", rho_sigma_message(med))

    outputs = [med, std_null]

    # any of these this implies that we have to plot
    if save_hist is not None or hist_ax is not None:
        # make new fig, ax if none was given
        if hist_ax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = hist_ax.get_figure(), hist_ax

        bins = 64
        ax.hist(
            rhos_null,
            bins=bins,
            label="null",
            color="xkcd:gray",
            alpha=0.5,
        )
        ax.hist(
            rhos,
            bins=bins,
            label="correlated",
            color="xkcd:bright blue",
            alpha=0.5,
        )

        # save hist to file if requested
        if save_hist is not None:
            d = Path(save_hist).parent
            d.mkdir(exist_ok=True)
            fig.savefig("rho_histograms/" + save_hist)

    return outputs

def new_rho_method(ax, xs, ys, covs):
    """Driver for pearson_mock_test and setting up one of the mockers below.
    This is what will go on most plots. Draws a box with rho and num
    sigma on top of the given axes.

    Since we are in the last phases of writing the paper, this function
    is called directly from the paper_scatter script, instead of
    integrating it with the rest of the plotting routines. The mock
    method chosen is just hardcoded here too.

    """
    mocker = RandomCovMock(xs, ys, covs)
    results = pearson_mock_test(mocker)
    # numbers to use for box
    rho = results['real_rho']
    srho = results['as measured']

    # choose best place to put it
    if rho > 0:
        ha = "left"
        xpos = 0.03
    else:
        ha = "right"
        xpos = 0.98

    text = f"$r = {rho:.2f}$"
    # rho_null might be interesting here too
    text += f"\n${np.abs(rho/srho):.1f}\\sigma$"

    ax.text(
        xpos,
        0.96,
        text,
        transform=ax.transAxes,
        horizontalalignment=ha,
        # bbox=dict(facecolor="white", edgecolor='none', alpha=0.5),
        verticalalignment="top",
    )

    
def pearson_mock_test(mocker, plot_hists=False):
    """mocker: subclass of PearsonNullMock, which also contains the data. It
    is recommended to rescale the data to avoid floating point issues.

    Some of the mock methods run into problems with invalid covariance
    matrices sometimes. Most of these (but not all of them) are fixed
    by rescaling the data.

    """
    xs, ys, covs = mocker.xs, mocker.ys, mocker.covs

    N = len(xs)
    M = 2048
    # physical samples for null hypothesis
    mock_x_p = np.zeros((M, N))
    mock_y_p = np.zeros((M, N))
    mock_cov = np.zeros((M, N, 2, 2))
    # shifted due to correlated measurement noise
    mock_y_meas = np.zeros((M, N))
    mock_x_meas = np.zeros((M, N))
    # also make shifted version of real data data
    data_x_shift = np.zeros((M, N))
    data_y_shift = np.zeros((M, N))

    for m in range(M):
        mock_x_p[m], mock_y_p[m], mock_cov[m] = mocker.mock()
        mock_x_meas[m], mock_y_meas[m] = shift_data(
            mock_x_p[m], mock_y_p[m], mock_cov[m]
        )
        mock_cov_too_big = np.abs(mock_cov[m, :, 0, 1]) > np.sqrt(
            mock_cov[m, :, 0, 0] * mock_cov[m, :, 1, 1]
        )
        if mock_cov_too_big.any():
            # hack this problem
            mock_cov[m, mock_cov_too_big, 0, 1] = (
                np.sign(mock_cov[m, mock_cov_too_big, 0, 1])
                * 0.99
                * np.sqrt(
                    mock_cov[m, mock_cov_too_big, 0, 0]
                    * mock_cov[m, mock_cov_too_big, 1, 1]
                )
            )
            mock_cov[m, mock_cov_too_big, 1, 0] = mock_cov[m, mock_cov_too_big, 0, 1]
            pass
            # print("problem with mock cov for indices", np.where(mock_cov_too_big))
            # print(mock_cov[m, mock_cov_too_big],
            #       mock_cov[m, mock_cov_too_big][:, 0,1]
            #       / np.sqrt(mock_cov[m, mock_cov_too_big][:, 0,0] * mock_cov[m, mock_cov_too_big][:,1,1]))

        data_x_shift[m], data_y_shift[m] = shift_data(xs, ys, covs)

    physical_nullrhos = all_rhos(mock_x_p, mock_y_p)
    measured_nullrhos = all_rhos(mock_x_meas, mock_y_meas)
    data_rhos = all_rhos(data_x_shift, data_y_shift)
    real_rho = np.corrcoef(xs, ys)[0, 1]

    if plot_hists:
        bins = np.linspace(-1, 1, 128)
        plt.hist(physical_nullrhos, bins=bins, label="physical null", alpha=1)
        plt.hist(measured_nullrhos, bins=bins, label="measured null", alpha=0.5)
        plt.hist(data_rhos, bins=bins, label="data wiggle", alpha=0.25)
        plt.legend()
        plt.axvline(real_rho, label="measured data", color="k")

    # num_sigma_to_null_physical = (real_rho - np.median(physical_rhos)) / np.std(physical_rhos)
    num_sigma_to_null_measured = (real_rho - np.median(measured_nullrhos)) / np.std(
        measured_nullrhos
    )
    num_sigma_to_null_wiggled = (
        np.median(data_rhos) - np.median(measured_nullrhos)
    ) / np.std(measured_nullrhos)
    return {
        "real_rho": real_rho,
        "null_rho": np.median(measured_nullrhos),
        "as measured": num_sigma_to_null_measured,
        "median of wiggle": num_sigma_to_null_wiggled,
    }


class PearsonNullMock:
    """Class providing different methods to generates data under the null
    hypothesis."""

    def __init__(self, xs, ys, covs):
        self.xs = xs
        self.ys = ys
        self.covs = covs
        self.N = len(xs)
        self.xs_unc = np.sqrt(covs[:, 0, 0])
        self.ys_unc = np.sqrt(covs[:, 1, 1])
        self.xs_rel_unc = self.xs_unc / xs
        self.ys_rel_unc = self.ys_unc / ys


class CommonDenominatorMock(PearsonNullMock):
    def __init__(self, xs, ys, covs, a, sa):
        super().__init__(xs, ys, covs)
        self.a = a
        self.sa = sa

    def mock(self):
        """
        An attempt to estimate the correlation significance of two variables
        x/a and y/a, with correlated errors due to a.
        """
        a_rel_unc = self.sa / self.a

        # create uncorrelated sample by scrambling
        y_order = random_order(self.N)
        ys_scr = self.ys[y_order]

        # assign AV randomly.
        a_order = random_order(self.N)

        # remove rel unc of original av, add rel unc of newly assigned av
        xs_unc_adj = self.xs * (self.xs_rel_unc - a_rel_unc + a_rel_unc[a_order])
        # remove rel unc of original av (matching to the original y point)
        ys_unc_scr_adj = ys_scr * (
            self.ys_rel_unc[y_order] - a_rel_unc[y_order] + a_rel_unc[a_order]
        )

        # calculate covs given this parameter set
        covs_scr = cov_common_denominator(
            self.xs,
            xs_unc_adj,
            ys_scr,
            ys_unc_scr_adj,
            self.a[a_order],
            self.sa[a_order],
        )
        return self.xs, ys_scr, covs_scr


class RandomCovMock(PearsonNullMock):
    def mock(self):
        y_order = random_order(self.N)
        covs_order = random_order(self.N)
        return self.xs, self.ys[y_order], self.covs[covs_order]


def all_rhos(all_xs, all_ys):
    """Parameters indexed on m, i (number of samples, data point)"""
    return [np.corrcoef(all_xs[m], all_ys[m])[0, 1] for m in range(len(all_xs))]


def random_order(size):
    order = np.arange(size)
    np.random.shuffle(order)
    return order


def shift_data(xs, ys, covs, plot=False):
    shifted = np.array(
        [np.random.multivariate_normal((xs[i], ys[i]), covs[i]) for i in range(len(xs))]
    )
    xs_shift = shifted[:, 0]
    ys_shift = shifted[:, 1]
    return xs_shift, ys_shift
