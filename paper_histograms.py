"""Demonstration of rho-histograms"""

import get_data
import pearson
data = get_data.get_merged_table()

def rho_histogram(xparam, yparam, name):
    xs, ys, covs = get_data.get_xs_ys_covs(data, xparam, yparam)
    med, std_null, (fig, ax) = pearson.pearson_mc(xs, ys, covs, return_hist_fig_ax=True)
    fig.savefig(f"./paper-plots/{name}.pdf")

rho_histogram('1_RV', 'NH_AV', 'insignificant')
rho_histogram('A1000_AV', 'NH_AV', 'significant')
