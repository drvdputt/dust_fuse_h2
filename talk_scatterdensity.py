from matplotlib import pyplot as plt
from get_data import get_merged_table, get_xs_ys_covs
from matplotlib import pyplot as plt
from covariance import plot_scatter_density
from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument('xparam')
ap.add_argument('yparam')
args = ap.parse_args()
xparam = args.xparam
yparam = args.yparam
data = get_merged_table()
xs, ys, covs = get_xs_ys_covs(data, xparam, yparam)
fig, ax = plt.subplots()
plot_scatter_density(ax, xs, ys, covs)
plt.show()
