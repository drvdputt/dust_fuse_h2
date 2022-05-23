import numpy as np
from astropy.table import Table
from covariance import make_cov_matrix, plot_scatter_auto
from plot_fuse_results import plot_rho_box, format_colname
from get_data import get_merged_table
from matplotlib import pyplot as plt

# tjenkins = Table.read('data/jenkins2009-table2.dat', format='ascii.cds')
# to get fstar, we need to load another table
tjenkins_F = Table.read("data/jenkins2009-table5.dat", format='ascii.cds')
data = get_merged_table()

our_hd_numbers = [int(data['Name'][i][2:]) for i in range(len(data)) if 'HD' in data['Name'][i]]
jenkins_hd_numbers = tjenkins_F['HD'][~tjenkins_F['HD'].mask]

hd_set = set(our_hd_numbers) & set(jenkins_hd_numbers[~jenkins_hd_numbers.mask])

print(f"The following {len(hd_set)} HD stars overlap")
print(hd_set)
print(f"And also these two that I found manually: BD+35d4258 and BD+53d2820")

# this does not work without converting to list first, for some reason
tjenkins_overlap = np.isin(tjenkins_F['HD'], list(hd_set))
# our data overlap based on number
data_overlap = [name[:2] == 'HD' and int(name[2:]) in hd_set for name in data['Name']]

jenkins_names = tjenkins_F['Name'][tjenkins_overlap]
our_names = data['Name'][data_overlap]
#the above matches by name

# now, another cut because some of the F* values are masked. Also some
# workaround for a numpy problem here (complains about this being a
# masked array but also says does not have mask)
fstar = tjenkins_F['F*o'][tjenkins_overlap]
fstar_unc = tjenkins_F['e_F*o'][tjenkins_overlap]
keep = ~fstar.mask
fstar = fstar[keep].data.data
fstar_unc = fstar_unc[keep].data.data

def plot_vs_fstar(ax, yparam):
    ys = data[yparam][data_overlap][keep]
    ys_unc = data[yparam + '_unc'][data_overlap][keep]
    covs = make_cov_matrix(fstar_unc**2, ys_unc**2)
    plot_scatter_auto(ax, fstar, ys, covs, 1)
    plot_rho_box(ax, fstar, ys, covs)
    ax.set_ylabel(format_colname(yparam))
    ax.set_xlabel("$F_*$")


fig, axs = plt.subplots(4, 1, sharex=True)
plot_vs_fstar(axs[0], 'CAV3')
plot_vs_fstar(axs[1], 'CAV4')
plot_vs_fstar(axs[2], 'gamma')
plot_vs_fstar(axs[3], 'A1000_NH')
fig.set_size_inches(3, 9)
plt.show()
