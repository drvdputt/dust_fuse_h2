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

our_names = data['Name'][data_overlap]
nhav = data['NH_AV'][data_overlap]
nhav_unc = data['NH_AV_unc'][data_overlap]

jenkins_names = tjenkins_F['Name'][tjenkins_overlap]
fstar = tjenkins_F['F*o'][tjenkins_overlap]
fstar_unc = tjenkins_F['e_F*o'][tjenkins_overlap]

#the above matches by name

# now, another cut because some of the F* values are masked
keep = ~fstar.mask
nhav = nhav[keep]
nhav_unc = nhav_unc[keep]
# also some workaround for a numpy problem here (complains about this
# being a masked array but also says does not have mask)
fstar = fstar[keep].data.data
fstar_unc = fstar_unc[keep].data.data

covs = make_cov_matrix(fstar_unc**2, nhav_unc**2)
plot_scatter_auto(plt.gca(), fstar, nhav, covs, 1)
plot_rho_box(plt.gca(), fstar, nhav, covs)
ax = plt.gca()
ax.set_ylabel(format_colname('NH_AV'))
ax.set_xlabel("$F_*$")
plt.show()
