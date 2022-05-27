"""I wanted to make a plot like GCC09, but with fewer curves. It should
show that there are curves with similar RV, that have rather large
differences in the UV. Dividing by AV is not working out well though...
Not sure what the problem is.

"""
from get_data import get_merged_table
from dust_extinction.shapes import FM90
from matplotlib import pyplot as plt
from astropy import units as u
import numpy as np

# f = "/Users/dvandeputte/Projects/FUSE H2/FUSE_Data/hd027778_fuse.fits"
# these don't have the extinction curve in them, only the flux data
# so I guess we'll have to use the fits unless I ask karl for the data

t = get_merged_table()

# a couple of low RV stars
left_names = ["HD014434", "HD148422", "HD099872"]
right_names = ["HD047129", "HD093827", "HD163522"]

x = np.linspace(3.8, 8.6, 100) / u.micron

def plot_by_name(ax, name, **plot_kwargs):
    row = t[t["Name"] == name][0]
    model = FM90(
        C1=row["CAV1"],
        C2=row["CAV2"],
        C3=row["CAV3"],
        C4=row["CAV4"],
        xo=row["x_o"],
        gamma=row["gamma"],
    )
    curve = model(x) / model(1000 * u.angstrom)
    ax.plot(x, curve, **plot_kwargs)


fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
for name in left_names:
    plot_by_name(ax[0], name)
for name in right_names:
    plot_by_name(ax[1], name)
plt.show()
