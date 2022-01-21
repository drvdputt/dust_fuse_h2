"""Quantities vs position on sky"""

import get_data
from matplotlib import pyplot as plt
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np

data = get_data.get_merged_table()

# The coordinates of the stars were retrieved by converting the star
# names into simbad format, then querying simbad. See get_gaia.py.
# Normally in the same order as the merged table, should double check
# to make sure.
simbad_coordinates = Table.read(
    "data/simbad_coordinates.dat", format="ascii.commented_header"
)
c = SkyCoord(
    ra=simbad_coordinates["RA"],
    dec=simbad_coordinates["DEC"],
    unit=(u.hourangle, u.deg),
)

# some quantities which could be interesting
# NH / AV
# A1000 / AV
# 1 / RV
# NH2
# NH
# fh2

# convert to galactic coords
cgal = c.galactic
l = cgal.l
b = cgal.b
babs = np.abs(b)

# let's just look at b first
def q_vs_one_coord(ax, qname, coord_values):
    """
    """
    q = data[qname]
    ax.scatter(coord_values, q)
    ax.set_ylabel(qname)


def plot_one_coord(coord_name):
    """
    coord_name: 'b' or 'l'
    """
    qs = ["1_RV", "A1000_AV", "AV_NH", "A1000_NH", "nh2", "fh2"]
    fig, axs = plt.subplots(len(qs), 1, sharex=True)
    for qname, ax in zip(qs, axs):
        q_vs_one_coord(ax, qname, b if coord_name == 'b' else l)
    
    axs[-1].set_xlabel(f"{coord_name} (degrees)")

plot_one_coord('b')
plot_one_coord('l')
plt.show()
