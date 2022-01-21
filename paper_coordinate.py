"""Quantities vs position on sky"""

import get_data
from matplotlib import pyplot as plt
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
from math import floor, ceil

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


def ideal_grid(numplots):
    root = np.sqrt(numplots)
    square_small = int(floor(root))
    square_big = int(ceil(root))

    diff_small = square_small ** 2 - numplots
    diff_big = square_big ** 2 - numplots

    if diff_small == 0:
        return square_small, square_small
    elif diff_big == 0:
        return square_big, square_big
    # else, take the best fitting square, and make it wider by reducing the height or increasing the width
    if abs(diff_small) < abs(diff_big):
        y = square_small
        x = int(ceil(numplots / y))  # make x wider
    else:
        x = square_big
        y = int(ceil(numplots / x))  # make y shorter

    return x, y


# let's just look at b first
def q_vs_one_coord(ax, qname, coord_values):
    """ """
    q = data[qname]
    ax.scatter(coord_values, q)
    ax.set_ylabel(qname)


def plot_one_coord(coord_name):
    """
    coord_name: 'b' or 'l'
    """
    qs = ["1_RV", "A1000_AV", "AV_NH", "A1000_NH", "nh2", "fh2"]
    fig, axs = plt.subplots(len(qs), 1, sharex=True, figsize=(5, 2 * len(qs)))
    for qname, ax in zip(qs, axs):
        q_vs_one_coord(ax, qname, b if coord_name == "b" else l)

    axs[-1].set_xlabel(f"{coord_name} (degrees)")


def q_vs_two_coords(ax, qname):
    q = data[qname]
    x = l.wrap_at(180 * u.deg).to(u.rad)
    y = b.to(u.rad)
    collection = ax.scatter(x, y, c=q)
    cbar = ax.get_figure().colorbar(collection, ax=ax)
    cbar.set_label(qname)
    ax.grid()


def plot_both_coords():
    qs = ["1_RV", "A1000_AV", "AV_NH", "A1000_NH", "nh2", "fh2"]
    ncols, nrows = ideal_grid(len(qs))
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 2 * nrows),
        subplot_kw={"projection": "aitoff"},
        squeeze=False
    )
    for qname, ax in zip(qs, axs.flatten()):
        q_vs_two_coords(ax, qname)


# plot_one_coord('b')
# plot_one_coord('l')

plot_both_coords()

plt.show()
