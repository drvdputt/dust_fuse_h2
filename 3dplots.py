from get_data import get_merged_table
from matplotlib import pyplot as plt
from astropy.table import Column
import numpy as np

data = get_merged_table()
# add comments for certain stars here
data.add_column(Column(["no"] * len(data), dtype="<U16", name="comment"))


def set_comment(name, s):
    data["comment"][data["Name"] == name] = s


for name in ["HD200775", "HD164906", "HD045314", "HD206773"]:
    set_comment(name, "lo_h_av")

set_comment("HD096675", "hi_h_av")

fh2 = data["fh2"]
rvm1 = 1 / data["RV"]
a1000 = data["A1000"]


def scatter_and_color(xparam, yparam, cparam, mask=np.full((len(data),), True)):
    x = data[xparam][mask]
    y = data[yparam][mask]
    c = data[cparam][mask]
    plt.figure()
    plt.scatter(x, y, c=c)
    plt.xlabel(xparam)
    plt.ylabel(yparam)
    cb = plt.colorbar()
    cb.set_label(cparam)


def threedee(xparam, yparam, zparam, cparam=None):
    x = data[xparam]
    y = data[yparam]
    z = data[zparam]
    c = None if cparam is None else data[cparam]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    p = ax.scatter(x, y, z, c=c)
    ax.set_xlabel(xparam)
    ax.set_ylabel(yparam)
    ax.set_zlabel(zparam)
    if cparam is not None:
        cb = fig.colorbar(p)
        cb.set_label(cparam)


# scatter_and_color("1_RV", "A1000", "fh2")
# scatter_and_color('A1000', 'fh2', '1_RV')

# scatter_and_color("1_RV", "fh2", "NH_AV", data["NH_AV"] < 8e21)
# scatter_and_color("1_RV", "fh2", "A1000")

# scatter_and_color("1_RV", "fh2", "CAV4")
# scatter_and_color("A1000", "fh2", "1_RV")
# scatter_and_color("A1000", "fh2", "CAV4")

# scatter_and_color("A1000", "CAV4", "fh2")

# threedee("1_RV", "A1000", "CAV4", "fh2")

scatter_and_color("A2175_NH", "AV_NH", "1_RV")
scatter_and_color("A1000_NH", "AV_NH", "1_RV")
scatter_and_color("1_RV", "A2175_AV", "AV_NH")

plt.show()
