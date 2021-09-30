from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
import itertools
from get_data import get_merged_table
from astropy.table import Column
from plot_fuse_results import plot_results_scatter

plt.rcParams.update({'font.size': 10})

data = get_merged_table()
# add comments for certain stars here
data.add_column(Column(["no"] * len(data), dtype="<U16", name="comment"))


def set_comment(name, s):
    data["comment"][data["Name"] == name] = s


for name in ["HD200775", "HD164906", "HD045314", "HD206773"]:
    set_comment(name, "lo_h_av")

set_comment("HD096675", "hi_h_av")

qnames = [
    "T01",
    "nhtot",
    "denhtot",
    "nhi",
    # "denhi",
    "nh2",
    # "denh2",
    "fh2",
    "NH_AV",
    # "AV_d",
    "EBV",
    "A1000",
    "A2175",
    "AV",
    # "A1000_d",
    "A1000_AV",
    "A2175_AV",
    "1_RV",
    # "CAV3",
    "CAV4",
    # "gamma",
    # "x0",
    "bump_amp",
    # "C3",
    # "C4",
    # 'd'
]
nq = len(qnames)

fig = plt.figure(figsize=(10, 10))
gs = GridSpec(nq, nq)

for yi, xi in itertools.combinations_with_replacement(range(nq), 2):
    if xi == yi:
        continue

    xpos = nq - 1 - xi
    ypos = nq - 1 - yi
    ax = plt.subplot(gs[ypos, xpos])

    xparam = qnames[xi]
    yparam = qnames[yi]
    plot_results_scatter(ax, data, xparam, yparam, alpha=1, mark_comments=["lo_h_av"])

    if ypos < nq - 1:
        ax.set_xticklabels([])
        ax.set_xlabel("")

    if xpos > 0:
        ax.set_yticklabels([])
        ax.set_ylabel("")


# plt.tight_layout(w_pad=0, h_pad=0)
plt.subplots_adjust(wspace=0, hspace=0, right=1, top=1, left=0.05, bottom=0.05)
plt.show()
plt.set_size_inches(17, 17)
plt.savefig("bigcorner.pdf")
