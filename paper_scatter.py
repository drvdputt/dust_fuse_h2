"""A limited selection of scatter plots to show in the paper.

These might have specific additions or omissions per plot, depending on
what's interesting to show. E.g., only fit a line when relevant.

Some of them are grouped using subplots

"""

from get_data import get_merged_table, get_bohlin78, get_shull2021
import plot_fuse_results
from plot_fuse_results import plot_results_scatter, plot_results_fit
from matplotlib import pyplot as plt
from astropy.table import Column

# change colors like this
plot_fuse_results.MAIN_COLOR = "k"

# base width for figures. Is equal to the text width over two columns
base_width = 7.1


def set_comment(name, s):
    """Set the comment for a specific star to the string s."""
    data["comment"][data["Name"] == name] = s


# main data and comments to help marking some points
data = get_merged_table()
comp = get_merged_table(True)
data.add_column(Column(["none"] * len(data), dtype="<U16", name="comment"))

set_comment("HD096675", "hi_h_av")
for name in ["HD200775", "HD164906", "HD045314", "HD206773"]:
    set_comment(name, "lo_h_av")

bohlin = get_bohlin78()
shull = get_shull2021()

def finalize_single(filename):
    pass

def finalize_double(fig, filename):
    fig.set_size_inches(base_width, 2 / 3 * base_width)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.03)
    fig.savefig("paper-plots" + filename)
    

###############################################################################
# PLOT 1 
# the first plot shows nh vs av and nh vs ebv. The main goal is showing
# that there are some outliers in AV, which are not necessarily outliers
# in EBV, and determining the main nh vs av trend in the sample

fig, (ax_av_nh, ax_ebv_nh) = plt.subplots(1, 2, sharey=True)

print("NH vs AV")
xs, ys, covs = plot_results_scatter(
    ax_av_nh,
    data,
    "AV",
    "nhtot",
    data_comp=comp,
    data_bohlin=bohlin,
    ignore_comments=["lo_h_av", "hi_h_av"],
)
plot_results_fit(xs, ys, covs, ax_av_nh)

print("NH vs EBV")
xs, ys, covs = plot_results_scatter(
    ax_ebv_nh,
    data,
    "EBV",
    "nhtot",
    data_comp=comp,
    data_bohlin=bohlin,
    data_shull=shull,
    ignore_comments=["hi_h_av"],
    mark_comments=["lo_h_av"],
)
plot_results_fit(xs, ys, covs, ax_ebv_nh)
ax_ebv_nh.set_ylabel("")

ax_ebv_nh.legend()
finalize_double(fig, "av_ebv_vs_nh.pdf")

###############################################################################

fig, (ax_av_nhav, ax_rvi_nhav) = plt.subplots(1, 2, sharey=True)
print("NH/AV vs AV")
xs, ys, covs = plot_results_scatter(
    ax_av_nhav,
    data,
    "AV",
    "NH_AV",
    pyrange=[0, 1.0e22],
    data_comp=comp,
    data_bohlin=bohlin,
    mark_comments=["lo_h_av", "hi_h_av"],
)
print("NH_AV vs 1/RV")
xs, ys, covs = plot_results_scatter(
    ax_rvi_nhav,
    data,
    "1_RV",
    "NH_AV",
    # pyrange=[0, 1.6e22],
    data_comp=comp,
    ignore_comments=["lo_h_av", "hi_h_av"])
plot_results_fit(xs, ys, covs, ax_rvi_nhav)
ax_rvi_nhav.set_ylabel("")
finalize_double(fig, "nhav_vs_av.pdf")

# redo the same thing, but with the outliers. Just need it for the numbers
print("NH_AV vs 1/RV (with outliers)")
xs, ys, covs = plot_results_scatter(
    ax_rvi_nhav,
    data,
    "1_RV",
    "NH_AV",
    # pyrange=[0, 1.6e22],
    data_comp=comp,
    ignore_comments=["hi_h_av"],
    mark_comments=["lo_h_av"])
plot_results_fit(xs, ys, covs, ax_rvi_nhav)
