"""
Plot of some Lya fits to put in paper.

One stis and one IUE should suffice.
"""
import lyafitting
import get_data
import paper_rcparams
from matplotlib import pyplot as plt
import numpy as np
import get_spectrum

data = get_data.get_merged_table()

# star1 = "HD099872"
star1 = "HD209339"
# star2 = "HD093028"
# star2 = "HD197770"
# star2 = "HD037525"
# star2 = "HD152248"
# star2 = "HD093827
# star2 = "HD047129"
star2 = "HD046202"

def plot_common(ax, target):
    lyafitting.prepare_axes(ax)
    ax.set_xlim((1160, 1350))
    ax.set_ylim((-1e-12, None))
    lognhi, fc, info = lyafitting.lya_fit(target, ax_fit=ax)
    handles, labels = ax.get_legend_handles_labels()
    order = np.argsort(labels)
    legend = plt.legend([handles[i] for i in order], [labels[i] for i in order], framealpha=0.84, facecolor='w', loc='lower right')
    legend.set_zorder(100)
    ax.text(0.1, 0.9, f"{target} | $\\mathrm{{log}}\\,N(\\mathrm{{HI}}) = {lognhi:.2f}$", transform=ax.transAxes,)
    fig.tight_layout()
    fig.set_size_inches(paper_rcparams.base_width, paper_rcparams.base_height / 3)
    fig.savefig(f"./paper-plots/lyafit-{target}.pdf", bbox_inches='tight')

for t in data['Name']:
    if t in get_spectrum.target_use_which_spectrum:
        fig, ax = plt.subplots()
        plot_common(ax, t)
        fig.set_size_inches(paper_rcparams.base_width, paper_rcparams.base_height / 3)
