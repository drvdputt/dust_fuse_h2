from get_data import get_merged_table
from extinction_curve_set import ExtinctionCurveSet
import numpy as np
from matplotlib import pyplot as plt, cm, colors


data = get_merged_table()
ec = ExtinctionCurveSet(data)

minwav = 950
maxwav = 3000
wavs = np.linspace(minwav, maxwav, num=500)

curves = np.zeros((len(data), len(wavs)))

# get all Alambda / AV curves
for iw, w in enumerate(wavs):
    curves[:, iw] = ec.evaluate(w)

# convert to Alambda / NH
for row in range(len(data)):
    curves[row] *= data["AV"][row] / data["nhtot"][row]

fig, ax = plt.subplots()

color_axis = "AV_NH"
color_data = data["AV_NH"]
cmap = cm.coolwarm
norm = colors.Normalize(vmin=min(color_data), vmax=max(color_data))
mp = cm.ScalarMappable(cmap=cmap, norm=norm)
line_colors = mp.to_rgba(color_data)

for curve, c in zip(curves, line_colors):
    # change color later (color code by 1/RV ?)
    ax.plot(1 / (wavs / 10000), curve, alpha=0.5, c=c)

ax.set_xlabel("$1/\\lambda$ [$\\rm{\\mu m}$]")
ax.set_ylabel("$A(\\lambda)/N(\\rm{H})$ [$\\rm{mag\\,cm}^{-2}$]")
ax.set_yscale("log")

cb = fig.colorbar(mp)
cb.set_label(color_axis)
plt.show()
