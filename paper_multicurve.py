from get_data import get_merged_table
from extinction_curve_set import ExtinctionCurveSet
import numpy as np
from matplotlib import pyplot as plt, cm, colors
import cmasher
from scipy.stats import linregress

lambdam1label = "$1/\\lambda$ [$\\rm{\\mu m}$]"

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


# try dividing by average curve too, and hope my brain survives trying to interpret this.
avgcurve = np.average(curves, axis=0)
sigmacurve = np.std(curves, axis=0)
# curves = (curves - avgcurve[None, :]) / sigmacurve[None, :]
# curves = curves / avgcurve

def plot_all_curves(fig, ax, color_axis):
    color_data = data[color_axis]
    # cmap = cm.coolwarm
    cmap = cmasher.iceburn_r
    norm = colors.Normalize(vmin=min(color_data), vmax=max(color_data))
    mp = cm.ScalarMappable(cmap=cmap, norm=norm)
    line_colors = mp.to_rgba(color_data)

    for curve, c in zip(curves, line_colors):
        # change color later (color code by 1/RV ?)
        ax.plot(1 / (wavs / 10000), curve, alpha=0.5, c=c)
        ax.set_xlabel(lambdam1label)
        ax.set_ylabel("$A(\\lambda)/N(\\rm{H})$ [$\\rm{mag\\,cm}^{-2}$]")
        
    # ax.set_yscale("log")
    cb = fig.colorbar(mp, ax=ax)
    cb.set_label(color_axis)

def slope_all_wavelengths(variable='1_RV'):
    """Calculate the slope of Alambda/NH with respect to 1/RV, for each
       wavelength individually.

       Doing it naively first. 
    """
    rvi = data[variable]

    xpoints = 1/(wavs / 10000)
    ypoints = np.zeros(len(wavs))
    extrapoints = np.zeros(len(wavs))
    for iw, w in enumerate(wavs):
        slope, interecept, r, p, stderr = linregress(rvi, curves[:, iw])
        ypoints[iw] = slope
        extrapoints[iw] = r**2
        
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(xpoints, ypoints)
    axs[0].set_ylabel("slope $A(\\lambda)/N(\\rm{H})$ vs " + variable)# + "relative to $\\langle A(\\lambda)/N(\\rm{H})\\rangle$")
    axs[0].axhline(0)
    axs[1].plot(xpoints, extrapoints)
    axs[1].set_ylabel("$R^2$")
    axs[1].set_xlabel(lambdam1label)
    
color_axis = "1_RV", "A2175_AV", "A1000_AV", "AV_NH"
fig, axs = plt.subplots(1, len(color_axis), figsize=(max(len(color_axis) * 3, 4), 3))
for c, ax in zip(color_axis, axs):
    plot_all_curves(fig, ax, c)

plt.tight_layout()

# slope_all_wavelengths()
# slope_all_wavelengths("AV_NH")
slope_all_wavelengths("A3000_NH")

plt.show()

