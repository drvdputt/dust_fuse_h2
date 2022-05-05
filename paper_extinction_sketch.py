from paper_rcparams import column_width
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from dust_extinction.shapes import FM90

fig, ax = plt.subplots()

# generate the curves and plot them
x = np.arange(3.8, 8.6, 0.1) / u.micron

ext_model = FM90()
ax.plot(x, ext_model(x), label="total")

ext_model = FM90(C3=0.0, C4=0.0)
ax.plot(x, ext_model(x), label="linear term")

# ext_model = FM90(C1=0.0, C2=0.0, C4=0.0)
ext_model = FM90(C1=0.0, C2=0.0, C4=0.0)
ax.plot(x, ext_model(x), label="bump term")

ext_model = FM90(C1=0.0, C2=0.0, C3=0.0)
ax.plot(x, ext_model(x), label="FUV rise term")

ax.set_xlabel(r"$x$ [$\mu m^{-1}$]")
ax.set_ylabel(r"$A(\lambda)/A(V)$")

ax.legend(loc="best")
fig.set_size_inches(column_width, column_width)
fig.savefig('paper-plots/curve.pdf', bbox_inches='tight')
