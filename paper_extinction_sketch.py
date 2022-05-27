from paper_rcparams import column_width
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from dust_extinction.shapes import FM90

TOTAL_COLOR = 'k'
LIN_COLOR = 'gray'
BUMP_COLOR = 'r'
RISE_COLOR = 'xkcd:bright blue'

fig, ax = plt.subplots()

# generate the curves and plot them
# x = np.arange(3.8, 8.6, 0.1) / u.micron
x = np.linspace(3.8, 8.6, 100) / u.micron

tot = FM90()(x)
rise = FM90(C3=0.0)(x)
bump = FM90(C4=0.0)(x)
lin = FM90(C3=0.0, C4=0.0)(x)

ax.plot(x, bump, color=BUMP_COLOR)
ax.plot(x, rise, color=RISE_COLOR)
ax.plot(x, lin, color=LIN_COLOR)
ax.plot(x, tot, label="total", color=TOTAL_COLOR, ls='dotted')
ax.fill_between(x.value, lin, bump, color=BUMP_COLOR, alpha=0.15)
ax.fill_between(x.value, lin, rise, color=RISE_COLOR, alpha=0.15)

ax.text(4.8, 4.4, "bump\n$C_3, \\gamma, x_0$", color=BUMP_COLOR, ha='center')
ax.text(8.30, 6.3, "rise\n$C_4$", color=RISE_COLOR, ha='center')
ax.text(5.2, 3.0, "linear\n$C_1 + C_2x$", color=LIN_COLOR)

# ax.set_xlabel(r"$x$ [$\mu m^{-1}$]") # x = 1/lambda label
ax.set_xlabel(r"$1/\lambda$ [$\mu m^{-1}$]") # 1/lambda label
# ax.set_xlabel(r"$\lambda [$\mu m$]") # lambda label
ax.set_ylabel(r"$A(\lambda)/A(V)$")

ax.legend(loc="best")
fig.set_size_inches(column_width, column_width)
# plt.show()
# fig.savefig('paper-plots/curve.pdf', bbox_inches='tight')
fig.savefig('paper-plots/curve.png', bbox_inches='tight', dpi=300)
