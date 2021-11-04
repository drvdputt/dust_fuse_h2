"""Functions used to rescale the data.

Used in 'pearson' and 'linear_ortho_fit'.

In some cases, the scale of y is very different from x, such as NH / AV
(~ e21) vs AV (~ 1). The uncertainties on these quantities are of
similar orders of magnitude. The covariance matrix for such a pair
quantities is of the order
Cxy = ( x**2 x*y
        x*y  y**2)

So if the orders of magnitude of x and y are very different, then Vx
will be very big, Vy very small, and cov(x,y) in between. This can cause
floating point precision problems.

A straightforward solution is linearly rescaling x, y, and Cxy so that
they are of similar order of magnitude. Statistic metrics such as the
correlation coefficient and the chi2 of a linear fit are invariant under
this transformation. The results of the fit, the slope and intercept,
can be rescaled again afterwards.

I have gathered some utility functions here, so it will be easier to
solve this type of problem when it comes up.

"""

import numpy as np

def rescale_data(xy, covs, factor_x, factor_y):
    """Multiply the data (x,y) by a scaling matrix
    S = ( fx 0
           0  fy ).

    The covariance matrices are transformed using
    C' = S.T C S

    Which multiplies Vx by fx^2, Vy by fy^2, and cov(x,y) by fx*fy, as
    it should.
    """
    S = np.array([[factor_x, 0], [0, factor_y]])
    xyr = np.einsum("ij,dj", S, xy)
    covr = np.einsum("ij,djk,kl", S.T, covs, S)
    return xyr, covr

def unscale_mb(m, b, factor_x, factor_y):
    """Unscale the slope and intercept of a linear fit that was performed to
       data that was rescaled."""
    # x * factor_x * m = y * factor_y --> y / x = ...
    m_real = m * factor_x / factor_y
    # b = y * factor_y --> y = ...
    b_real = b / factor_y
    return m_real, b_real
