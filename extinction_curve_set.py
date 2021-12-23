import numpy as np

class ExtinctionCurveSet:
    """Tools to evaluate FM90 function for many stars at once"""

    micron = 10000
    params_needed = [f"CAV{n}" for n in range(1, 5)] + ["gamma", "x_o"]

    def __init__(self, data):
        """data : astropy table containing the named columns "CAV[1-4]", "x_o", "gamma" """
        # unpack like this, so the equations are more readable
        (self.cav1, self.cav2, self.cav3, self.cav4, self.gamma, self.x0) = [
            data[p] for p in ExtinctionCurveSet.params_needed
        ]
        (
            self.cav1_unc,
            self.cav2_unc,
            self.cav3_unc,
            self.cav4_unc,
            self.gamma_unc,
            self.x0_unc,
        ) = [data[p + "_unc"] for p in ExtinctionCurveSet.params_needed]

    def evaluate(self, w):
        """
        Evaluate the FM90 function for all sightlines, at a given wavelength.

        w : wavelength in angstrom
        """
        # x is 1/w in micron
        x = 1 / (w / ExtinctionCurveSet.micron)
        # drude
        D = x ** 2 / ((x ** 2 - self.x0 ** 2) ** 2 + (x * self.gamma) ** 2)
        F = 0.5392 * (x - 5.9) ** 2 + 0.05644 * (x - 5.9) ** 3
        Aw_AV = self.cav1 + self.cav2 * x + self.cav3 * D + self.cav4 * F
        return Aw_AV

    def evaluate_unc(self, w):
        """
        Uncertainty on the FM90 function, as propagated from the parameter
        uncertainties.
        """
        x = 1 / (w / ExtinctionCurveSet.micron)

        # drude
        D_denom = (x ** 2 - self.x0 ** 2) ** 2 + (x * self.gamma) ** 2
        D = x ** 2 / D_denom
        F = 0.5392 * (x - 5.9) ** 2 + 0.05644 * (x - 5.9) ** 3

        # dD / dg = -2gx^4 / (g^2 x^2 + (x0^2 + x^2)^2)^2
        # dD / dx0 = -4mx^2(m^2+x^2)/(g^2x^2+(m^2+x^2)^2)^2

        # multivariate error propagation
        D_unc_gamma = 2 * self.gamma * x ** 4 / D_denom ** 2 * self.gamma_unc
        D_unc_x0 = (
            4 * self.x0 * x ** 2 * (x ** 2 - self.x0 ** 2) / D_denom ** 2 * self.x0_unc
        )
        VD_rel = (D_unc_gamma ** 2 + D_unc_x0 ** 2) / D ** 2
        # sum of relative variances
        Vcav3_rel = (self.cav3_unc / self.cav3) ** 2
        Vterm3 = (self.cav3 * D) ** 2 * (Vcav3_rel + VD_rel)

        VA = (
            self.cav1_unc ** 2
            + (self.cav2_unc * x) ** 2
            + Vterm3
            + (self.cav4_unc * F) ** 2
        )
        return np.sqrt(VA)


