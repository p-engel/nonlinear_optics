# unit test of correlation function with Reimann sum
import matplotlib.pyplot as plt
import numpy as np
from optical_rectification import definitions
from optical_rectification.definitions import Gaussian, corr

def test_corr_gaussian_zero_shift():
    w0 = 204.22  # [THz]
    tau = np.sqrt(2) * 75e-15 * 1e12  # [ps]
    w = np.linspace(150, 250, 2**10)
    dw = w[1] - w[0]

    g = Gaussian(w0=w0, tau=tau, E0=1)
    Ew = g.field_w(w)

    I0 = corr(Ew, dw, 0)
    expected = abs(g.E0_w)**2 * np.sqrt(2*np.pi) / g.tau

    assert np.isclose(I0, expected, rtol=1e-2), (
        f"The autocorrelation (at 𝜏 = 0) for a noormalised gaussian is 1"
    )
