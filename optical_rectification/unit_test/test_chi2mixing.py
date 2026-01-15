# unit test chi2_mixing.py
import numpy as np
from optical_rectification.definitions import Gaussian, Index, Dispersion,\
chi2_mixing

def test_chi2mixing_gaussian_zero_shift():
    """
    Test that a single input field, with zero-frequency shift,
    yields the autocorrelation at maximum
    """
    w0 = 204.22                                                             # [THz]
    t_fwhm = 75e-15 * 1e12                                                     # [ps]
    E0 = 5.4315195283 * 1e8                                                 # [V/m]
    pulse = Gaussian(w0=w0, t_fwhm=t_fwhm, E0=E0)
    w = np.linspace((w0 - 3*pulse.delta), (w0 + 3*pulse.delta), 2**10)      # [THz]
    dw = w[1] - w[0]

    Ew = pulse.field_w(w)
    k = Dispersion(w, Index(w).sellmeier())

    I_max = chi2_mixing(Ew, dw, 0, k.phase_match(), 4e-3)
    expected = abs(pulse.E0_w)**2 * np.sqrt(2*np.pi) / pulse.tau

    assert np.isclose(I_max, expected, rtol=1e-2), (
        f"The autocorrelation (at 𝜏 = 0) for a noormalised gaussian is 1"
    )
    return 0.0

print( test_chi2mixing_gaussian_zero_shift() )
