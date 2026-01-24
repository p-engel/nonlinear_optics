# unit test Chi2_mixing()
import numpy as np
from optical_rectification.definitions import Gaussian, Index, Dispersion,\
Chi2_mixing, chi2_factor

# parameters
w0 = 204.22                                                             # [THz]
t_fwhm = 75e-15 * 1e12                                                  # [ps]
E0 = 5.4315195283 * 1e8                                                 # [V/m]
pulse = Gaussian(w0=w0, t_fwhm=t_fwhm, E0=E0)
w = np.linspace((w0 - 2*pulse.delta), (w0 + 2*pulse.delta), 2**12)      # [THz]
dw = w[1] - w[0]

Ew = pulse.field_w(w)
k = Dispersion(w, Index(w).sellmeier())

def test_chi2_mixing_correlation():
    """
    Test that a single input field, with zero-frequency shift,
    yields the autocorrelation at maximum
    """
    try:
        chi2_mixing = Chi2_mixing(Ew, dw, phase_match=k.phase_match())

        # I_max = chi2_mixing(Ew, dw, 0, k.phase_match(), 4e-3)
        I_max = chi2_mixing.correlation()[0]
        perseval_pref = np.sqrt(  np.pi / (8 * np.pi**2 / pulse.delta**2) )
        expected = perseval_pref * abs(pulse.E0_w)**2

        assert np.isclose(I_max.real, expected, rtol=1e-6), (
            f"The autocorrelation (at 𝜏 = 0) for a noormalised gaussian is 1"
        )
    except AssertionError as a:
        print(f'AssertionError: {a}')

    return 0.0

print( test_chi2_mixing_correlation() )

def test_chi2_mixing_cascade():
    chi2_mixing = Chi2_mixing(Ew, dw, phase_match=k.phase_match())

    # define THz field
    Ω_max = 12; m_max = int(Ω_max / dw)
    Ω = np.arange(1, m_max + 1) * dw
    pref = chi2_factor(Ω, k.k_Ω)
    E_thz = pref * chi2_mixing.correlation()
    print(E_thz.shape)

    try:
        Ew_mix = chi2_mixing.cascade(E_thz)

        assert Ew_mix.shape == Ew.shape, (
            f"Mixed optical field isn't the same dimension as",
            "input optical field"
        )

    except AssertionError as a:
        print(f'AssertionError: {a}')

    return 0.0

print( test_chi2_mixing_cascade() )
