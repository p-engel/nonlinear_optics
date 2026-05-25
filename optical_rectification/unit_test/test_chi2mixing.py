# unit test Chi2_mixing()
import numpy as np
from optical_rectification.definitions import Gaussian, Index, Dispersion,\
Chi2_mixing, chi2_factor

# global variables
pulse = Gaussian()
dw = pulse.w[1] - pulse.w[0]
Ew = pulse.field_w(0)
bandwidth = 5.88   # [THz]
n_dps = 2 * int( 2*np.pi*bandwidth / dw )
N = n_dps + 1
Ω = np.linspace(0, n_dps, N) * dw

def test_chi2_mixing_correlation():
    """
    Test that a single input field, with zero-frequency shift,
    yields the autocorrelation at maximum
    """
    try:
        chi2_mixing = Chi2_mixing(Ew, dw, N)

        I_max = chi2_mixing.correlation()[20, 0]
        perseval_pref = np.sqrt(  np.pi / (2 / pulse.delta**2) )
        A0_w = pulse.amplitude(pulse.r, 0) / (pulse.delta / np.sqrt(2))
        expected = perseval_pref * abs(A0_w)**2

        assert np.isclose(I_max.real, expected[20], rtol=1e-6), (
            f"The autocorrelation (at 𝜏 = 0) for a noormalised gaussian is 1"
        )
    except AssertionError as a:
        print(f'AssertionError: {a}')

    return 0.0

print( test_chi2_mixing_correlation() )

# def test_chi2_mixing_cascade():
#     chi2_mixing = Chi2_mixing(Ew, dw, phase_match=k.phase_match())
# 
#     # define THz field
#     Ω_max = 12; m_max = int(Ω_max / dw)
#     Ω = np.arange(1, m_max + 1) * dw
#     pref = chi2_factor(Ω, k.k_Ω)
#     E_thz = pref * chi2_mixing.correlation()
#     print(E_thz.shape)
# 
#     try:
#         Ew_mix = chi2_mixing.cascade(E_thz)
# 
#         assert Ew_mix.shape == Ew.shape, (
#             f"Mixed optical field isn't the same dimension as",
#             "input optical field"
#         )
# 
#     except AssertionError as a:
#         print(f'AssertionError: {a}')
# 
#     return 0.0
# 
# print( test_chi2_mixing_cascade() )
