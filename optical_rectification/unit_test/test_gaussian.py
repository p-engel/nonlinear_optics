# test gaussian pulse
import numpy as np
from matplotlib import pyplot as plt

from optical_rectification.definitions import Gaussian, TBP, c

# constants
EPS0 = 8.85e-12                         # [C^2 / Kg^1 / m^3 /s^2]

def test_gaussian():
    try:
        pulse = Gaussian()
        E_w = pulse.field_w()
        E_peak = pulse.E0_w
        I_peak = (EPS0*c/2) * np.abs(E_peak)**2
        I_w = (EPS0*c/2) * np.abs(E_w)**2
        
        ind_fwhm = abs(1/2 - I_w/I_peak) < 1.2e-3
        w_fwhm = abs( 
            max(pulse.w[ind_fwhm]) - min(pulse.w[ind_fwhm])
        )
        w_fwhm_expected = (
            pulse.delta * np.sqrt(2 * np.log(2)) / (2*np.pi)
        )
        t_fwhm = pulse.tau * np.sqrt( 2 * np.log(2) )
        print(
            "The bandwith (FWHM) is ", w_fwhm/(2 * np.pi),
            " [THz]." 
        )
        print(
            "The bandwith (FWHM) is ", w_fwhm_expected,
            "[THz]."
        )
        cond = abs( 
            (t_fwhm * w_fwhm/(2 * np.pi)) - TBP 
        ) < 1e-2
        assert cond, (
            "The pulse doesn't satisfy the time-bandwith product"
        )

    except AssertionError as a:
        print(f'Assertion error: {a}')
    return pulse.w, E_w, I_w

fig, ax = plt.subplots(figsize=(6.4, 4.8))

w, E_w, I_w = test_gaussian()
lam = c * 2*np.pi / (w * 1e12) * 1e9
ax.plot(lam, E_w, 'r')
ax.set_title("Gaussian Pulse")
ax.set_xlabel(r"$\lambda \quad [nm]$")
ax.set_ylabel(r"$E(\omega) \quad [V/m\cdot ps]$")
ax.tick_params(axis='y', labelcolor='r')
ax1 = ax.twinx()
ax1.plot(lam, I_w*1e-5, 'k')
ax1.set_ylabel(r"$I(\omega) \quad [GW/cm^2 \cdot ps^2]$")
ax1.tick_params(axis='y',labelcolor='k')

plt.grid(True)
plt.show()
