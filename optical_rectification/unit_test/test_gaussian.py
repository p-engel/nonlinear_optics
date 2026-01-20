# test gaussian pulse
import numpy as np
from matplotlib import pyplot as plt

from optical_rectification.definitions import Gaussian, c, TBP

# constants
EPS0 = 8.85e-12                         # vacuum permitivity [C^2 / Kg^1 / m^3 /s^2]
w0 = c / 1468e-9 * 1e-12                # [THz]
t_fwhm = 75e-15 * 1e12                  # [ps]
E0 = 5.4315195283 * 1e8                 # [V/m]

def test_gaussian():
    try:
        pulse = Gaussian(w0=w0, t_fwhm=t_fwhm, E0=E0)
        sigma = 1/ np.sqrt(2) * pulse.delta
        w = np.linspace( (w0 - sigma), (w0 + sigma), 2**11 )
        E_w = pulse.field_w(w)
        E_peak = pulse.E0_w
        I_peak = (EPS0*c/2) * np.abs(E_peak)**2
        I_w = (EPS0*c/2) * np.abs(E_w)**2
        
        ind_fwhm = abs(1/2 - I_w/I_peak) < 1.2e-2
        w_fwhm = abs( max(w[ind_fwhm]) - min(w[ind_fwhm]) )
        print("The bandwith (FWHM) is ", w_fwhm, " [THz]." )
        cond = abs((t_fwhm * w_fwhm) - TBP) < 1e-2
        assert cond, "The pulse doesn't satisfy the time-bandwith product"

    except AssertionError as a:
        print(f'Assertion error: {a}')
    return w, E_w, I_w

fig, ax = plt.subplots(figsize=(6.4, 4.8))

w, E_w, I_w = test_gaussian()
ax.plot(w, E_w, 'r')
ax.set_title("Gaussian Pulse")
ax.set_xlabel(r"$\omega \quad [THz]$")
ax.set_ylabel(r"$E(\omega) \quad [V/m\cdot ps]$")
ax.tick_params(axis='y', labelcolor='r')
ax1 = ax.twinx()
ax1.plot(w, I_w, 'k')
ax1.set_ylabel(r"$I(\omega) \quad [W/m^2 \cdot ps^2]$")
ax1.tick_params(axis='y',labelcolor='k')

plt.grid(True)
plt.show()
