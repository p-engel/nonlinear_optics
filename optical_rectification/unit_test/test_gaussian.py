# test gaussian pulse
import numpy as np
from matplotlib import pyplot as plt

from optical_rectification import definitions

# constants
EPS0 = 8.85e-12                         # vacuum permitivity [C^2 / Kg^1 / m^3 /s^2]
w0 = definitions.c / 1468e-9 *1e-12     # [THz]
t_fwhm = 75e-15 * 1e12                  # [ps]
E0 = 5.4315195283 * 1e8                 # [V/m]

def test_gaussian():
    try:
        pulse = definitions.Gaussian(w0=w0, t_fwhm=t_fwhm, E0=E0)
        w = np.linspace(
                (w0 - 3*pulse.delta), (w0 + 3*pulse.delta), 2**10
        )  # [THz] domain
        E_w = pulse.field_w(w)
        ind_cond = abs(1/2 - E_w/pulse.E0_w) < 1e-2
        print(w)
        print(ind_cond)
        print(E_w/pulse.E0_w)
        w_fwhm = abs( max(w[ind_cond]) - min(w[ind_cond]) ) / np.sqrt(2)
        print("The bandwith (FWHM) is ", w_fwhm, " [THz]." )
        cond = abs((t_fwhm * w_fwhm / (2*np.pi)) - definitions.TBP) < 1e-2
        assert cond, "The pulse doesn't satisfy the time-bandwith product"
    except AssertionError as a:
        print(f'Assertion error: {a}')
    return w, pulse

fig, ax = plt.subplots(figsize=(6.4, 4.8))

w, pulse = test_gaussian()
ax.plot(w, pulse.field_w(w), 'r')
ax.set_title("Gaussian Pulse")
ax.set_xlabel(r"$\omega \quad [THz]$")
ax.set_ylabel(r"$E(\omega) \quad [V/m\cdot ps]$")
ax.tick_params(axis='y', labelcolor='r')
ax1 = ax.twinx()
ax1.plot(w, (EPS0*definitions.c/2)*np.abs(pulse.field_w(w))**2, 'k')
ax1.set_ylabel(r"$I(\omega) \quad [W/m^2 \cdot ps^2]$")
ax1.tick_params(axis='y',labelcolor='k')

plt.grid(True)
plt.show()
