# test gaussian pulse
import numpy as np
from matplotlib import pyplot as plt

from optical_rectification import definitions

# constants
w0 = definitions.c / 1468e-9 *1e-12  # [THz]
t_fwhm = np.sqrt(2) * 75e-15 * 1e12  # [ps]
E0 = 5.4315195283 * 1e8  # [V/m]
w = np.linspace(150, 250, 2**10)  # [THz] domain

def test_gaussian():
    try:
        pulse = definitions.Gaussian(w0=w0, t_fwhm=t_fwhm, E0=E0)
        E_w = pulse.field_w(w)
        ind_cond = abs(1/2 - E_w/pulse.E0_w) < 1e-3
        w_fwhm = abs( max(w[ind_cond]) - min(w[ind_cond]) )
        print("The bandwith (FWHM) is ", w_fwhm, " [THz]." )
        cond = abs((t_fwhm * w_fwhm / (4*np.pi)) - definitions.tbp) < 1e-2
        assert cond, "The pulse doesn't satisfy the time-bandwith product"
    except AssertionError as a:
        print(f'Assertion error: {a}')
    return pulse

fig, ax = plt.subplots(figsize=(6.4,4.8))

pulse = test_gaussian()
ax.plot(w, pulse.field_w(w))
ax.set_title("Gaussian Pulse")
ax.set_xlabel(r"$\omega \quad [THz]$")
ax.set_ylabel(r"$E(\omega) \quad [V/m\cdot ps]$")

plt.grid(True)
plt.show()
