# test gaussian pulse
import numpy as np
from matplotlib import pyplot as plt

from optical_rectification.definitions import Gaussian, TBP, Z0, c

# constants
r = np.linspace(-4e-3, 4e-3, 300)
r_pos = r[r >= 0]
z = np.linspace(-25e-6, 25e-6, 1000)

def test_gaussian_amplitude():
    try:
        g = Gaussian()
        print("The peak power is ", g.peak_power*1e-9, " [GW]")
        amplitude = g.amplitude(r_pos, 400e-6) 
        amplitude0 = np.sqrt(4 * Z0 * g.peak_power / np.pi) / g.waist0
        cond = ( abs(amplitude[0]) - amplitude0 ) <= 1e-2
        assert cond, (
            "At r=0 the amplitude should be maximum"
        )

        cond2 = ( abs(amplitude[1]) - amplitude0 ) <= 1e-2
        assert cond2, (
            "For a large Rayleigh range the phase term is negligible"
        )

    except AssertionError as a:
        print(f'Assertion error: {a}')
    return

def test_gaussian():
    try:
        pulse = Gaussian()
        E_w = pulse.field_w(0)
        E_peak = pulse.amplitude(0,0) / ( pulse.delta / np.sqrt(2) )
        I_peak = np.abs(E_peak)**2 / (2*Z0)
        I_w = np.abs(E_w)**2 / (2*Z0)

        ind_fwhm = abs(1/2 - I_w[0, :]/I_peak) < 1.2e-2
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

test_gaussian_amplitude()

### Gaussian beam plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
g = Gaussian()
Z, R = np.meshgrid(z, r)

# --- left: Gaussian envelope at z=0 ---
I0 = abs( g.amplitude(r_pos, 0) )**2 / (2 * 377)
P0 = I0 * (np.pi * g.waist0**2 / 2)
# print( Gaussian().waist0)
# ax1.plot(r_pos*1e3, I0*1e-13, 'r')
# ax1.set_ylabel(r"$I_0(r, z=0) \quad [GW/cm^2]$")
ax1.plot(r_pos*1e3, P0*1e-9, 'r')
ax1.set_ylabel(r"$P_{\rm peak}(r, z=0) \quad [GW]$")
ax1.set_title("Gaussian envelope")
ax1.set_xlabel(r"$r \quad [mm]$")
ax1.set_xlim([0, max(r_pos*1e3)])

# --- right: Gaussian beam field ---
E = g.amplitude(R, Z) * np.exp(1j * g.k0 * Z)
vmax = np.max(np.abs(E)) 
im = ax2.imshow(
    abs(np.real(E)),
    extent=[z.min()*1e6, z.max()*1e6, r.min()*1e3, r.max()*1e3],
    origin='lower',
    aspect='auto',
    cmap='jet',
    # cmap='RdBu_r',
    # cmap='hot',
    vmin=-0*vmax,
    vmax=vmax,
)
ax2.set_xlabel('z [µm]')
ax2.set_ylabel('r [mm]')
ax2.set_title('Gaussian Beam Electric Field')
plt.colorbar(
    im, ax=ax2,
    label=r'$\vert {\rm Re}\{E_0\} \vert \quad [V/m]$'
)

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(6.4, 4.8))
w, E_w, I_w = test_gaussian()
# lam = c * 2*np.pi / (w * 1e12) * 1e9
Pw = ( np.pi * g.waist0**2 / 2 ) * I_w * 1e-6
ax.plot(w/(2*np.pi), abs(E_w[50, :])*1e-7, 'r')
ax.set_title("Gaussian Pulse")
ax.set_xlabel(r"$\nu \quad [THz]$")
ax.set_ylabel(r"$|E| \times 10^{7} \quad [V/m\cdot ps]$")
ax.set_ylim(0,3)
ax.tick_params(axis='y', labelcolor='r')
ax1 = ax.twinx()
ax1.plot(w/(2*np.pi), Pw[0, :], 'k')
ax1.set_ylim(0,5)
ax1.set_ylabel(r"$P \quad [MW \cdot ps^2]$")
ax1.tick_params(axis='y',labelcolor='k')
plt.grid(True)
plt.show()
