# test disperison relation
import numpy as np
from matplotlib import pyplot as plt

from optical_rectification.definitions import Index, Dispersion, c

def test_dispersion():
    Ω = np.linspace(1e-12, 12, 2**9) # [THz]
    w = np.linspace(115, 299, 2**9)  # [THz]
    w0 = 203
    dispersion = Dispersion(
                            w, Index(w).sellmeier(), 
                            Ω, Index(Ω).n()
    )
    try:
        f = dispersion.k * dispersion.nu
        cond = np.mean( np.abs(f - w) ) <= 1e-2;
        cond1 = np.shape( dispersion.deltak(w0=w0) ) == np.shape(Ω)
        cond2 = np.shape( dispersion.phase_match() ) == (len(w), len(Ω))
        assert cond, "the dispersion relation should satisfy \
                        w = k * nu"
        assert cond1, "∆k the phase matching condition \
                        is approximated as ∆k(Ω) and thus has Dim(Ω)"
        assert cond2, "∆k the phase matching condition has Dim(w) by Dim(Ω)"
    except AssertionError as a:
        print(f"Assertion error: {a}");

    return w0, w, Ω, dispersion

# plot dispersion relation
fig, ax1 = plt.subplots(figsize=(6.4,4.8))
w0, w, Ω, dispersion = test_dispersion()
k0 = w / c
ax1.plot(w, k0, label="DSTMS crystal")
ax1.plot(w, dispersion.k, '--', label="free space")
ax1.set_title('Dispersion')
ax1.set_ylabel(r'$k(\omega)$ [$\rm{m}^{-1}$]')
ax1.set_xlabel(r'$\omega$ [THz]')
ax1.legend()

# phase match
iw0 = np.argmin(np.abs(w - w0))
fig2, ax2 = plt.subplots(figsize=(6.4,4.8))
ax2.plot(
            Ω, dispersion.phase_match()[iw0, :] * 1e-3, 
            label="exact: $k(\omega_0 + \Omega) - k(\omega_0) - k(\Omega)$"
)
ax2.plot(
            Ω, dispersion.deltak()[iw0, :] * 1e-3, 
            "--", label=r"approx: $\Omega n_g(\omega_0)/c - k(\Omega)$"
)
ax2.set_title('Phase matching condition DSTMS')
ax2.set_ylabel(r'${\Delta}k$ [${\rm mm}^{-1}$]')
ax2.set_xlabel(r'$\Omega$ [THz]')
ax2.legend()


plt.grid(True)
plt.show()
