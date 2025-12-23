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
        cond = np.mean(np.abs(f - w)) <= 1e-2;
        cond1 = np.shape(dispersion.deltak()) == np.shape(Ω)
        assert cond, "the dispersion relation should satisfy \
                        w = k * nu"
        assert cond, "∆k the phase matching condition \
                        is approximated as ∆k(Ω) and thus has Dim(Ω)"
    except AssertionError as a:
        print(f"Assertion error: {a}");

    return w, dispersion.k

# plot dispersion relation
fig, ax1 = plt.subplots(figsize=(6.4,4.8))
w, k = test_dispersion()
k0 = w / c
#ax1.plot(w, k0, label="DSTMS crystal")
#ax1.plot(f, k0, '--', label="DSTMS crystal")
ax1.plot(w, k0, label="DSTMS crystal")
ax1.plot(w, k, '--', label="free space")
ax1.set_title('Dispersion')
ax1.set_ylabel(r'$k(\omega)$ [$\rm{m}^{-1}$]')
ax1.set_xlabel(r'$\omega$ [THz]')
ax1.legend()
plt.grid(True)
plt.show()
