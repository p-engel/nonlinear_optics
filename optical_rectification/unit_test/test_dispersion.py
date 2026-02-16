# test disperison relation
import numpy as np
from matplotlib import pyplot as plt

from optical_rectification.definitions import Index, Dispersion, c_thz, c,\
Gaussian

def test_dispersion():
    # Ω = np.linspace(0, 10, 2**9) # [THz]
    # w = np.linspace(115, 299, 2**9)  # [THz]
    w0 = c_thz / 1468e-9  # [THz]
    t_fwhm = 75e-15 * 1e12  # [ps]
    E0 = 5.4315195283 * 1e8  # [V/m]
    pulse = Gaussian(w0=w0, t_fwhm=t_fwhm, E0=E0, Nw=2**10)
    w = pulse.w
    dw = abs(w[1] - w[0])                     # freq per grid spacing
    print(dw)
    Ω_max = 12; m_dps = int(Ω_max / dw)
    NΩ = m_dps + 1
    mvals = np.linspace(1, m_dps, NΩ); print(len(mvals))
    Ω = mvals * dw
    # w0 = 203
    index_Ω = Index(Ω)
    dispersion = Dispersion(
                            w, Index(w).sellmeier(), 
                            Ω=Ω, n_Ω=index_Ω.n()
    )
    try:
        f = dispersion.k * (c_thz / dispersion.n)
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

# # plot dispersion relation
# fig, ax1 = plt.subplots(figsize=(6.4,4.8))
# w0, w, Ω, dispersion = test_dispersion()
# k0 = w / c
# ax1.plot(w, k0, label="DSTMS crystal")
# ax1.plot(w, dispersion.k, '--', label="free space")
# ax1.set_title('Dispersion')
# ax1.set_ylabel(r'$k(\omega)$ [$\rm{m}^{-1}$]')
# ax1.set_xlabel(r'$\omega$ [THz]')
# ax1.legend()
# 
# # phase match
# iw0 = np.argmin(np.abs(w - w0))
# fig2, ax2 = plt.subplots(figsize=(6.4,4.8))
# ax2.plot(
#             Ω, dispersion.phase_match()[iw0, :] * 1e-3, 
#             '.', label=r"exact: $k(\omega_0 + \Omega) - k(\omega_0) - k(\Omega)$"
# )
# ax2.plot(
#             Ω, dispersion.deltak()[iw0, :] * 1e-3, 
#             "-k", label=r"approx: $\Omega n_g(\omega_0)/c - k(\Omega)$"
# )
# ax2.set_title('Phase matching condition DSTMS')
# ax2.set_ylabel(r'${\Delta}k$ [${\rm mm}^{-1}$]')
# ax2.set_xlabel(r'$\Omega$ [THz]')
# ax2.legend()
# 
# 
# plt.grid(True)
# plt.show()
