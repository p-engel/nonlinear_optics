# propagation.py
import numpy as np
from .definitions import c_thz, chi2_factor, chi2_mixing, Dispersion

class ORPropagator:
    def __init__(self, w, Ω, index_w, index_Ω):
        """Assumes w and Ω are on a uniform grid"""
        self.w = w
        self.Ω = Ω
        self.dw = w[1] - w[0]

        # self.alpha_w = index_w.alpha()
        self.alpha_w = 1
        self.alpha_Ω = index_Ω.alpha()

        self.dispersion = Dispersion(
            w , index_w.sellmeier(), Ω=self.Ω, n_Ω=index_Ω.n()
        )

        self.pref_w = chi2_factor(w, self.dispersion.k)
        self.pref_Ω = chi2_factor(Ω, self.dispersion.k_Ω)

        # Phase matching matrices
        self.Dk_up = self.dispersion.phase_match(conj=False)
        self.Dk_dn = self.dispersion.phase_match(conj=True)

        self.Nw = len(w)
        self.NΩ = len(Ω)

    def pack(self, Ew, EΩ):
        return np.concatenate([Ew, EΩ])

    def unpack(self, y):
        return y[:self.Nw], y[self.Nw:]

    def rhs(self, z, y):
        Ew, EΩ = self.unpack(y)

        dEw = np.zeros_like(Ew, dtype=complex)
        dEΩ = np.zeros_like(EΩ, dtype=complex)

        # --- terahertz field ode ---
        for m in range(self.NΩ):
            dEΩ[m] = (
                -0.5 * self.alpha_Ω[m] * EΩ[m] +
                -0.5 * 1j * self.pref_Ω[m] * chi2_mixing(Ew, dw, m, Dk_up, z)
            )

        # --- optical field ode ---
        for m in range(self.Nw):
            dEw[m] = (
                -0.5 * self.alpha_w * Ew[m]
                # -0.5 * self.alpha_w[m] * Ew[m] +
                # -0.5 * 1j * self.pref_Ω[m]
                # * chi2_mixing(Ew+Ω?, dw, m, Dk_up.T, z, E_conj=EΩ)
                # -0.5 * 1j * self.pref_Ω[m]
                # * chi2_mixing(Ew+Ω?, dw, m, Dk_dn.T, z, E_conj=EΩ)
            )

    return self.pack(dEw, dEΩ)
