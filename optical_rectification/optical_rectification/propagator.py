# propagator.py
import numpy as np
from scipy.integrate import solve_ivp
from .definitions import c_thz, chi2_factor, chi2_mixing, Dispersion, Index

class ORPropagator:
    def __init__(self, w, Ω_max, pulse=None):
        self.w = w
        self.dw = w[1] - w[0]
        self.Ω = np.arange(1, int(Ω_max/self.dw) + 1) * self.dw
        self.index_w = Index(w)
        self.index_Ω = Index(self.Ω)

        self.alpha_w = 1
        self.alpha_Ω = self.index_Ω.alpha()

        self.dispersion = Dispersion(
            w , self.index_w.sellmeier(), 
            Ω=self.Ω, n_Ω=self.index_Ω.n()
        )

        self.pref_w = chi2_factor(w, self.dispersion.k)
        self.pref_Ω = chi2_factor(self.Ω, self.dispersion.k_Ω)

        # Phase matching matrices
        self.Dk_up = self.dispersion.phase_match(conj=False)
        self.Dk_dn = self.dispersion.phase_match(conj=True)

        self.Nw = len(w)
        self.NΩ = len(self.Ω)

        self.pulse = pulse

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
                -0.5 * 1j * self.pref_Ω[m] 
                * chi2_mixing(Ew, self.dw, m, self.Dk_up, z)
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


def run_simulation(model, Ew0, z_span, z_eval=None):
    EΩ0 = np.zeros_like(model.Ω, dtype=complex)
    y0 = model.pack(Ew0, EΩ0)

    sol = solve_ivp(
        model.rhs,
        z_span,
        y0,
        method="DOP853",
        t_eval=z_eval,
        rtol=1e-5,
        atol=1e-8,
        max_step=(z_span[1] - z_span[0]) / 200
    )

    return sol
