# propagator.py
import numpy as np
from scipy.integrate import solve_ivp
from .definitions import c_thz, chi2_factor, Chi2_mixing, Dispersion, Index, par

class ORPropagator:
    def __init__(self, w, Ω_max, index_w, 
        index_Ω=None, pulse=None, cascade=True
    ):
        self.w = w
        self.dw = w[1] - w[0]
        self.Ω = np.arange(1, int(Ω_max/self.dw) + 1) * self.dw
        self.index_w = index_w
        self.index_Ω = Index(self.Ω) if index_Ω is None else index_Ω

        self.alpha_w = self.index_w.alpha()
        self.alpha_Ω = self.index_Ω.alpha()

        self.dispersion = Dispersion(
            w , self.index_w.sellmeier(), 
            Ω=self.Ω, n_Ω=self.index_Ω.n()
        )

        self.pref_w = chi2_factor(w, self.dispersion.k)
        self.pref_Ω = chi2_factor(self.Ω, self.dispersion.k_Ω)

        # Phase matching matrices
        # self.Dk_up = self.dispersion.phase_match(conj=False)
        # self.Dk_dn = self.dispersion.phase_match(conj=True)

        self.Nw = len(w)
        self.NΩ = len(self.Ω)

        self.pulse = pulse
        self.cascade = cascade

    def pack(self, Ew, EΩ):
        return np.concatenate([Ew, EΩ])

    def unpack(self, y):
        return y[:self.Nw], y[self.Nw:]

    def rhs(self, z, y):
        Ew, EΩ = self.unpack(y)
        
        Dk = self.dispersion.phase_match()
        chi2_mixing = Chi2_mixing(Ew, self.dw, phase_match=Dk, z=z)

#        dEw = np.zeros_like(Ew, dtype=complex)
#        dEΩ = np.zeros_like(EΩ, dtype=complex)
#
        # --- terahertz field ode ---
#         for m in range(self.NΩ):
#             dEΩ[m] = (
#                 -0.5 * self.alpha_Ω[m] * EΩ[m] +
#                 -0.5 * 1j * self.pref_Ω[m] 
#                 * chi2_mixing(Ew, self.dw, m, self.Dk_up, z)
#             )

        dEΩ = (
            -0.5 * self.alpha_Ω * EΩ
            -0.5j * self.pref_Ω * chi2_mixing.correlation()
        )

        # --- optical field ode ---
#         for m in range(self.Nw):
#             dEw[m] = (
#                 -0.5 * self.alpha_w[m] * Ew[m]
#             )
        dEw = -0.5 * self.alpha_w * Ew
        if self.cascade:
            dEw += -0.5j * self.pref_w * chi2_mixing.cascade(EΩ)

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
