# propagator.py
# import numpy as np
from numpy import arange, zeros_like, concatenate
# from scipy.integrate import solve_ivp
from .definitions import chi2_factor, Chi2_mixing, Dispersion, Index

class ORPropagator:
    def __init__(self, Ω_max, pulse, cascade=True):
        self.pulse = pulse
        self.w = pulse.w
        self.dw = self.w[1] - self.w[0]
        self.Ω = arange(1, int(Ω_max/self.dw) + 1) * self.dw

        self.index_Ω = Index(self.Ω)
        self.index_w = Index(self.w)

        self.alpha_w = self.index_w.alpha()
        self.alpha_Ω = self.index_Ω.alpha()

        self.dispersion = Dispersion(
            self.w, self.index_w.sellmeier(), 
            Ω=self.Ω, n_Ω=self.index_Ω.n()
        )

        self.pref_w = chi2_factor(self.w, self.dispersion.k)
        self.pref_Ω = chi2_factor(self.Ω, self.dispersion.k_Ω)

        self.Nw = len(self.w)
        self.NΩ = len(self.Ω)

        self.cascade = cascade

        self.Ew0 = self.pulse.field_w()
        self.EΩ0 = zeros_like(self.Ω, dtype=complex)

    def pack(self, Ew, EΩ):
        return concatenate([Ew, EΩ])

    def unpack(self, y):
        return y[:self.Nw], y[self.Nw:]

    def rhs(self, z, y):
        Ew, EΩ = self.unpack(y)

        Dk = self.dispersion.phase_match()
        chi2_mixing = Chi2_mixing(Ew, self.dw, phase_match=Dk, z=z)

        # --- terahertz field ode ---
        dEΩ = (
            -0.5 * self.alpha_Ω * EΩ
            -0.5j * self.pref_Ω * chi2_mixing.correlation()
        )
        # --- optical field ode ---
        dEw = -0.5 * self.alpha_w * Ew
        if self.cascade:
            dEw += -0.5j * self.pref_w * chi2_mixing.cascade(EΩ)

        return self.pack(dEw, dEΩ)


# def run_simulation(model, Ew0, z_span, z_eval=None):
#     EΩ0 = np.zeros_like(model.Ω, dtype=complex)
#     y0 = model.pack(Ew0, EΩ0)
# 
#     sol = solve_ivp(
#         model.rhs,
#         z_span,
#         y0,
#         method="DOP853",
#         t_eval=z_eval,
#         rtol=1e-5,
#         atol=1e-8,
#         max_step=(z_span[1] - z_span[0]) / 200
#     )
# 
#     return sol
