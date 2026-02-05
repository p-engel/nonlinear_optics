# propagator.py
from numpy import arange, zeros_like, concatenate, abs
from . import par
from .definitions import three_photon_loss, chi2_factor, Chi2_mixing, \
Dispersion, Index

class ORPropagator:
    def __init__(self, Ω_max, pulse, cascade=True):
        self.pulse = pulse
        self.w = pulse.w
        self.dw = self.w[1] - self.w[0]
        self.Ω = arange(1, int(Ω_max/self.dw) + 1) * self.dw

        self.index_Ω = Index(self.Ω, param=par.param_thz)
        self.index_w = Index(self.w, param=par.param_op, k=1)

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
        dEw = -0.5 * (
            self.alpha_w + three_photon_loss(Ew, self.index_w.n()) 
        ) * Ew
        if self.cascade:
            dEw += -0.5j * self.pref_w * chi2_mixing.cascade(EΩ)

        return self.pack(dEw, dEΩ)
