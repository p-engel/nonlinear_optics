# propagator.py
from numpy import linspace, zeros_like, concatenate
from . import par
from .definitions import three_photon_loss, chi2_factor, Chi2_mixing, \
Dispersion, Index

class ORPropagator:
    def __init__(self, pulse, Ω_max, NΩ, cascade=True):
        self.dw = abs(pulse.w[1] - pulse.w[0])          # freq per step/spacing
        self.m_dps = int(Ω_max / self.dw)               # no. of steps
        self.Nw = len(pulse.w)
        self.NΩ = self.m_dps + 1
        self.Ω = linspace(
            1, self.m_dps, self.NΩ
        ) * self.dw

        self.index_w = Index(
            pulse.w, param=par.param_op, k=1
        )
        self.index_Ω = Index(
            self.Ω, param=par.param_thz
        )

        self.dispersion = Dispersion(
            pulse.w, self.index_w.sellmeier(), 
            Ω=self.Ω, n_Ω=self.index_Ω.n()
        )

        self.pref_w = chi2_factor(
            pulse.w, self.dispersion.k
        )
        self.pref_Ω = chi2_factor(
            self.Ω, self.dispersion.k_Ω
        )

        self.cascade = cascade

        self.Ew0 = pulse.field_w()
        self.EΩ0 = zeros_like(self.Ω, dtype=complex)

    def pack(self, Ew, EΩ):
        return concatenate([Ew, EΩ])

    def unpack(self, y):
        return y[:self.Nw], y[self.Nw:]

    def rhs(self, z, y):
        Ew, EΩ = self.unpack(y)

        chi2_mixing = Chi2_mixing(
            Ew, self.dw, self.NΩ,
            Dk_up=self.dispersion.phase_match(),
            Dk_dwn=self.dispersion.phase_match(conj=True),
#             Dk_up=self.dispersion.deltak(),
#             Dk_dwn=-self.dispersion.deltak(),
            z=z
        )

        # --- terahertz field ode ---
        dEΩ = (
            -0.5 * self.index_Ω.alpha() * EΩ
            -0.5j * self.pref_Ω * chi2_mixing.correlation()
        )
        # --- optical field ode ---
        dEw = -0.5 * (
            self.index_w.alpha() + 
            three_photon_loss(Ew, self.index_w.n()) 
        ) * Ew
        if self.cascade:
            dEw += -0.5j * self.pref_w * chi2_mixing.cascade(EΩ)

        return self.pack(dEw, dEΩ)
