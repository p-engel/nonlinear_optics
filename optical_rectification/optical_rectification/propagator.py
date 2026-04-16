# propagator.py
from numpy import arange, floor, linspace, zeros_like, concatenate, pi
from . import par
from .definitions import three_photon_loss, chi2_factor, Chi2_mixing, \
Dispersion, Index, DEPTH

class ORPropagator:
    def __init__(self, pulse, Ω_max=2*pi*10, cascade=True):
        self.dw = abs(pulse.w[1] - pulse.w[0])          # freq per step/spacing
        self.m_dps = int(floor(Ω_max / self.dw))     # no. of steps
        self.Ω = arange(self.m_dps + 1) * self.dw
        self.NΩ = len(self.Ω)
        self.Nw = len(pulse.w)
#         self.NΩ = self.m_dps + 1
#         self.Ω = linspace(
#             1, self.m_dps, self.NΩ
#         ) * self.dw

        self.index_w = Index(
            pulse.w, param=par.p2, s=par.s2
        )
        self.index_Ω = Index(self.Ω)

        self.dispersion = Dispersion(
            pulse.w0, pulse.w, self.index_w.sellmeier(), 
            Ω=self.Ω, n_Ω=self.index_Ω.n()
        )

        self.pref_w = chi2_factor(
            pulse.w, self.index_w.sellmeier()
        )
        self.pref_Ω = chi2_factor(
            self.Ω, self.index_Ω.n()
        )
        self.pref_Ω[0] = 0.0

        self.cascade = cascade

        self.Ew0 = pulse.field_w()
        self.EΩ0 = zeros_like(self.Ω, dtype=complex)

        self.DEPTH = DEPTH

    def pack(self, Ew, EΩ):
        return concatenate([Ew, EΩ])

    def unpack(self, y):
        return y[:self.Nw], y[self.Nw:]

    def rhs(self, z, y):
        Ew, EΩ = self.unpack(y)

        chi2_mixing = Chi2_mixing(
            Ew, self.dw, self.NΩ,
#             Dk_up=self.dispersion.phase_match(),
#             Dk_dwn=self.dispersion.phase_match(conj=True),
            Dk_up=self.dispersion.deltak(),
            Dk_dwn=-self.dispersion.deltak(),
            z=z
        )

        # --- terahertz field ode ---
        dEΩ = (
            -0.5 * self.index_Ω.alpha() * EΩ
            +0.5j * self.pref_Ω * chi2_mixing.correlation()
        )
        # --- optical field ode ---
        dEw = (
            -0.5 * self.index_w.alpha()
            -0.5j * self.dispersion.gvd()
        ) * Ew
        if self.cascade:
            EΩ_eff = EΩ.copy()
            EΩ_eff[0] = 0.0
            dEw += 0.25j * self.pref_w * chi2_mixing.cascade(EΩ_eff)

        return self.pack(dEw, dEΩ)
