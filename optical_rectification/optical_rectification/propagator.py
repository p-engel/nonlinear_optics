# propagator.py
from numpy import arange, floor, linspace, zeros_like, concatenate, pi
from . import par
from .definitions import three_photon_abs, chi2_factor, Chi2_mixing, \
Dispersion, Index, DEPTH

class ORPropagator:
    def __init__(self, t_fwhm, f0, U, gam3PA, Ω_max=2*pi*10, cascade=True):
        """
        Parameters
        ----------
        t_fwhm      : pulse duration [ps]
        f0          : pulse carrier frequency [THz]
        U           : pulse energy
        gam3PA      : three-photon absorption coefficient [m^3/W^2]
        """
        self.pulse = Gaussian(duration=t_fwhm, freq=f0, energy=U)
        self.dw = abs(self.pulse.w[1] - pulse.w[0])
        self.m_dps = int(floor(Ω_max / self.dw))
        self.Ω = arange(self.m_dps + 1) * self.dw
        self.NΩ = len(self.Ω)
        self.Nw = len(pulse.w)

        # --- spectral / material properties ---
        self.index_w = Index(pulse.w, param=par.p2, s=par.s2)
        self.index_Ω = Index(self.Ω)

        self.dispersion = Dispersion(
            pulse.w0, pulse.w, self.index_w.sellmeier(), 
            Ω=self.Ω, n_Ω=self.index_Ω.n()
        )

        self.pref_w = chi2_factor(pulse.w, self.index_w.sellmeier())  # (Nw,)
        self.pref_Ω = chi2_factor(self.Ω, self.index_Ω.n())           # (NΩ,)
        self.pref_Ω[0] = 0.0

        self.cascade = cascade
        self.DEPTH = DEPTH

        # --- transverse direction initial conditions ---
        # pulse.field_w(r, z) returns 
        # amplitude(r,z) * spectral_envelope(w)
        self.Ew0 = self.pulse.field_w(self.r[:, None], 0)             # (Nr, Nw)
        self.EΩ0 = zeros((Nr, self.NΩ), dtype=complex)                # (Nr, NΩ)
        A_r = self.pulse.amplitude(self.r, 0)                         # (Nr,)
        self.tpa = three_photon_abs(gam3PA, A_r, 1)                   # (Nr,)

    def pack(self, Ew, EΩ):
        """
        Flatten the 2D fields into a single 1D vector for solve_ivp
        """
        return concatenate([Ew.ravel(), EΩ.ravel()])
 
    def unpack(self, y):
        """
        Reverse pack: split and reshape the flat vector to 2D arrays
        """
        Ew = y[:self.Nr * self.Nw].reshape(self.Nr, self.Nw)
        EΩ = y[self.Nr * self.Nw:].reshape(self.Nr, self.NΩ)
        return Ew, EΩ

    def rhs(self, z, y):
        """
        Right-hand side of the coupled ODEs, 
        evaluated at propagation depth z
        All operations are vectorized over the Nr radial slices
        """
        Ew, EΩ = self.unpack(y)

        chi2 = Chi2_mixing(
            Ew, self.dw, self.NΩ,
#             Dk_plus=self.dispersion.phase_match(),
#             Dk_minus=self.dispersion.phase_match(conj=True),
            Dk_plus=self.dispersion.deltak(w0=1),
            Dk_minus=-self.dispersion.deltak(w0=1),
            z=z
        )

        # --- terahertz field ode ---
        dEΩ = (
            -0.5 * self.index_Ω.alpha() * EΩ
            +0.5j * self.pref_Ω * chi2.correlation()
        )
        # --- optical field ode ---
        dEw = (
            -0.5 * ( self.index_w.alpha() + self.tpa )
            -0.5j * self.dispersion.gvd()
        ) * Ew
        if self.cascade:
#             EΩ_eff = EΩ.copy()
# #             EΩ_eff[0] = 0.0
            dEw += 0.5j * self.pref_w * chi2.cascade(EΩ)

        return self.pack(dEw, dEΩ)