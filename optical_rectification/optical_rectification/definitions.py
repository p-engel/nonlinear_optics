import numpy as np
import os
import pandas as pd

from optical_rectification import par
"""
OR Simulation with consistent scaling
-------------------------------------------
- Optical: Sellmeier index
- THz: Lorentz oscillator model
- Propagation with χ² and 3PA
- Uses SciPy Runge-Kutta (solve_ivp)
"""
# constants
c = 299792458.0             # speed of light [m / s]
c_thz = c * 1e-12           # speed of light [m / ps]
TBP = 2*np.log(2) / np.pi   # time-bandwith product
# CHI2 = 428e-12            # [m / V]
CHI2 = 320e-12
DEPTH = 0.37e-3             # crystal length [m]
EPS0 = 8.85e-12  			# permitivity [F/m]
Z0 = 1 / (c * EPS0)         # free space impedance [V/A]
# gam3PA = 0*6e-26			# [m^3/W^2] 3 photon absorption


class Index():
	"""
	Refractive index n(ω) and absorption α(ω) 
	with Lorentz model.
	"""
	def __init__(self, w, param=par.p, s=par.s):
		"""
		Parameters
		----------
		w     : 1d array, frequency grid
		param : tuple, (1 + 3*N,)
				n_inf, w0_1, gam0_1, a_1, ..., w0_N, gam0_N, a_N
					n_inf, real index value at infinity [1]
					w0_n, resonant frequency [rad/ps]
					gam0_n, damping rate [rad/ps]
					a_n, oscillator strength [(rad/ps)^2]
		s     : scaling factor [1 / (rad/ps * m)]
		"""
		self.w = np.array(w)
		self.n_inf = float(param[0])
		self.osc_params = np.array(param)[1:].reshape(-1, 3)
		self.w0 = self.osc_params[:, 0]
		self.gam0 = self.osc_params[:, 1]
		self.a = self.osc_params[:, 2]
		self.s = s
		self.n_osc = len(self.w0)

	def lorentz(self, w0, gam0):
		"""
		Return
		------
		f(ω) : np 1d array [1/Hz^4]
			Lorentz PDF
		"""
		f = gam0*(self.w**2) \
			/ ( (w0**2 - self.w**2)**2 + (gam0**2)*(self.w**2) )
		return f

	def sellmeier(self, lam0=532, q=1.45):
		"""
		lam        : free space wavelength [nm]
		-----
		Return
		n          : refractive index, 1d array [1]
		"""
		lam = 2*np.pi * c_thz / self.w * 1e9                                 # [nm]
		epsillon = (
					self.n_inf**2 
					+ (q * lam0**2) / (lam**2 - lam0**2)
		)
		return np.sqrt(epsillon)

	def n(self):
		"""
		Return
		------
		n : np 1d array
			real refractive index n(ω) [1]
		"""
		n = np.full_like(self.w, self.n_inf, dtype=float)
		for i in range(self.n_osc):
# 			real_part = (
# 				self.a[i] * ( self.w0[i]**2 - self.w**2 ) 
# 				/ ( self.gam0[i]*(self.w**2) )
# 			)
# 			n += real_part * self.lorentz(self.w0[i], self.gam0[i])
			denom = (
				( self.w0[i]**2 - self.w**2 )**2
                + ( self.gam0[i] * self.w )**2
			)
			n += self.a[i] * (self.w0[i]**2 - self.w**2) / denom
		return n

	def alpha(self):
		"""
		Return
		------
		alpha : np 1d array
			imaginary refractive index α(ω) [1/m]
		"""
		alpha = np.zeros_like(self.w, dtype=float)
		for i in range(self.n_osc):
			imag_part = self.s * self.a[i]
			alpha += imag_part * self.lorentz(self.w0[i], self.gam0[i])

		return alpha


class Dispersion():
	"""
	dispersion relation for optical rectification (OR)     [rad/m]
	"""
	def __init__(self, w0, w, n, Ω=[1e-9], n_Ω=[1e-9]):
		"""
		w       : frequency domain of input optical pulse  [rad/ps]
		n(w)    : refractive index in medium               [1]
		Ω       : terahertz domain  Ω << w
		n_Ω     : n(Ω)
		"""
		self.w0 = w0
		self.w = np.array(w); self.Ω = np.array(Ω)
		self.n = np.array(n); self.n_Ω = np.array(n_Ω)
		self.k = self.w * self.n / c_thz
		self.k_Ω = self.Ω * self.n_Ω / c_thz

	def iw0(self, w0): return np.argmin(np.abs(self.w - w0))

	def dk_dw(self, w0=None):
		"""inverse group velocity of input pulse"""
		dk_dw = np.gradient(self.k, self.w)

		if w0 is not None:
			return dk_dw[self.iw0(w0)]
		else:
			return dk_dw

	def ng(self, w0=None):
		"""group index"""
		dn_dw = np.gradient(self.n, self.w)
		ng = self.n + ( self.w * dn_dw )

		return ng

	def beta2(self, w0=None):
		"""group velocity dispersion"""
		beta2 = np.gradient(self.dk_dw(), self.w)

		if w0 is not None: 
			return beta2[self.iw0(w0)]
		else: 
			return beta2
        
	def gvd(self):
		return (
			self.beta2(w0=self.w0)
			* (self.w0  - self.w)**2
		)
		return self.k - k_gvd

	def deltak(self, w0=None):
		"""
		----
		Return
		∆k(Ω)   : approximate OR phase matching condition
					Ω [1/nu_g - 1/nu_Ω] = Ω/c [ng - n_Ω]
					where n is the refractive index
		"""
		if w0 is None:
			return self.Ω[None, :] / c_thz * (
				self.ng()[:, None] - self.n_Ω[None, :]
			)
		else:
			return self.Ω * self.dk_dw(w0=self.w0) - self.k_Ω

	def phase_match(self, plus_branch=True):
		"""
		----
		Return
		∆k(w, Ω)   : exact OR phase matching condition
					k(w + Ω) - k(w) - k(Ω)
		"""
		if plus_branch: w_Ω = self.w[:, None] + self.Ω[None, :]
		else: w_Ω = self.w[:, None] - self.Ω[None, :]

		n_wΩ = Index(w_Ω, param=par.p2, s=par.s2).sellmeier()

		k_Ω = self.k_Ω if plus_branch else (-1  * self.k_Ω)
		k_wΩ = w_Ω * n_wΩ / c_thz
		k_diff = k_wΩ - self.k[:, None] - k_Ω[None, :]
		return k_diff


class Gaussian():
    """
    Wave package with gaussian envelop, 
    propagating sinusoidially at carrier frequency f0
    """
    def __init__(
    	self, 
        duration=75e-3,
        freq=204, 
    	waist=1.699e-3, 
        energy=181e-6,
        Nw=2**10,
        Nr=50
    ):
        """
        duration	: FWHM in time [ps]
        freq		: carrier frequency [THz]
        waist 		: 1/e of focused dbeam waist [m]
        energy		: per pulse [J]
        """
        # pulse profile parameters
        self.tau = duration / np.sqrt( 2 * np.log(2) )
        self.delta = 2 / self.tau
        self.w0 = 2 * np.pi * freq
        self.k0 = self.w0 / c_thz
        self.lam0 = 2*np.pi / self.k0
        self.w = np.linspace(
        			self.w0 - 3.0*np.pi*self.delta,
        			self.w0 + 2.5*np.pi*self.delta,
        			Nw
        )
        self.detuning = self.w0 - self.w

        self.peak_power = energy / (
            self.tau*1e-12 * np.sqrt(np.pi / 2)
        )                                                           # [W]

        # spatial profile parameters
        # self.waist0 = b0 / np.sqrt( 2 * np.log(2)  )
        self.waist0 = waist
        self.zR = np.pi * self.waist0**2 / self.lam0
        self.r = np.linspace(0, 1.7*self.waist0, Nr)
        self.Nr = Nr

    def amplitude(self, r=None, z=0):
        r = self.r if r is None else r
        z = np.atleast_1d(z).astype(float)    
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_R = np.where(
                z == 0, 0.0, 1.0 / (z * (1 + (self.zR / z)**2))
            )

        waist = self.waist0 * np.sqrt( 1 + (z/self.zR)**2 )
        gouy_phase = np.exp( 1j * np.arctan( z / self.zR ) )
        phase_term = np.exp( 1j * self.k0 * r**2 / 2 * inv_R )
        phase = phase_term * gouy_phase
        waist_ratio = self.waist0 / waist
        envelope = np.exp( -1 * (r / waist)**2 )
        A0 = np.sqrt(
            4 * Z0 * self.peak_power / np.pi
        ) / self.waist0

        return  A0 * waist_ratio * envelope * phase

    def field_t(self, t):
        """ t - time, 1d np array [ps] """
        E = self.amplitude() * (
            np.exp( -1 * (t / self.tau)**2 )
            * np.exp( -1j * self.w0 * t )
        )
        return E

    def field_w(self):
        A = (
        	self.amplitude() / (self.delta / np.sqrt(2))
        )[:, None]                                                  # (Nr, 1)
        spectral = np.exp(-1 * (self.detuning / self.delta)**2)     # (Nw,)
        return A * spectral                                         # (Nr, Nw)

    def intensity(self, field_amplitude, n):
        """
        Calculate average intensity
        """
        return  n / (2*Z0) * np.abs( field_amplitude )**2

    def power_spectrum(self, field_w, n=1):
        if field_w.ndim != 2:
            raise ValueError(f"expected 2D array (R, w), got {field_w.ndim}D")
        if field_w.shape[0] != self.Nr:
            raise ValueError(f"first dimension should have R components")
        dr = np.abs( self.r[0] - self.r[1] )
        dEr = np.abs( field_w[0, 0] - field_w[1, 0] )
        if np.abs( dEr / dr ) > 0.1:
            raise ValueError(f"the radial grid spacing is large, the slope of "
            f"E(r) is {dEr/dr}"
            )

        return 2*np.pi * dr * np.sum( 
                self.r[:, None] * self.intensity(field_w, n),
                axis=0 
        )


def chi2_factor(w, n):
    """
    Second-order nonlinear mixing
    freq    : 1d array [rad/ps]
    k       : 1d array, dispersion relation [rad / m]
    return  : [1/V]
    """
    # CHI2 * w**2 / (c_thz**2 * k)
    return CHI2 * w / (np.sqrt(2*np.pi) * c_thz * n)

def three_photon_abs(gam3PA, A, n):
    I = n/(2*Z0) * np.abs(A)**2
    return gam3PA * I**2


class Chi2_mixing():
    """
    Vectorized second-order nonlinear mixing over a radial grid
 
    Ew      : (Nr, Nw) complex array — optical field at each radial slice
    domega  : scalar — frequency spacing [rad/ps]
    NΩ      : int — number of THz frequency points
    Dk_plus : (Nw, NΩ) array — phase mismatch for E(ω+Ω) branch
    Dk_minus: (Nw, NΩ) array — phase mismatch for E(ω-Ω) branch
    z       : scalar — current propagation depth [m]
    """
    def __init__(
    	self, E_opt, domega, NΩ, 
    	Dk_plus=2*np.pi, Dk_minus=-2*np.pi, z=0.0
    ):
        self.Ew = E_opt
        self.dw = domega
        self.Dk_plus = Dk_plus
        self.Dk_minus = Dk_minus
        self.z = z
        self.Nr, self.Nw = E_opt.shape
        self.NΩ = NΩ

        # --- validity masks, shape (Nw, NΩ), computed once at init ---
        #
        # For the diff kernel K[i, l, m] = Ew[i, l+m]:
        #   valid when l + m < Nw
        #
        # For the sum kernel K[i, l, m] = Ew[i, l-m]:
        #   valid when l - m >= 0, i.e. m <= l
        l_idx = np.arange(self.Nw)[:, None]          # (Nw, 1)
        m_idx = np.arange(self.NΩ)[None, :]          # (1, NΩ)
        self.mask_diff = (l_idx + m_idx) < self.Nw   # (Nw, NΩ) True where valid
        self.mask_sum  = m_idx <= l_idx              # (Nw, NΩ) True where valid
 
    def kernel(self, mode="diff"):
        """
        Build the full 3D mixing kernel for all radial slices.
 
        Returns K of shape (Nr, Nw, NΩ) where:
            mode = "diff" -> K[i, l, m] = Ew[i, l+m]   i.e. E(ω + Ω)
            mode = "sum"  -> K[i, l, m] = Ew[i, l-m]   i.e. E(ω - Ω)
 
        Uses as_strided to build the kernel as a strided view of Ew,
        with no data copying
        """
        from numpy.lib.stride_tricks import as_strided
        s = self.Ew.itemsize
 
        if mode == "diff":
            # We want K[i, l, m] = Ew[i, l+m].
            # Strides:
            #   i -> i+1 : jump Nw elements forward = Nw*s bytes
            #   l -> l+1 : jump 1 element forward   = s bytes
            #   m -> m+1 : jump 1 element forward   = s bytes
            K = as_strided(
                self.Ew,
                shape=(self.Nr, self.Nw, self.NΩ),
                strides=(self.Nw * s, s, s)
            )
            # Zero out entries where l+m >= Nw (out of bounds)
            K = np.where(self.mask_diff, K, 0.0)                    # (Nr, Nw, NΩ)
            K = K * np.exp(1j * self.z * self.Dk_plus)
 
        elif mode == "sum":
            # We want K[i, l, m] = Ew[i, l-m].
            # Strides:
            #   i -> i+1 : jump Nw elements forward = Nw*s bytes
            #   l -> l+1 : jump 1 element forward   = s bytes
            #   m -> m+1 : jump 1 element backward  = -s bytes
            #
            # With a negative stride in m, as_strided needs to start reading
            # from Ew[i, l] for m=0, then step backward. Since the base
            # pointer of Ew already points to Ew[i=0, l=0], and we step
            # forward in l by s bytes, K[i, l, 0] = Ew[i, l] as required.
            K = as_strided(
                self.Ew,
                shape=(self.Nr, self.Nw, self.NΩ),
                strides=(self.Nw * s, s, -s)
            )
            K = np.where(self.mask_sum, K, 0.0)
            K = K * np.exp(1j * self.z * self.Dk_minus)
 
        else:
            raise ValueError("mode must be 'sum' or 'diff'")
 
        return K
 
    def correlation(self):
        """
        Computes the OR source term for all radial slices at once:
 		C[i, m] = dw * sum_l  Ew[i, l+m] * Ew*[i, l] * exp(i*z*Dk_plus[l, m])
 
        Returns
        -------
        result : (Nr, NΩ) complex array
        """
        # K shape: (Nr, Nw, NΩ)
        K = self.kernel(mode="diff")
 
        # np.conj(self.Ew) shape: (Nr, Nw)
        # We add [:, :, None] to make it (Nr, Nw, 1) so it broadcasts
        # against K (Nr, Nw, NΩ) along the NΩ axis.
        # Then sum over l (axis=1) to get (Nr, NΩ).
        return self.dw * np.sum(
            K * np.conj(self.Ew)[:, :, None],
            axis=1
        )
 
    def cascade(self, EΩ):
        """
        Computes the cascade back-action on the optical field for all
        radial slices at once.
 
        (two branches):
            plus  branch: sum_m  Ew[i, l+m] * EΩ*[i, m] * exp(i*z*Dk_plus[l, m])
            minus branch: sum_m  Ew[i, l-m] * EΩ[i, m]  * exp(i*z*Dk_minus[l, m])
 
        Parameters
        ----------
        EΩ : (Nr, NΩ) complex array — current THz field
 
        Returns
        -------
        result : (Nr, Nw) complex array
        """
        K_plus  = self.kernel(mode="diff")      # (Nr, Nw, NΩ)
        K_minus = self.kernel(mode="sum")       # (Nr, Nw, NΩ)
 
        # np.conj(EΩ) shape: (Nr, NΩ)
        # We add [:, None, :] to make it (Nr, 1, NΩ) so it broadcasts
        # against K (Nr, Nw, NΩ) along the Nw axis.
        # Then sum over m (axis=2) to get (Nr, Nw).
        return self.dw * (
            np.sum(K_plus  * np.conj(EΩ)[:, None, :], axis=2)
          + np.sum(K_minus * EΩ[:, None, :], axis=2)
        )
