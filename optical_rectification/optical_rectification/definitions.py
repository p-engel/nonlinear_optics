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
        t_fwhm=75e-3,
        f0=204, 
    	r_fwhm=2e-3, 
        E=181e-6,
        Nw=2**10
    ):
        """
        t_fwhm  : FWHM in time [ps]
        f0      : carrier frequency [THz]
        r_fwhm  : FWHM of focused dbeam waist [m]
        E       : pulse energy
        """
        # pulse profile parameters
        self.tau = t_fwhm / np.sqrt( 2 * np.log(2) )
        self.delta = 2 / self.tau                                   # 1 / e width in freq.
        self.w0 = 2 * np.pi * f0                                    # [rad / ps]
        self.k0 = self.w0 / c_thz                                   # [rad / m]
        self.lam0 = 2*np.pi / self.k0
        self.w = np.linspace(
        			self.w0 - 5.3*np.pi*self.delta,
        			self.w0 + 3.5*np.pi*self.delta,
        			Nw
        )
        self.detuning = self.w0 - self.w

        self.peak_power = E / (
            self.tau*1e-12 * np.sqrt(np.pi / 2)
        )                                                           # [W]

        # spatial profile parameters
        self.waist0 = r_fwhm / np.sqrt( 2 * np.log(2)  )
        self.zR = np.pi * self.waist0**2 / self.lam0
        # self.A = A
        # self.Aw = np.sqrt(2)*A / self.delta

    def amplitude(self, r, z):
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

    def field_t(self, r, z, t):
        """ t - time, 1d np array [ps] """
        E = self.amplitude(r, z) * (
            np.exp( -1 * (t / self.tau)**2 )
            * np.exp( -1j * self.w0 * t )
        )
        return E

    def field_w(self, r, z):
        E = (
            self.amplitude(r, z) / ( self.delta / np.sqrt(2) )
            * np.exp( -1 * ( self.detuning / self.delta )**2 )
        )
        return E


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
    I = (n * EPS0 * c) * np.abs(A)**2
    return gam3PA * I**2


class Chi2_mixing():
    def __init__(
    	self, E_opt, domega, NΩ, 
    	Dk_plus=2*np.pi, Dk_minus=-2*np.pi, z=0.0
    ):
        self.Ew = E_opt
        self.dw = domega
        self.Dk_plus = Dk_plus
        self.Dk_minus = Dk_minus
        self.z = z
        self.Nw = len(E_opt)
        self.NΩ = NΩ 

    def kernel(self, mode="diff"):
        """
        mode = "diff"    -> K(ω, Ω) = E(ω + Ω)
        mode = "sum"   -> K(ω, Ω) = E(ω - Ω)
        """
        K = np.zeros((self.Nw, self.NΩ), dtype=complex)

        if mode == "diff":
            for l in range(self.Nw):
                max_m = min(self.NΩ, self.Nw - l)
                K[l, :max_m] = self.Ew[l : l + max_m]

            K *= np.exp(1j * self.z * self.Dk_plus)

        elif mode == "sum":
            for l in range(self.Nw):
                max_m = min(self.NΩ, l + 1)                         # Ω ≤ ω
                K[l, :max_m] = self.Ew[l::-1][:max_m]

            K *= np.exp(1j * self.z * self.Dk_minus)

        else:
            raise ValueError("mode must be 'sum' or 'diff'")

        return K

    def correlation(self):
        """Integrates K(ω, Ω) with E*(ω) over w"""
        return self.dw * np.sum(
            self.kernel(mode="diff") 
            * np.conjugate(self.Ew)[:, None], 
            axis=0
        )

    def cascade(self, E_thz):
        """Integrates K(ω, Ω) with E_THz*(Ω) over Ω for diff"""
        K_plus = self.kernel(mode="diff")
        K_minus = self.kernel(mode="sum")

        return self.dw * (
            np.sum(K_plus * np.conjugate(E_thz[None, :]), axis=1)
            + np.sum(K_minus * E_thz[None, :], axis=1)
        )
