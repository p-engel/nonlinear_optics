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
c = 299792458.0             	# speed of light [m * Hz]
c_thz = c * 1e-12           	# speed of light [m * THz]
p = np.array(par.param_thz[1:])
TBP = 2*np.log(2) / np.pi   	# time-bandwith product
CHI2 = 428e-12              	# [m / V]
DEPTH = 0.4e-3              	# crystal length [m]


class Index():
	"""
	Refractive index n(ω) and absorption α(ω) 
	with Lorentz model.
	"""
	def __init__(self, w, 
			w0=None, gam0=None, a=None, n_inf=None, k=None
			):
		"""
		Parameters
		----------
		w : np 1d array 
			free space wavelength or frequency
		w0 : np 1d array
			resonant frequencies for n oscillators [Hz]
		gam0 : np 1d array
			damping rates for n oscillators [Hz]
		a : np 1d array
			oscillator strenght for n oscillators [Hz^2]
		n_inf : float
			real index value at infinity [1]
		k : float
			global absorption scaling factor [1/(Hz*mm) ?]
		"""
		self.w = w;
		self.n_inf = par.param_thz[0] if n_inf is None else n_inf;
		self.w0 = p[:,0] if w0 is None else w0; 
		self.gam0 = p[:,1] if gam0 is None else gam0;
		self.a = p[:,2] if a is None else a;
		self.k = par.k if k is None else k
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

	def sellmeier(self, lam0=455, q=0.17):
		"""
		lam        : free space wavelength [nm]
		-----
		Return
		n          : refractive index, 1d array [1]
		"""
		lam = np.sort( c_thz / self.w * 1e9 )                         # [nm]
		epsillon = (
					self.n_inf**2 
					+ (q * lam0**2) / (lam[::-1]**2 - lam0**2)
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
			real_part = (
				self.a[i] * ( self.w0[i]**2 - self.w**2 ) 
				/ ( self.gam0[i]*(self.w**2) )
			)
			n += real_part * self.lorentz(self.w0[i], self.gam0[i])

		return n

	def alpha(self):
		"""
		Return
		------
		alpha : np 1d array
			imaginary refractive index α(ω) [1/mm]
		"""
		alpha = np.zeros_like(self.w, dtype=float)
		for i in range(self.n_osc):
			imag_part = self.k * self.a[i]
			alpha += imag_part * self.lorentz(self.w0[i], self.gam0[i])

		return alpha


class Dispersion():
	"""
	dispersion relation for optical rectification (OR)
	"""
	def __init__(self, w, n, Ω=None, n_Ω=None):
		"""
		w       : frequency domain of input optical pulse  [THz]
		n(w)    : refractive index in medium
		Ω       : terahertz domain  Ω << w  [THz]
		n_Ω     : n(Ω)
		"""
		self.w = np.array(w)
		self.Ω = np.array(Ω) if Ω is not None else np.array([1e-9]) 
		self.n = np.array(n)
		self.n_Ω = np.array(n_Ω)
		self.nu = c_thz / self.n                    # phase velocity
		self.nu_Ω = c_thz / self.n_Ω if n_Ω is not None else 1e-9
		self.k = self.w * (1 / self.nu)
		self.k_Ω = self.Ω * (1 / self.nu_Ω)

	def nu_g(self, w0=None):
		"""group velocity"""
		ng = self.n + (self.w * np.gradient(self.n, self.w))

		if w0 is not None: 
			iw0 = np.argmin(np.abs(self.w - w0))
			return c_thz / ng[iw0]
		else: 
			return c_thz / ng

	def deltak(self, w0=None):
		"""
		----
		Return
		∆k(Ω)   : approximate OR phase matching condition
					Ω [1/nu_g - 1/nu_Ω] = Ω/c [ng - n_Ω]
					where n is the refractive index
		"""
		if w0 is not None:
			return self.Ω * (1/self.nu_g(w0=w0) - 1/self.nu_Ω)
		else:
			return self.Ω * (1/self.nu_g()[:, None] - 1/self.nu_Ω) 

	def phase_match(self):
		"""
		----
		Return
		∆k(w, Ω)   : exact OR phase matching condition
					k(w + Ω) - k(w) - k(Ω)
		"""
		w_Ω = self.w[:, None] + self.Ω

		n_wΩ = Index(w_Ω).n()

		k_wΩ = w_Ω * n_wΩ / c_thz
		k_diff = k_wΩ - self.k[:, None]
		k_diff = k_diff - k_diff[:, 0][:, None]  # k(w+Ω=0) - k(w) = 0;
		return k_diff - self.k_Ω


class Gaussian():
    """
    Wave package with gaussian envelop, 
    propagating sinusoidially at carrier frequency
    """
    def __init__(self, t_fwhm=None, w0=None, E0=None):
        """
        t_fwhm  : full width at half maximum in time [s]
        w0      : carrier frequency [Hz]
        E       : peak intensity in time [V/m]
        """
        self.tau = np.sqrt(2) * (t_fwhm) / ( 2 * np.sqrt(np.log(2)) )
        self.delta = 2 / self.tau  # 1 / e width in frequency domain
        self.w0 = w0
        self.E0 = E0
        self.E0_w = E0 * np.sqrt(np.pi) * 2 / self.delta
        return
    
    def field_t(self, t):
        """ t - time 1d np array [s] """
        E = self.E0 * np.exp( -1 * (np.array(t) / self.tau)**2 ) \
            * np.exp( -1j*self.w0*np.array(t) )
        return E

    def field_w(self, w):
        """ w - frequency 1d np array [Hz] """
        E = self.E0_w \
            * np.exp( -1 *
            	( 2*np.pi * (self.w0 - np.array(w)) / self.delta )**2 
            )
        return E


def chi2_factor(freq, k):
    """
    Second-order nonlinear mixing
    freq : 1d array [THz]
    k    : 1d array, dispersion relation [1 / m]
    """
    return CHI2 * freq**2 / (c_thz**2 * k)              # [1 / V]


class Chi2_mixing():
    def __init__(self, E_opt, domega, phase_match=None, z=0.0):
        self.Ew = E_opt
        self.dw = domega
        self.Dk = phase_match
        self.z = z
        self.Nw = len(E_opt)
        self.NΩ = self.Dk.shape[1] if self.Dk is not None else self.Nw

        if self.Dk is not None:
            assert self.Dk.shape == (self.Nw, self.NΩ)

    def kernel(self, mode="sum"):
        """
        mode = "sum"    -> K(ω, Ω) = E(ω + Ω)
        mode = "diff"   -> K(ω, Ω) = E(ω - Ω)
        """
        K = np.zeros((self.Nw, self.NΩ), dtype=complex)

        if mode == "sum":
            for l in range(self.Nw):
                max_m = min(self.NΩ, self.Nw - l)
                K[l, :max_m] = self.Ew[l : l + max_m]

        elif mode == "diff":
            for l in range(self.Nw):
                max_m = min(self.NΩ, l + 1)                         # Ω ≤ ω
                K[l, :max_m] = self.Ew[l::-1][:max_m]

        else:
            raise ValueError("mode must be 'sum' or 'diff'")

        if self.Dk is not None:
            K *= np.exp(-1j * self.z * self.Dk)

        return K

    def correlation(self):
        """Integrates K(ω, Ω) with E*(ω) over w"""
        return self.dw * np.sum(
            self.kernel(mode="sum") * np.conjugate(self.Ew)[:, None], 
            axis=0
        )

    def cascade(self, E_thz):
        """Integrates K(ω, Ω) with E_THz*(Ω) over Ω"""
        Kup = self.kernel(mode="sum")
        Kdwn = self.kernel(mode="diff")

        return self.dw * (
            np.sum(Kup * np.conjugate(E_thz[None, :]), axis=1) +
            np.sum(Kdwn * E_thz[None, :], axis=1)
        )
