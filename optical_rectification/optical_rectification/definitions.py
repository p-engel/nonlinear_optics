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
	def __init__(self, w, n, Ω=[0], n_Ω=[0]):
		"""
		w       : frequency domain of input optical pulse  [THz]
		n(w)    : refractive index in medium
		Ω       : terahertz domain  Ω << w  [THz]
		n_Ω     : n(Ω)
		"""
		self.w = np.array(w)
		self.Ω = np.array(Ω)
		self.n = np.array(n)
		self.n_Ω = np.array(n_Ω)
		self.nu = c_thz / self.n                    # phase velocity
		self.nu_Ω = c_thz / self.n_Ω
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

	def phase_match(self, conj=False):
		"""
		----
		Return
		∆k(w, Ω)   : exact OR phase matching condition
					k(w + Ω) - k(w) - k(Ω)
		"""
		if not conj: w_Ω = self.w[:, None] + self.Ω
		else: w_Ω = self.w[:, None] - self.Ω; self.k_Ω = -self.k_Ω

		# M = len(self.Ω)
		n_wΩ = Index(w_Ω).n()

		# k_wΩ = w_Ω * n_wΩ.T / c_thz
		k_wΩ = w_Ω * n_wΩ / c_thz
		k_diff = k_wΩ - self.k[:, None]
		k_diff = k_diff - k_diff[:, 0][:, None]  # k(w+Ω=0) - k(w) = 0;
		return k_diff - self.k_Ω


class Spectrum():
	"""
	read and graph spectrum
	"""
	def __init__(self, filename):
		self.f = filename

	def find_ind(self, arr, value):
		arr = np.array(arr)
		idx = (np.abs(arr - value)).argmin()
		return idx

	def read_spec(self):
		df = pd.read_csv(self.f, header=None, names=['wl', 'alpha'])
		wavelen = df['wl'].values  # [nm]
		alpha = df['alpha'].values
		spectrum = np.array(
			[[wl, alpha[k]] for k, wl in enumerate(wavelen)]
		)
		return spectrum[spectrum[:,0].argsort()]  # sorted spectrum

	def opt_spec(self):
		"""
		Read absorption spectrum and return truncated spectrum
		in the optical region: 150 - 800 THz
		Return 
		spectrum : numpy array (N, 2)
		"""
		# constant
		w_min = c/200; w_max = c/800;  # [nm]
		spec = self.read_spec(); w = spec[:,0]
		opt_spec = spec[
			self.find_ind(w, w_min):self.find_ind(w, w_max), :
		]
		return opt_spec


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
        self.E0_w = E0 * np.sqrt(np.pi) * self.tau
        return
    
    def field_t(self, t):
        """ t - time 1d np array [s] """
        E = self.E0 * np.exp( -1*np.array(t)**2 / self.tau**2 ) \
            * np.exp( -1j*self.w0*np.array(t) )
        return E

    def field_w(self, w):
        """ w - frequency 1d np array [Hz] """
        E = self.E0_w \
            * np.exp( 
            	-1 * ( 2*np.pi*(self.w0 - np.array(w)) )**2 
            	/ self.delta**2 
            )
        return E


def corr(E, domega, m, up=True):
    """
    E       : complex 1D array on uniform grid E(w)
    domega  : grid spacing dw
    m       : integer shift index, Ω_m = m * dw

    returns : Riemann sum ∫ E(w + Ω_m) * conjugate[E(w)] dw,
              else ∫ E(w - Ω_m) * E(w) dw, if down=True
    """
    if m < 0: raise ValueError("m must be non-negative")

    N = len(E)
    if m >= N: return 0.0

    if up:
        # Up shift/conversion
        return domega * np.sum(E[m:] * np.conjugate(E[:N - m]))
    elif not up:
        # Down shift/conversion
        return domega * np.sum(E[:N-m] * E[m:])
 

def chi2_factor(freq, k):
    """
    Second-order nonlinear mixing
    freq : 1d array [THz]
    k    : 1d array, dispersion relation [1 / m]
    """
    return CHI2 * freq**2 / (c_thz**2 * k)              # [1 / V]


def chi2_mixing(E, domega, m, Dk, z, up=True, E_conj=None):
    """
    Phase-matched correlation integral

    Input
    ------
    E       : optical field E(ω)
    E_conj  : conjugate field if cross correlation
    domega  : frequency spacing
    m       : integer shift index, Ω_m = m * Δω
    Dk      : 2D array Δk(ω, Ω), shape (Nω, NΩ)
    z       : propagation distance
    up      : up/down conversion

    Returns
    -------
    ∫ E(ω±Ω) E*(ω) exp(-i z Δk) dω
    """
    if m < 0: raise ValueError("m must be non-negative")

    N = len(E)
    if m >= N: return 0.0
    if E_conj is None: E_conj = np.conj(E)

    phase = np.exp(-1j * z * Dk[:N-m, m])

    if up:
        integrand = E[m:] * E_conj[:N-m] * phase
    else:
        integrand = E[:N-m] * np.conj(E_conj[m:]) * phase

    return domega * np.sum(integrand)
