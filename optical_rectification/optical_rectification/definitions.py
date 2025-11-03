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
c = 299792458.0  # speed of light [m/s]
p = np.array(par.param[1:])


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
		self.n_inf = par.param[0] if n_inf is None else n_inf;
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

	def sellmeier(self, n_inf, w0, q):
		"""
		"""
		epsillon = n_inf**2 + (q * w0**2)/(self.w**2 - w0**2)
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
			real_part = self.a[i] * \
				( self.w0[i]**2 - self.w**2 ) / ( self.gam0[i]*(self.w**2) )
			n += real_part*self.lorentz(self.w0[i], self.gam0[i])

		return n

	def alpha(self):
		"""
		Return
		------
		alpha : np 1d array
			imaginary refractive index α(ω) [1/mm]
		"""
		alpha = np.zeros_like(self.w)
		for i in range(self.n_osc):
			imag_part = self.k*self.a[i]
			alpha += imag_part*self.lorentz(self.w0[i], self.gam0[i])

		return alpha


class Dispersion():
	"""
	dispersion relation of field in dielectric medium
	"""
	def __init__(self,w,n):
		"""
		w - 1d array: floats
			spectral frequency domain
		n(w) - 1d arrya: floats
			refractive index of medium
		"""
		self.w = np.array(w); self.n = np.array(n);

	def phase_velocity(self): return c/self.n

	def k(self): return self.w/self.phase_velocity()


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
		spectrum = np.array([[wl, alpha[k]] for k, wl in enumerate(wavelen)])
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
		opt_spec = spec[self.find_ind(w, w_min):self.find_ind(w, w_max), :]
		return opt_spec