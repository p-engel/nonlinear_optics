import numpy as np
import os
import pandas as pd
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


class Index():
	"""
	Refractive index n(ω) and absorption α(ω) 
	with Lorentz model.
	"""
	def __init__(self, omega, omega0, gamma, osc_strength, n_inf, k):
		"""
		Parameters
		----------
		omega : np 1d array [rad/s]
		omega0 : np 1d array
			resonant frequencies for n oscillators [rad/s]
		gamma : np 1d array
			damping rates for n oscillators [rad/s]
		osc_strength : np 1d array
			for n oscillators [(rad/s)^2]
		n_inf : float
			real index value at infinity [1] 
		k : float
			global absorption scaling factor [1/(rad/s)•1/mm]   
		"""
		self.w = omega; self.w0 = omega0; self.gamma = gamma
		self.a = osc_strength
		self.n_inf = n_inf; self.k = k
		self.n_osc = len(omega0)

	def lorentz(self, w0, gam0):
		"""
		Parameters
		----------
		w0, gam0 : float [rad/s]
		Return
		------
		f(ω) : np 1d array [1/(rad/s)^4]
			Lorentz PDF
		"""
		f = gam0*(self.w**2) \
			/ ( (w0**2 - self.w**2)**2 + (gam0**2)*(self.w**2) )
		return f

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
				( self.w0[i]**2 - self.w**2 ) / ( self.gamma[i]*(self.w**2) )
			n += real_part*self.lorentz(self.w0[i], self.gamma[i])

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
			alpha += imag_part*self.lorentz(self.w0[i], self.gamma[i])

		return alpha


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
		wavelen = df['wl'].values * 1e-9  # [m]
		alpha = df['alpha'].values
		spectrum = np.array(
					[[c/wl, alpha[k]] for k, wl in enumerate(wavelen)]
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
		w_min = 200e12; w_max = 800e12;  # Hz
		spec = self.read_spec(); w = spec[:,0]
		opt_spec = spec[self.find_ind(w, w_min):self.find_ind(w, w_max), :]
		return opt_spec
