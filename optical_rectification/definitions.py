import numpy as np
"""
OR Simulation with consistent scaling
-------------------------------------------
- Optical: Sellmeier index
- THz: Lorentz oscillator model
- Propagation with χ² and 3PA
- Uses SciPy Runge-Kutta (solve_ivp)
"""
class Index():
	"""
	Refractive index n(ω) and absorption α(ω) 
	with Lorentz model.
	"""
	def __init__(omega, omega0, gamma, osc_strength, n_inf, k):
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
			global absorption scaling factor [?]   
		"""
		self.w = omega; self.w0 = omega0; self.gamma = gamma
		self.a = osc_strength
		self.n_inf = n_inf; self.k = k
		self.n_osc = len(omega0)

	def lorentz(w0, gam0):
		"""
		Parameters
		----------
		w0, gam0 : float [rad/s]
		Return
		------
		f(ω) : np 1d array [1/(rad/s)^4]
			un-normalized Lorentz PDF
		"""
		f = 1 / (w0**2 - self.w**2)**2 + 4*(gam0**2)*(self.w**2)
		return f

	def n(self):
		"""
		Return
		------
		n : np 1d array
			real refractive index n(ω) [1]
		"""
		n = np.full_like(self.w, self.n_inf, dtype=float)
		for i in self.n_osc:
			real_part = self.a[i] * (self.w0[i]**2 - self.w**2)
			n += real_part * lorentz(self.w0[i], self.gamma[i])

		return n

	def alpha(self):
		"""
		Return
		------
		alpha : np 1d array
			imaginary refractive index α(ω) [?]
		"""
		alpha = np.zeros_like(omega)
		for i in self.n_osc:
			imag_part = self.k * self.a[i] * (2*self.gamma[i]*(self.w**2))
			alpha += imag_part * lorentz(self.w0[i], self.gamma[i])

		return alpha