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
    __init__():
    	return

    def lorentz(omega, omega0, gamma):
		"""
		Parameters
	    ----------
	    omega : np 1d array [rad/s]
	    omega0 : float [rad/s]
	    gamma : float [rad/s]
		Return
	    ------
	    f(ω) : np 1d array [1/(rad/s)^4]
	    	un-normalized Lorentz PDF
	    """
		f = 1 / (omega0**2 - omega**2)**2 + 4*(gamma**2)*(omega**2)
		return f

	def n(omega, omega0, gamma, osc_strength, n_inf):
		"""
		Parameters
	    ----------
	    omega : np 1d array [rad/s]
	    omega0 : np 1d array
	    	for n oscillators [rad/s]
	    gamma : np 1d array
	    	for n oscillators [rad/s]
	    osc_strength : np 1d array
	    	for n oscillators [(rad/s)^2]
	    n_inf : float
	    	index value at infinity [1]
	    Return
	    ------
	    n : np 1d array
	    	real refractive index n(ω) [1]
	    """
	    n = np.full_like(omega, n_inf, dtype=float)
	    for i in self.n_osc:
	    	real_part = osc_strength[i] * (omega0[i]**2 - omega**2)
	    	n += real_part * lorentz(omega, omega0[i], gamma[i])

	    return n

	def alpha(omega, omega0, gamma, osc_strength, k):
		"""
		Parameters
	    ----------
	    omega : np 1d array [rad/s]
	    omega0 : np 1d array
	    	for n oscillators [rad/s]
	    gamma : np 1d array
	    	for n oscillators [rad/s]
	    osc_strength : np 1d array
	    	for n oscillators [(rad/s)^2]
	    k : float
	    	Global absorption scaling factor [?]
	    Return
	    ------
	    alpha : np 1d array
	    	imaginary refractive index α(ω) [1]
	    """
	    alpha = np.zeros_like(omega)
	    for i in self.n_osc:
	    	imag_part = k * osc_strength[i] * (2*gamma[i]*(omega**2))
	    	alpha += imag_part * lorentz(omega, omega0[i], gamma[i])

	    return alpha

	def lorentz(omega, omega0, gamma, osc_strength, k):
		"""
		Parameters
	    ----------
	    omega : np 1d array [rad/s]
	    omega0 : float [rad/s]
	    gamma : float [rad/s]
	    osc_strength : float [(rad/s)^2]
	    k : float [?]
	    	Global absorption scaling factor
		Return
	    ------
	    real : np 1d array [1]
	    	real dispersion n(ω)
	    imag : np 1d array [?]
	    	imaginary dispersion α(ω)
	    """
		D = (omega0**2 - omega**2)**2 + 4*(gamma**2)*(omega**2)
		real = osc_strength * (omega0**2 - omega**2) / D
		imag = k * osc_strength * (2*gamma*omega**2) / D
		return real, imag



	def lorentz_imag(omega, omega0, gamma, osc_strength, k):
		"""
		Parameters
	    ----------
	    omega : np 1d array [rad/s]
	    omega0 : float [rad/s]
	    gamma : float [rad/s]
	    osc_strength : float [(rad/s)^2]
	    K : float [?]
	    	Global absorption scaling factor
		Return
	    ------
	    alpha : np 1d array [?]
	    	imaginary refractive index α(ω)
	    """
		D = (omega0**2 - omega**2)**2 + 4*(gamma**2)*(omega**2)
		alpha = k * (omega**2) * (2*gamma*osc_strength) / D
		return alpha

class Index():
	"""
    Calculate refractive index n(ω) and absorption α(ω) 
    with Lorentz model.

    Parameters
    ----------
    omega : array
        Angular frequency grid [rad/s].
    params_n : array-like
        Parameters from the n(ω) fit:
        [n_inf, f1,g1,a1, f2,g2,a2, ..., f8,g8,a8]
        with f,g in THz and a in THz^2.
    K : float
        Global absorption scaling factor.
    """


def dispersion(omegas, params_n, K):
    """
    Calculate refractive index n(ω) and absorption α(ω) 
    with Lorentz model.

    Parameters
    ----------
    omega : array
        Angular frequency grid [rad/s].
    params_n : array-like
        Parameters from the n(ω) fit:
        [n_inf, f1,g1,a1, f2,g2,a2, ..., f8,g8,a8]
        with f,g in THz and a in THz^2.
    K : float
        global absorption scaling factor.

    Returns
    -------
    n_fit : array
        Refractive index as function of ω.
    alpha_fit : array
        Absorption [1/mm] as function of ω.
    """
    n_inf = params_n[0]  # index at infinity
    n_par = 3	# no. parameters per oscillator
    n_osc = len(params_n[1:])/n_par # no. of oscillators
    scale = 2*np.pi*1e12
    osc_params = np.array(
    	params_n[1:]).reshape(n_osc, n_par)
    fs = scale*osc_params[:,1]  # natural resonance [rad/s]
    gs = scale*osc_params[:,2]  # damping [rad/s]
    as = scale*osc_params[:,3]  # oscillator strength

	n = np.full_like(omegas, n_inf, dtype=float)
    alpha = np.zeros_like(omegas, dtype=float)
    for fi, gi, ai in osc_params:
        omega_i = 2*np.pi*fi*1e12   # resonance [rad/s]
        gamma_i = 2*np.pi*gi*1e12   # damping [rad/s]
        ai_si = ai * (2*np.pi*1e12)**2   # convert THz^2 → (rad/s)^2
        D = (omega_i**2 - omegas**2)**2 + 4*(gamma_i**2)*(omegas**2)
		n += ai_si * (omega_i**2 - omega**2) / D
		alpha += K * (omega**2) * (2*gamma_i*ai_si) / D

    return n, alpha