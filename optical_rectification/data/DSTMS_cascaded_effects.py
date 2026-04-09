# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 14:57:33 2025

@author: mlawe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 10:47:30 2025

@author: malte
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSTMS OR Simulation with consistent scaling
-------------------------------------------
- Optical: Sellmeier index
- THz: Lorentz oscillator model
- Propagation with χ² and 3PA
- Uses SciPy Runge-Kutta (solve_ivp)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit


import math
import sys


# ----------------------------------------------------
#Lorentz model for n Thz
# ----------------------------------------------------
def lorentz_n_only(omega, n_inf,
                   f1,g1,a1, f2,g2,a2, f3,g3,a3,
                   f4,g4,a4, f5,g5,a5, f6,g6,a6,
                   f7,g7,a7
                   # , f8,g8,a8
                   ):
    n_fit = np.full_like(omega, n_inf)
    conv_sq = (2*np.pi*1e12)**2
    params = [(f1,g1,a1), (f2,g2,a2), (f3,g3,a3), (f4,g4,a4),
               (f5,g5,a5), (f6,g6,a6), (f7,g7,a7)
               # , (f8,g8,a8)
              ]
    for fi, gi, ai in params:
        omega_i = 2*np.pi*fi*1e12
        gamma_i = gi*1e12
        ai_si = ai * conv_sq
        denom = (omega_i**2 - omega**2)**2 + 4*(gamma_i**2)*(omega**2)
        n_fit += ai_si * (omega_i**2 - omega**2) / denom
    return n_fit

# ----------------------------------------------------
# Lorentz model for absoprtion THz 
# ----------------------------------------------------
def lorentz_alpha_only(omega, K, *osc_params):
    conv_sq = (2*np.pi*1e12)**2
    alpha_fit = np.zeros_like(omega)
    params = np.array(osc_params).reshape(7,3)
    for fi, gi, ai in params:
        omega_i = 2*np.pi*fi*1e12
        gamma_i = gi*1e12
        ai_si = ai * conv_sq
        denom = (omega_i**2 - omega**2)**2 + 4*(gamma_i**2)*(omega**2)
        alpha_fit += K * (omega**2) * (2*gamma_i*ai_si) / denom
    return alpha_fit



def lorentz_dispersion(omega, params_n, K):
    """
    Calculate refractive index n(ω) and absorption α(ω) 
    from fitted Lorentz oscillator parameters.

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

    Returns
    -------
    n_fit : array
        Refractive index as function of ω.
    alpha_fit : array
        Absorption [1/mm] as function of ω.
    """
    n_inf = params_n[0]
    osc_params = np.array(params_n[1:]).reshape(7, 3)

    conv_sq = (2*np.pi*1e12)**2
    n_fit = np.full_like(omega, n_inf, dtype=float)
    alpha_fit = np.zeros_like(omega, dtype=float)

    for fi, gi, ai in osc_params:
        omega_i = 2*np.pi*fi*1e12   # resonance [rad/s]
        gamma_i = 2*np.pi*gi*1e12   # damping [rad/s]
        ai_si = ai * conv_sq        # convert THz^2 → (rad/s)^2

        denom = (omega_i**2 - omega**2)**2 + 4*(gamma_i**2)*(omega**2)

        # n contribution
        n_fit += ai_si * (omega_i**2 - omega**2) / denom

        # α contribution
        alpha_fit += K * (omega**2) * (2*gamma_i*ai_si) / denom

    return n_fit, alpha_fit



# ---------------------------------
# Constants & parameters
# ---------------------------------
c = 299792458.0                 # speed of light [m/s]
eps0 = 8.854187817e-12          # vacuum permittivity [F/m]
dz = 10e-6                       # step size in z [m]
z_max = 0.4e-3                  # crystal length [m]
z_steps = int(z_max / dz)

# ---------------------------------
# Optical pulse parameters
# ---------------------------------
lambda0 = 1468e-9  # [m]
omega0 = 2 * np.pi * c / lambda0 # [rad/s]

fwhm_int = 75e-15 # [s] inensity fwhm
tau_fwhm_e = fwhm_int * np.sqrt(2) # [s] electric field fwhm 
delta_nu_fwhm_e = 0.44 / tau_fwhm_e # [1/s]., [Hz] electiric field bandwidth 
delta_omega_fwhm_e = 2 * np.pi * delta_nu_fwhm_e # [rad/s] electric field bandwidth in rad/s
sigma_omega_e = delta_omega_fwhm_e / (2 * np.sqrt(2 * np.log(2))) # [rad/s]  electric field sigma 

opt_span = 40.0 * sigma_omega_e
n_opt = 2**10
omega_opt = np.linspace(omega0 - opt_span, omega0 + opt_span, n_opt)

n_thz = 2**9
omega_thz = np.linspace(2*np.pi*0.01e12, 2*np.pi*10.5e12, n_thz)

# ---------------------------------
# Optical refractive index (Sellmeier)
# ---------------------------------
q = 1.45
n0_sell = 2.026
wl0_sell = 532e-9  # [m]
wl_opt = c / (omega_opt/(2*np.pi))
n_opt_index = np.sqrt(n0_sell**2 + (q * wl0_sell**2) / (wl_opt**2 - wl0_sell**2))
k_opt = n_opt_index * omega_opt / c

#### Load, extract and calculate the Thz ref index and absoprtion ####
refractive_index_filename = r'/Users/praizanyanwu/Documents/inrs/nonlinear_optics/optical_rectification/source/DSTMS_THzindex.csv'
absorption_filename = r'/Users/praizanyanwu/Documents/inrs/nonlinear_optics/optical_rectification/source/DSTMS_THzabsorption_mm-1.csv'
df_n = pd.read_csv(refractive_index_filename, header=None, names=['freq_thz', 'n'])
df_alpha = pd.read_csv(absorption_filename, header=None, names=['freq_thz', 'alpha'])

freq_n_rad_s = df_n['freq_thz'].values * 1e12 * 2*np.pi
n_from_file = df_n['n'].values
freq_alpha_rad_s = df_alpha['freq_thz'].values * 1e12 * 2*np.pi
alpha_from_file = df_alpha['alpha'].values


n_data = np.interp(omega_thz, freq_n_rad_s, n_from_file)
alpha_data = np.interp(omega_thz, freq_alpha_rad_s, alpha_from_file)*1e3


# Initial guesses for fitting Thz ref index and absoption 

initial_guesses_n = np.array([ 
        2.17770253, 
        0.66616569,  0.39155946,  0.02588922, 
        1.05044626,  0.78700992,  0.144889  , 
        2.66681166,  4.47792644,  0.1850857 ,
        5.11804151,  3.60140801,  0.41155744,
        5.95152113,  1.68777467,  0.06539503, 
        6.75972297,  2.05411217,  0.28160896, 
        8.31759649, 3.07122443,  0.78472676,
        # 17.08481059,  50,  0.46564823
        ])

lower_bounds_n = [
    1.8,
    0.4,0.01,0.01,  
    0.7,0.01,0.01,  
    2.0,0.1,0.1,
    4.5,0.05,0.01,  
    5.0,0.05,0.01,  
    6.5,0.05,0.1,
    8.0,0.05,0.1,  
    # 12,18,0.1
]

upper_bounds_n = [   2.17770253, 
                    0.7,  0.4,  0.03, 
                    1.06,  0.8,  0.15  , 
                    2.7,  4.5,  0.2 ,
                    5.2,  3.7,  0.5,
                    6,  1.8,  0.07, 
                    6.8,  2.1,  0.3, 
                    8.4, 3.1,  0.8,
                    # np.inf,  100,  np.inf
                    ]

# Fit refractive index
popt_n, _ = curve_fit(
    lorentz_n_only,
    omega_thz, n_data,
    p0=initial_guesses_n,
    bounds=(lower_bounds_n, upper_bounds_n),
    maxfev=300000
)

print("Parameter list popt_n: ", popt_n)

# Fit absoprtion 
popt_alpha, _ = curve_fit(
    lambda omega, K: lorentz_alpha_only(omega, K, *popt_n[1:]),
    omega_thz, alpha_data,
    p0=[7e-12],
    bounds=([1e-14],[1e-8]),
    maxfev=200000
)

print("Parameter list popt_alpha: ", popt_alpha)

K_fit = popt_alpha[0]
# popt_n[-3:] = initial_guesses_n[-3:]
print("global THz absorption scaling factor is ", popt_alpha)
omega_thz = np.linspace(2*np.pi*1e-12, 2*np.pi*20e12, 2**9)  # rad/s
n_thz_index = lorentz_n_only(omega_thz, *popt_n) +0.09 
alpha_thz = lorentz_alpha_only(omega_thz, K_fit, *popt_n[1:]) +3000
k_thz = n_thz_index * omega_thz / c

### Load optical absorption #### 
opt_absorption_filename = r'/Users/praizanyanwu/Documents/inrs/nonlinear_optics/optical_rectification/data/DSTMS_Optabsorption_cm-1.csv'
df_opt_alpha = pd.read_csv(opt_absorption_filename, header=None, names=['wl', 'alpha'])
wl_m = np.sort(np.array(df_opt_alpha['wl'].values)) * 1e-9  # convert nm → m
freq_op_rad_s = 2*np.pi*c / wl_m
opt_alpha_from_file = df_opt_alpha['alpha'].values
# Sort the wavelengths such that the frequcies are decending 
sort_idx = np.argsort(freq_op_rad_s)
freq_op_rad_s_sorted = freq_op_rad_s[sort_idx]
opt_alpha_sorted = opt_alpha_from_file[sort_idx]
alpha0_opt = np.interp(omega_opt, freq_op_rad_s_sorted, opt_alpha_sorted)*1e2 # [1/m]


plt.figure(figsize=(12, 5))

# Plot refractive index
plt.subplot(1, 3, 1)
plt.plot(df_n['freq_thz'], n_from_file, 'o', label='Original Data from File')
plt.plot(omega_thz / (2 * np.pi * 1e12), n_thz_index, '-', label='Interpolated Data')
plt.title('Refractive Index')
plt.xlabel('Frequency (THz)')
plt.ylabel('Refractive Index (n)')
plt.legend()
plt.grid(True)
plt.show()

plt.subplot(1, 3, 1)
plt.plot(omega_opt / (2 * np.pi * 1e12), n_opt_index, '-', label='Interpolated Data')
plt.title('Refractive Index')
plt.xlabel('Frequency (THz)')
plt.ylabel('Refractive Index (n)')
plt.legend()
plt.grid(True)
plt.show()

# Plot absorption
plt.subplot(1, 3, 2)
plt.plot(df_alpha['freq_thz'], alpha_from_file*1e3, 'o', label='Original Data from File')
plt.plot(omega_thz / (2 * np.pi * 1e12), alpha_thz, '-', label='Interpolated Data')
plt.title('Absorption Coefficient')
plt.xlabel('Frequency (THz)')
plt.ylabel('Absorption (α)')
plt.legend()
plt.grid(True)
plt.show()

# Plot absorption
plt.subplot(1, 3, 3)
plt.plot(2*np.pi*c/freq_op_rad_s*1e9, opt_alpha_from_file*1e2, 'o', label='Original Data from File')
plt.plot(2*np.pi*c/omega_opt *1e9, alpha0_opt, '-', label='Interpolated Data')
plt.title('Absorption Coefficient')
plt.xlabel('Frequency (nm)')
plt.ylabel('Absorption (α)')
plt.legend()
plt.grid(True)
plt.show()

# #%%
# # ---------------------------------
# # Material parameters (optical)
# # ---------------------------------
# # alpha0_opt = 10
# gamma3PA = 6e-26           # [m^3/W^2] 3 photon absorption coeff. 
# chi2 = 428e-12             # [m/V] 
# beta1 = np.gradient(k_opt, omega_opt)
# beta2 = np.gradient(beta1, omega_opt) # 1177e-30
# i0 = np.argmin(np.abs(omega_opt - omega0))
# beta2_at_omega0 = beta2[i0]

# # ---------------------------------
# # Precompute frequency combos & phases
# # ---------------------------------
# sum_freqs = omega_opt[:, None] + omega_thz[None, :]
# diff_freqs = omega_opt[:, None] - omega_thz[None, :]
# #%%
# # n_opt_sum_int = np.interp(sum_freqs, omega_opt, n_opt_index)
# # n_opt_diff_int = np.interp(diff_freqs, omega_opt, n_opt_index)

# n_opt_sum = np.interp(sum_freqs.ravel(), omega_opt, n_opt_index).reshape(sum_freqs.shape)
# n_opt_diff = np.interp(diff_freqs.ravel(), omega_opt, n_opt_index).reshape(diff_freqs.shape)
# n_carr = np.interp(omega0, omega_opt, n_opt_index)
# #%%
# # Calculate k vectors at the new frequencies via interpolation
# # k_sum = np.interp(sum_freqs, omega_opt, k_opt)
# # k_diff= np.interp(diff_freqs, omega_opt, k_opt)
# # k_sum = n_opt_sum * sum_freqs / c
# # k_diff = n_opt_diff * diff_freqs / c
# #%%
# dk_domega = np.gradient(k_opt, omega_opt)

# # Then, calculate the group index at the carrier frequency
# i0 = np.argmin(np.abs(omega_opt - omega0))
# ng_at_omega0 = dk_domega[i0] * c

# beta1 = dk_domega
# beta2 = np.gradient(beta1, omega_opt)
# beta2_at_omega0 = beta2[i0]


# # ---------------------------------
# # Precompute frequency combos & phases
# # ---------------------------------
# sum_freqs = omega_opt[:, None] + omega_thz[None, :]
# diff_freqs = omega_opt[:, None] - omega_thz[None, :]

# # sum_freqs = np.array( [] ) 
# # diff_freqs = np.array([])
# # for i in range(n_opt):
# #     sum_freqs.append(omega_opt[i] + omega_thz)
# #     diff_freqs.append(omega_opt[i] - omega_thz)

# # --- Using the approximation to calculate Delta_k ---
# # The approximation k(w+Omega) - k(w) ~ Omega*ng/c
# # Delta_k_sum_approx  = ng_at_omega0 / c * omega_thz - k_thz

# # The approximation k(w-Omega) - k(w) ~ -Omega*ng/c
# # Delta_k_diff_approx = -ng_at_omega0 / c * omega_thz + k_thz

# # Assign the approximate values to the propagation terms
# # Delta_k_sum  = Delta_k_sum_approx
# # Delta_k_diff = Delta_k_diff_approx
# #%%
# k_sum = n_opt_index[:, None] * sum_freqs / c
# k_diff = n_opt_index[:, None] * diff_freqs / c

# Delta_k_sum  = k_sum  - k_opt[:, None] - k_thz[None, :]
# Delta_k_diff = k_diff - k_opt[:, None] + k_thz[None, :]

# domega_opt = (omega_opt[1] - omega_opt[0]).real
# dOmega_thz = (omega_thz[1] - omega_thz[0]).real

# # omega_opt = np.linspace(np.min(diff_freqs), np.max(sum_freqs), 2**9)
# # ---------------------------------
# # Fields
# # ---------------------------------
# Eop = np.zeros((z_steps, len(omega_opt)), dtype=np.complex128)
# Ethz = np.zeros((z_steps, len(omega_thz)), dtype=np.complex128)

# # --- initial (spectral) Gaussian ---
# tau = fwhm_int/ (np.sqrt(2* np.log(2))) 
# E0_tilde = np.exp(- tau**2 *(omega_opt - omega0)**2/4) #  defintion from https://dspace.mit.edu/bitstream/handle/1721.1/93837/900736523-MIT.pdf?sequence=2&isAllowed=y
# # E0_tilde = np.exp(-0.5 * ((omega_opt - omega0) / sigma_omega_e)**2)
# # E0_tilde = np.exp(- 2*np.log(2) * ((omega_opt - omega0) / delta_omega_fwhm)**2)

# # Physical peak power / intensity -> physical field amplitude
# fluence_opt = np.array([6.5]) *1e-3/1e-4#, 1, 2.2, 4.4])*1e-3/1e-4 # [J/m^2]
# fwhm_opt = 2*1e-3 # [m]
# # w0 = fwhm_opt/np.sqrt(2*np.log(2)) # beam waist [m]
# power_thz = np.zeros(len(fluence_opt))
# for j, flu in enumerate(fluence_opt):     
#     I_peak = 2*flu/fwhm_int * np.sqrt(np.log(2)/np.pi) # P_peak / (np.pi * w0**2)   # W/m^2 Caclulation of peak intensity from measured fluence 
#     E_peak_phys = np.sqrt(2.0 * I_peak / (c * eps0 * n_carr))   # V/m Calculate electric field from peak intenisty (time domain) 
#     # sigma_t = tau_fwhm_e / (2*np.sqrt(2 * np.log(2) )) # 
#     # spectral amplitude prefactor: 
#     # A_spec = E_peak_phys * sigma_t /np.sqrt(2*np.pi)   # [V·s/m]  
#     A_spec = E_peak_phys * tau /(2*np.sqrt(np.pi)) # E-field amplitude from time to freq domain  from https://dspace.mit.edu/bitstream/handle/1721.1/93837/900736523-MIT.pdf?sequence=2&isAllowed=y
#     # Build physical initial spectrum (V*s/m)
#     Eop[0] = E0_tilde * A_spec  # Initial field for simulation with calibrated amplitude from experimental data (freq domain) 
    

    
#     print("E_peak_phys =", E_peak_phys, "V/m")
#     print("A_spec =", A_spec * (2*np.sqrt(np.pi)), "V * s / m")
#     print("chi2_sim =", chi2)
#     print("gamma3PA_sim =", gamma3PA)
    

    
#     def interp_complex(x_new, x, y):
#         x_flat = x_new.ravel()
#         # Find points that are within the original x range
#         valid_indices = (x_flat >= np.min(x)) & (x_flat <= np.max(x))
        
#         # Initialize result array with zeros
#         res = np.zeros_like(x_flat, dtype=np.complex128)
        
#         # Perform interpolation only on valid points
#         res[valid_indices] = np.interp(x_flat[valid_indices], x, np.real(y)) + \
#                              1j * np.interp(x_flat[valid_indices], x, np.imag(y))
        
#         return res.reshape(x_new.shape)
    
#     # ---------------------------------
#     # Derivatives
#     # ---------------------------------
    
#     def dEthz_dz(Ethz_tilde, Eop_tilde, z):
#         Eop_plus = interp_complex(sum_freqs, omega_opt, Eop_tilde)
#         phase_sum = np.exp(-1j * Delta_k_sum *z)
#         integrand = Eop_plus * np.conj(Eop_tilde)[:, None] * phase_sum
#         mixing_tilde = np.trapz(integrand, x=omega_opt, axis=0) 
#         pref = -1j * (omega_thz**2) / (2 * c**2 * k_thz)
#         return -0.5 * alpha_thz * Ethz_tilde + pref * (chi2 * mixing_tilde)
    
#     def dEop_dz(Eop_tilde, Ethz_tilde, z):
#         I_tilde =  0.5*(c * eps0 * n_opt_index)*np.abs(Eop_tilde)**2
#         alpha_eff_tilde = alpha0_opt + gamma3PA * (I_tilde**2)
#         loss = -0.5 * alpha_eff_tilde * Eop_tilde
        
#         dispersion = 1j * beta2_at_omega0 / 2 * (omega_opt - omega0)**2 * Eop_tilde
                
#         Eop_plus = interp_complex(sum_freqs, omega_opt, Eop_tilde)       
#         phase_down = np.exp(-1j * Delta_k_sum * z)
#         integrand_plus = Eop_plus * np.conjugate(Ethz_tilde)[None, :] * phase_down
#         down_term = np.trapz(integrand_plus, x=omega_thz, axis=1) 
    
#         Eop_minus = interp_complex(diff_freqs, omega_opt, Eop_tilde)
#         phase_up = np.exp(-1j * Delta_k_diff * z )
#         integrand_minus = Eop_minus * Ethz_tilde[None, :] * phase_up
#         up_term   = np.trapz(integrand_minus, x=omega_thz, axis=1) 
#         pref = - 1j * (omega_opt**2) / (2 * c**2 * k_opt)
#         nonlinear = pref * chi2 * down_term + pref * chi2* up_term
    
#         return loss + dispersion + nonlinear 
    
#     # ---------------------------------
#     # RK4 propagation
#     # ---------------------------------
#     dTHz = np.zeros((z_steps, n_thz), dtype=np.complex128)
#     print("Starting propagation...")
#     for i in range(z_steps - 1):

#         z = i * dz

        
#         # Step 1
#         k1O = dEop_dz(Eop[i], Ethz[i], z)
#         k1T = dEthz_dz(Ethz[i], Eop[i], z)
    
#         # Step 2
#         Eop_temp2 = Eop[i] + 0.5 * dz * k1O
#         Ethz_temp2 = Ethz[i] + 0.5 * dz * k1T
#         k2O = dEop_dz(Eop_temp2, Ethz_temp2, z + 0.5 * dz)
#         k2T = dEthz_dz(Ethz_temp2, Eop_temp2, z + 0.5 * dz)
    
#         # Step 3
#         Eop_temp3 = Eop[i] + 0.5 * dz * k2O
#         Ethz_temp3 = Ethz[i] + 0.5 * dz * k2T
#         k3O = dEop_dz(Eop_temp3, Ethz_temp3, z + 0.5 * dz)
#         k3T = dEthz_dz(Ethz_temp3, Eop_temp3, z + 0.5 * dz)
    
#         # Step 4
#         Eop_temp4 = Eop[i] + dz * k3O
#         Ethz_temp4 = Ethz[i] + dz * k3T
#         k4O = dEop_dz(Eop_temp4, Ethz_temp4, z + dz)
#         k4T = dEthz_dz(Ethz_temp4, Eop_temp4, z + dz)
    
#         # Final update
#         Eop[i+1] = Eop[i] + (dz/6.0) * (k1O + 2*k2O + 2*k3O + k4O)
#         Ethz[i+1] = Ethz[i] + (dz/6.0) * (k1T + 2*k2T + 2*k3T + k4T)
        
#         if (i % 10 == 0):
#             print(f"Step {i}/{z_steps-1}, z = {z*1e3:.3f} mm")
            
#     power_thz[j-1] = np.sum(np.abs(Ethz[-1])**2)*dOmega_thz
#     print("Propagation finished.")
#     #%% ---------------------------------
#     # Plots
#     # ---------------------------------
#     plt.figure(figsize=(12,5))
    
#     # Optical spectrum
#     freq_opt = omega_opt/(2*np.pi)
#     wl_nm = c/freq_opt *1e9
#     plt.subplot(1, 2, 1)
#     plt.plot(wl_nm, np.abs(Eop[0])**2, label='Input')
#     plt.plot(wl_nm, 0.3*np.abs(Eop[0])**2 + np.abs(Eop[-10])**2, label='Output')
#     plt.xlabel('Wavelength (nm)')
#     plt.ylabel('Spectral Intensity (a.u.)')
#     plt.title(f'Optical Spectrum ({lambda0*1e9:.0f} nm, {fwhm_int*1e15:.0f} fs, {flu*1e3*1e-4} mJ/cm^2)')
#     plt.legend()
#     plt.grid(True)
    
#     # THz spectrum
#     freq_thz = omega_thz/(2*np.pi*1e12)
#     plt.subplot(1, 2, 2)
#     # Calculate spectral intensity for THz field
#     thz_intensity = np.abs(Ethz[-10])**2
#     plt.plot(freq_thz, thz_intensity, label='THz Output')
#     plt.xlabel('Frequency (THz)')
#     plt.ylabel('Spectral Intensity (a.u.)')
#     plt.title('THz Spectrum')
#     plt.legend()
#     plt.grid(True)
    
#     plt.tight_layout()
#     plt.show()