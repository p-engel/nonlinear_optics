# parameter file
import os
from numpy import pi, array, concatenate

## parameters for eight Lorentz oscillators modeling refractive index in
## Terahertz region
## [n_inf, [w1, gam1, a1], [w2, gam2, a2],...[w8, gam8, a8]]
## from data in ref:
prefac = (6.28318531e12)*(5.78013078e-09)  ## global absorption scaling factor
param_thz = [
    2.12358423e+00, 
    [6.64150004e-01, 0.3*3.87674924e-01, 2.56647179e-02],
    [1.05051147e+00, 0.3*7.91317790e-01, 1.45358886e-01], 
    [2.67450746e+00, 0.3*4.50000000e+00, 1.92339389e-01], 
    [5.11752954e+00, 0.3*3.70000000e+00, 4.35075208e-01], 
    [5.95339784e+00, 0.3*1.70403682e+00, 6.75687665e-02],
    [6.76106905e+00, 0.3*2.10000000e+00, 2.94151150e-01], 
    [8.31525915e+00, 0.3*3.09324550e+00, 7.98205276e-01], 
    [1.75300342e+01, 0.3*1.80000000e+01, 2.57575216e+01]
]
param_radps = [param_thz[0]] + [
    [ 
        2*pi * oscillator[0], 
        2*pi * oscillator[1], 
        (2*pi)**2 * oscillator[2]
    ] 
    for oscillator in param_thz[1:]
]
ninf = param_radps[0]
oscillators = array(param_radps[1:]).flatten()                  # [rad / ps]
p = concatenate(([ninf], oscillators))
s = prefac / (2*pi)                                             # [1 / (m * rad/ps)]

## IR region; from data in ref:
param2_thz = [
    2.026,
    [1.51172213e+02, 2*1.00000000e+01, 5.26207870e+03],
    [1.65349167e+02, 2*4.81439707e+00, 1.81928019e+03],
    [1.71710164e+02, 2*2.00000000e+00, 9.97494678e+02],
    [1.76000000e+02, 2*3.43231330e+00, 6.54147241e+03],
    [2.14169102e+02, 2*7.00000000e+00, 1.99168749e+03],
    [2.49791614e+02, 2*9.06570323e+00, 2.19861101e+03],
    [2.94059844e+02, 2*2.60000000e+01, 3.35931650e+03]
]
param2_radps = [param2_thz[0]] + [
    [ 
        2*pi * osc[0], 
        2*pi * osc[1], 
        (2*pi)**2 * osc[2]
    ]
    for osc in param2_thz[1:]
]
ninf2 = param2_radps[0]
oscillators2 = array(param2_radps[1:]).flatten()
p2 = concatenate(([ninf2], oscillators2))
s2 = 9.34145391e-01 / (2*pi)

# Optical absorption spectrum from data in ref
# filepath
base_dir = os.path.dirname(__file__)  # directory of this Python file
fname = os.path.join(base_dir, '..', 'data', 'DSTMS_optical_absorption_nm-cm-1.csv')
fpath_n_opt = os.path.join(base_dir, '..', 'data', 'DSTMS_optical_refractve_index_nm.csv')
fpath_ng = os.path.join(base_dir, '..', 'data', 'DSTMS_group-index_nm.csv')
fname_alpha_opt = os.path.abspath(fname)
fname_n_opt = os.path.abspath(fpath_n_opt)
fname_ng = os.path.abspath(fpath_ng)