# measurement.py
from numpy import sum, abs
from .definitions import Gaussian, EPS0, c_thz
from .propagator import ORPropagator
from . import run

def energy_density(Ew, dw): 
    "spectral energy density [ J/m^3 * (rad/ps)^{-1} ]"
    return EPS0 * sum( abs(Ew)**2 ) * dw

def conversion_efficiency(
    f0: float = 203,         # [THz]
    t_fwhm: float = 75e-3,   # [ps]
    A: float = 5.4315e8,     # [V/m]
    gam3PA: float = 6e-26,   # [m^3/W^2]
    fluence: float = None,   # [J/m]
    cascade=True
):
        
    # --- initial pulse ---
    pulse = Gaussian(t_fwhm=t_fwhm, f0=f0, A=A)

    # --- OR propagator model ---
    model = ORPropagator(pulse, gam3PA, cascade=cascade)

    output = run.or_simulation(model)
    
    if fluence is not None:
        return (
            c_thz*energy_density(output["EΩ"], model.dw)
            / fluence
        )
    else:
        return (
        energy_density(output["EΩ"], model.dw) 
        / ( 2 * energy_density(pulse.field_w(), model.dw) )
        )
