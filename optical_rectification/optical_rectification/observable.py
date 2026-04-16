# measurement.py
from numpy import sum, abs
from .definitions import Gaussian
from .propagator import ORPropagator
from . import run

def optical_energy(Ew, dw): return sum( abs(Ew)**2 ) * dw

def conversion_efficiency(
    f0: float = 203,        # [THz]
    t_fwhm: float = 75e-3,  # [ps]
    A: float = 5.4315e8,   # [V/m]
    cascade=True
):
    # --- initial pulse ---
    pulse = Gaussian(t_fwhm=t_fwhm, f0=f0, A=A)

    # --- OR propagator model ---
    model = ORPropagator(pulse, cascade=cascade)

    output = run.or_simulation(model)

    return (
        optical_energy(output["EΩ"], model.dw) 
        / optical_energy(pulse.field_w(), model.dw)
    )
