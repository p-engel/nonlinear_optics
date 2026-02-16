# measurement.py
from numpy import sum, abs
from .definitions import Gaussian
from .propagator import ORPropagator
from . import run

def optical_energy(Ew, dw): return sum( abs(Ew)**2 ) * dw

def conversion_efficiency(
    w0: float = 203,        # [THz]
    t_fwhm: float = 75e-3,  # [ps]
    E0: float = 5.4315e8,   # [V/m]
    cascade=True
):
    # --- initial pulse ---
    pulse = Gaussian(t_fwhm=t_fwhm, w0=w0, E0=E0)

    # --- OR propagator model ---
    model = ORPropagator(pulse, cascade=cascade)

    output = run.or_simulation(model)

    return (
        1 - (
            optical_energy(output["Ew"], model.dw) 
            / optical_energy(pulse.field_w(), model.dw)
        )
    )
