from numpy import ndarray, array, zeros
from scipy.integrate import solve_ivp
from .propagator import ORPropagator
from .observable import Observable
from typing import Any
from dataclasses import dataclass

@dataclass
class ORSimResult:
    z: ndarray
    Ew: ndarray
    EΩ: ndarray
    sol: Any
    model: ORPropagator


def or_simulation(model: ORPropagator) -> ORSimResult:
    # --- propagation ---
    y0 = model.pack(model.Ew0, model.EΩ0)

    sol = solve_ivp(
        model.rhs,
        (0, model.DEPTH),
        y0,
        method="DOP853",
        rtol=1e-5, atol=1e-8,
        max_step=200
    )

    Ewf, EΩf = model.unpack(sol.y[:, -1])

    return ORSimResult(
        z=sol.t,
        Ew=Ewf,
        EΩ=EΩf,
        sol=sol,
        model=model
    )


def sweep(*,
    t_fwhm: float = 35e-3, 
    b0: float = 1.699e-3, 
    pulse_freq0: ndarray = array([291, 235, 221, 207]),
    pulse_energy: float = 181e-6,
    mode: bool = False
):
    """
    sweep through pump pulse carrier frequency and pulse energy
    """
    conv_efficiency = zeros(pulse_freq0.shape, dtype=float)
    for i, f0 in enumerate( pulse_freq0 ):
        model = ORPropagator(
            t_fwhm=t_fwhm, b0=b0, f0=f0, U=pulse_energy, cascade=mode
        )
        result = or_simulation(model)
        conv_efficiency[i] = Observable(result).conversion_efficiency()

    return conv_efficiency
