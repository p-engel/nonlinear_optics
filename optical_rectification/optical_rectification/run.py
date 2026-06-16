from numpy import ndarray
from scipy.integrate import solve_ivp
from .propagator import ORPropagator
from typing import TypedDict, Any
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