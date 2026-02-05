from scipy.integrate import solve_ivp
from .propagator import ORPropagator
from .definitions import Gaussian 

def or_simulation(
    w0: float = 203,        # [THz]
    t_fwhm: float = 75e-3,  # [ps]
    E0: float = 5.4315e8,   # [V/m]
    Ω_max: int = 10,        # [THz]
    depth: float = 1e-3,    # [m]
    Nz: int = 200,
    cascade=True
):
    # --- initial pulse ---
    pulse = Gaussian(t_fwhm=t_fwhm, w0=w0, E0=E0)

    model = ORPropagator(Ω_max, pulse, cascade=cascade)

    # --- propagation ---
    y0 = model.pack(model.Ew0, model.EΩ0)

    sol = solve_ivp(
        model.rhs,
        (0, depth),
        y0,
        method="DOP853",
        rtol=1e-5, atol=1e-8,
        max_step=depth/Nz
    )

    Ewf, EΩf = model.unpack(sol.y[:, -1])

    return {
        "z": sol.t,
        "Ew": Ewf,
        "EΩ": EΩf,
        "sol": sol,
        "model": model,
        "pulse": pulse
    }
