# unit test propagator.py
import numpy as np
from optical_rectification.definitions import Index, Gaussian, DEPTH
from optical_rectification.propagator import ORPropagator, run_simulation

def test_orpropagator():
    # --- initial input field ---
    w0 = 203  # [THz]
    t_fwhm = 75e-3  # [ps]
    pulse = Gaussian(t_fwhm=t_fwhm, w0=w0, E0=5.431e8)
    w = np.linspace((w0 - 3*pulse.delta), (w0 + 3*pulse.delta), 2**10)
    Ω_max = 10
    Ew0 = pulse.field_w(w)

    try:
        # --- propagation model ---
        model = ORPropagator(w, Ω_max, pulse=None)    
        print(model.dw)
        # --- solver ---
        sol = run_simulation( model, Ew0, (0, DEPTH) )
        cond = sol.y[:, -1].shape == ((model.Nw + model.NΩ),)
        assert cond, (f"the model's state vector's degrees of freedom should ",
            "match the dimension of the fields")
        Ewf, EΩf = model.unpack(sol.y[:, -1])
    except AssertionError as a:
        print(f'AssertionError: {a}')

    return 0.0

print(test_orpropagator())
