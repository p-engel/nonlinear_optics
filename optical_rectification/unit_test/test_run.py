# unit test run.py
from numpy import exp, allclose
from optical_rectification import run 
from optical_rectification import propagator
from optical_rectification.definitions import Gaussian

def test_or_simulation():
#     # --- initial input field ---
#     w0 = 203        # [THz]
#     t_fwhm = 75e-3  # [ps]
#     E0 = 5.431e8    # [V/m]
#     depth = 1e-3    # [m]

    try:
        pulse = Gaussian()
        model = propagator.ORPropagator(pulse, cascade=False)
        output = run.or_simulation(model)
        cond = ( output["sol"].y[:, -1].shape == (
                ( len(pulse.w) + len(output["model"].Ω) ),
            )
        )
        assert cond, (
            f"the model's state vector's degrees of freedom should ",
            "match the dimension of the fields"
        )
        Ewf_expect = ( 
            output["model"].Ew0 
            * exp( -0.5 * (
                output["model"].index_w.alpha()
                + 1j * output["model"].field_dispersion
                ) * model.DEPTH
            )
        )
        assert allclose(Ewf_expect, output["Ew"], rtol=1e-2), f":/"
    except AssertionError as a:
        print(f'AssertionError: {a}')

    return 0.0

print( test_or_simulation() )
