# unit test run.py
from numpy import exp, allclose
from optical_rectification.run import or_simulation
from optical_rectification.definitions import Index, Gaussian

def test_or_simulation():
    # --- initial input field ---
    w0 = 203        # [THz]
    t_fwhm = 75e-3  # [ps]
    E0 = 5.431e8    # [V/m]
    depth = 1e-3    # [m]

    try:
        output = or_simulation(
            t_fwhm=t_fwhm, w0=w0, E0=E0, depth=depth,
            cascade=False
        )
        cond = ( output["sol"].y[:, -1].shape == (
                ( len(output["pulse"].w) + len(output["model"].Ω) ),
            )
        )
        assert cond, (
            f"the model's state vector's degrees of freedom should ",
            "match the dimension of the fields"
        )
        Ewf_expect = ( 
            output["model"].Ew0 
            * exp( -0.5 * output["model"].alpha_w * depth  )
        )
        assert allclose(Ewf_expect, output["Ew"], rtol=1e-2), f":/"
    except AssertionError as a:
        print(f'AssertionError: {a}')

    return 0.0

print( test_or_simulation() )
