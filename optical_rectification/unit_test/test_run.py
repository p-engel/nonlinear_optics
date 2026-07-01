# unit test run.py
from numpy import exp, allclose, ndarray
from optical_rectification import run 
from optical_rectification import propagator
from optical_rectification.definitions import Gaussian

def test_sweep():
    try:
        conv_efficiency = run.sweep(mode=True)
        assert isinstance(conv_efficiency, ndarray), (
            f"expected sweep output as array of conversion efficiencies"
        )

    except AssertionError as a:
        print(f'AssertionError: a')

    return conv_efficiency

print( test_sweep() )

# def test_or_simulation():
#     try:
#         pulse = Gaussian()
#         model = propagator.ORPropagator(pulse, cascade=False)
#         output = run.or_simulation(model)
#         cond = ( output["sol"].y[:, -1].shape == (
#                 ( len(pulse.w) + len(output["model"].Ω) ),
#             )
#         )
#         assert cond, (
#             f"the model's state vector's degrees of freedom should ",
#             "match the dimension of the fields"
#         )
#         Ewf_expect = ( 
#             output["model"].Ew0 
#             * exp( -0.5 * (
#                 output["model"].index_w.alpha()
#                 + 1j * output["model"].field_dispersion
#                 ) * model.DEPTH
#             )
#         )
#         assert allclose(Ewf_expect, output["Ew"], rtol=1e-2), f":/"
#     except AssertionError as a:
#         print(f'AssertionError: {a}')
# 
#     return 0.0
# 
# print( test_or_simulation() )
