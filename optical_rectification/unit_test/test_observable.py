from numpy import sqrt, pi, abs
from optical_rectification import observable, run, propagator

model = propagator.ORPropagator( 
    t_fwhm=35e-3, f0=210, U=181e-6, b0=1.699e-3, cascade=False 
)
result = run.or_simulation( model )

def test_observable():
    try:
        analyse = observable.Observable( result )
        ps = analyse.power_spectrum( result.model.Ew0 )
        cond = abs( analyse.energy(ps) - model.pulse.energy ) <= 1e-2
        val = analyse.conversion_efficiency()
        cond1 = ( val < 1 )
        assert cond, "integral of input power_spectrum should equal input pulse energy"
        assert cond1, "conversion efficiency is greater than unity"
    except AssertionError as a:
        print(f"AssertionError: ", a)
    return val

print(test_observable())
