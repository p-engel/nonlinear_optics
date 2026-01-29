# test class FabryPerot
import numpy as np
from definition import FabryPerot

def test_fabryperot():
    L = np.linspace(1, 2000, 2**10) * 1e-6  # [m]
    cavity = FabryPerot(reflectance=0.5, index=3.42, length=L)
    try:
        delta_nu = cavity.peak_bandwidth()
        transmittance = cavity.T()

        assert delta_nu.shape==L.shape, f"input shape should match output"
        assert np.isclose(transmittance, 1).all(), (
            f"by definition transmittance is 1 for no incident frequency"
        )

    except AssertionError as a: print(f"AssertionError: {a}")

    return None

print( test_fabryperot() )
