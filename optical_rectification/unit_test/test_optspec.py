import numpy as np
from matplotlib import pyplot as plt

from optical_rectification import definitions, par

def test_optspec():
    """
    check measurred absorption spectrum in optical regime
    """
    spec = definitions.opt_spec();
    w = spec[:,0]
    try:
        cond1 = spec.shape == (len(w), 2);
        cond2 = w[5] <= w[6];
        assert cond1, "the spectrum should have to columns for \
                frequency and intensity."
        assert cond2, "the spectral frequency axis should be an increasing \
                sequence"
    except AssertionError as a:
        print(f"Assertion error: {a}");
    except Exception as e:
        print(f"Unexpected error: {e}");

        return None

    return spec

# plot optical absorption spectrum of DSTMS
plt.figure(figsize=(12,5))
spec = test_optspec()
plt.plot(spec[:,0]*1e-12, spec[:,1])
plt.title('Optical Absorption of DSTMS')
plt.xlabel(r'$\omega$ (THz)')
plt.ylabel(r'$\alpha(\omega)$')
plt.grid(True)
plt.show()
