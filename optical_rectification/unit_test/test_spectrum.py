import numpy as np
from matplotlib import pyplot as plt

from optical_rectification import definitions, par

def test_spectrum():
    """
    check measurred absorption spectrum in optical regime
    """
    # filename = "../../../data/DSTMS_Optabsorption_cm-1.csv"
    filename = par.fname_aplha_opt
    s = definitions.Spectrum(filename);
    spec = s.read_spec_file(); w = spec[:,0]
    optspec = s.opt_spec();
    try:
        cond1 = spec.shape == (len(w), 2);
        cond2 = w[5] <= w[6];
        cond3 = optspec is not None or optspec.shape == (len(w), 2);
        assert cond1, "the spectrum should have two columns for \
                frequency and intensity."
        assert cond2, "the spectral frequency axis should be an increasing \
                sequence"
        assert cond3, "the optical spetral output is an array type \
                with shape (N,2)"
    except AssertionError as a:
        print(f"Assertion error: {a}");
    except Exception as e:
        print(f"Unexpected error: {e}");

        return None

    return spec, optspec

# plot optical absorption spectrum of DSTMS
plt.figure(figsize=(12,5))
spec, optspec = test_spectrum()
plt.plot(spec[:,0]*1e-12, spec[:,1])
plt.plot(optspec[:,0]*1e-12, optspec[:,1])
plt.title('Optical Absorption of DSTMS')
plt.xlabel(r'$\omega$ (THz)')
plt.ylabel(r'$\alpha(\omega)$')
plt.grid(True)
plt.show()
