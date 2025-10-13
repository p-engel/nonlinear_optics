# test lorentz distribution function (pdf)
import numpy as np
from matplotlib import pyplot as plt

from optical_rectification import definitions, par

#Initialize index variables
w = np.linspace(2*np.pi*0.01, 2*np.pi*3, 2**9)  # [rad/s]
Index = definitions.Index(w,
            par.param[1:][0], 2*par.param[1:][1], par.param[1:][0],
            par.param[0], par.k)

def test_lorentzpdf():
    # parameters
    p = par.param[1]; w0 = p[0]; gam0 = 2*p[1];
    pdf = Index.lorentz(w0, gam0);

    try:
        condition = abs( max(pdf) - 1/gam0 ) <= 2e-3;
        assert condition, "the maximum of Lorentz PDF should be equal to \
            the inverse of the line-width"
    except AssertionError as a:
        print(f"Assertion error: {a}");
    except Exception as e:
        print(f"Unexpected error: {e}");
        
        return None

    return pdf

# plot lorentz pdf
plt.figure(figsize=(12,5))
pdf = test_lorentzpdf()
plt.plot(w/(2*np.pi), pdf)
plt.title('Lorentz PDF')
plt.xlabel(r'$\omega$ (THz)')
plt.ylabel(r'$P(\omega)$')
plt.grid(True)
plt.show()


