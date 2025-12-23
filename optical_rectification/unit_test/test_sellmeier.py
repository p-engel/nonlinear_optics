# test Sellmeier equation
import numpy as np
from matplotlib import pyplot as plt

from optical_rectification import definitions, par

# measured data
fname = par.fname_n_opt  # refractive index
s = definitions.Spectrum(fname)
spec = s.read_spec();
# parameter
lam_min = 700; lam_max = 2000;  # [nm]
ind_min = s.find_ind(spec[:,0], lam_min)
ind_max = s.find_ind(spec[:,0], lam_max)
xmin = spec[ind_min, 0]; xmax = spec[ind_max, 0];
x = np.linspace(xmin, xmax, 2**9)
w = definitions.c_thz / (x * 1e-9)                      # [THz]
index = definitions.Index(x)                            # Index class

def test_sellmeier():
    try:
        n = index.sellmeier(n_inf=2.026, lam0=455, q=0.17)
        cond = abs(n[0] - spec[ind_min, 1]) < 1e-1;
        assert cond, "The Sellmeier equation doesn't fit the data."
    except AssertionError as a:
        print(f"Assertion error: {a}");
    return n

# plot refractive index spectrum of DSTMS
plt.figure(figsize=(6.4,4.8))
n = test_sellmeier()
plt.plot(spec[:,0], spec[:,1], '-o')
plt.plot(x, n, '-')
plt.title('Refractive Index of DSTMS')
plt.xlabel(r'$\lambda$ (nm)')
plt.ylabel(r'$n(\lambda)$')
plt.grid(True)
plt.show()


