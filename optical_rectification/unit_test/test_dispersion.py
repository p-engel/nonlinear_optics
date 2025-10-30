# test disperison relation
import numpy as np
from matplotlib import pyplot as plt

from optical_rectification import definitions, par

# constants
c = definitions.c
w = np.linspace(0.01, 2, 2**9)  # [THz]
index = definitions.Index(w);

def test_dispersion():
    dispersion = definitions.Dispersion(w,index.n())
    try:
        v = dispersion.phase_velocity()
        k = dispersion.k()
        f = v*k
        cond = np.mean(np.abs(f - w)) <= 1e-2;
        assert cond, "the dispersion relation should satisfy \
                w = v*k"
    except AssertionError as a:
        print(f"Assertion error: {a}");

    return v, k

# plot dispersion relation
fig, ax1 = plt.subplots(figsize=(6.4,4.8))
v, k = test_dispersion()
#ax1.plot(w, k*1e9, label="DSTMS crystal")
#ax1.plot(v*k, k*1e9, label="DSTMS crystal")
ax1.plot(k*1e9, v*k, label="DSTMS crystal")
ax1.plot(k*1e9, c*k, label="free space")
ax1.set_title('Dispersion')
ax1.set_xlabel(r'$k$ ($\rm{nm}^{-1}$)')
ax1.set_ylabel(r'$\omega$ (THz)')
ax1.legend()
plt.grid(True)
plt.show()
