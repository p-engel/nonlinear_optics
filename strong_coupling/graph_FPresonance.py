# graph Fabry Perot resonances
import numpy as np
from matplotlib import pyplot as plt
from definition import FabryPerot

L = np.linspace(1750, 2000, 2**10) * 1e-6  # [m]
cavity = FabryPerot(reflectance=0.5, index=3.42, length=L)
delta_nu = cavity.peak_bandwidth()
print(delta_nu[-1]*1e3)
print(cavity.fsr()[-1]*1e3)

fig, ax = plt.subplots()
ax.plot(L*1e6, delta_nu*1e3, "-k", label="bandwidth")
ax.plot(L*1e6, cavity.fsr()*1e3, "--k", label="free spectral range")
ax.set_title(f"Fabry-Perot resonance in slab n=3.42, R=0.5")
ax.set_xlabel(r"slab thickness [${\rm {\mu}m}$]")
ax.set_ylabel(r"$[{\rm GHz}]$")
plt.legend()
plt.grid(True)
plt.show()
