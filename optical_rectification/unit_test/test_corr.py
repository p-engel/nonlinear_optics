# unit test of correlation function with Reimann sum
import matplotlib.pyplot as plt
import numpy as np
from optical_rectification.definitions import Gaussian, corr

def test_corr_gaussian_zero_shift():
    w0 = 204.22  # [THz]
    tau = np.sqrt(2) * 75e-15 * 1e12  # [ps]
    w = np.linspace(150, 250, 2**10)
    dw = w[1] - w[0]

    g = Gaussian(w0=w0, tau=tau, E0=1)
    Ew = g.field_w(w)

    I0 = corr(Ew, dw, 0)
    expected = abs(g.E0_w)**2 * g.sigma_w * np.sqrt(np.pi)

    assert np.isclose(I0, expected, rtol=1e-2) \
        f "The autocorrelation (at 𝜏 = 0) for a noormalised gaussian is 1"

# constants
w0 = 204.22  # [THz]
tau = np.sqrt(2) * 75e-15 * 1e12  # [ps]
w = np.linspace(150, 250, 2**10)  # [THz] domain
dw = w[1] - w[0]
gaussian = definitions.Gaussian(w0=w0, tau=tau, E0=1)
k_max = int(100 / dw)
kvals = np.arange(0, k_max + 1)
Ω = kvals * dw

def test_corr():
    try:
        I_k = [definitions.corr(gaussian.field_w(w), dw, k) for k in kvals]
        I_norm = (I_k / 
                    (abs(gaussian.E0_w)**2 * gaussian.sigma_w * np.sqrt(np.pi))
        )
        cond = abs(I_norm[0] - 1) < 1e-2
        assert cond, \
            "The autocorrelation (at 𝜏 = 0) for a noormalised gaussian is 1"
    except AssertionError as a:
        print(f"AssertionError : {a}")
    return I_norm

fig, ax = plt.subplots(figsize=(6.4, 4.8))

ax.plot(Ω, test_corr())
ax.set_ylabel(r"$\frac{1}{\vert E_0 \vert^2\sigma\sqrt{\pi}}$"
                + "$\int d\omega E(\omega + \Omega)E(\omega)$"
)
ax.set_xlabel(r"$\Omega \quad [THz]$")
ax.set_title("Correlation of a Gaussian field")

plt.grid(True)
plt.show()
