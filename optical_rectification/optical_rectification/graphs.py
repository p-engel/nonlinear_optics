# graphs.py
import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass, field

from .run import ORSimResult
from .observable import Observable
from .definitions import Z0, c_thz


def graph_efficiency(
    pulse_energy: np.ndarray, pulse_freq: np.ndarray, eta: np.ndarray
):
    """
    pulse_energy - (n,) [J]
    pulse_freq - (m,) [nm]
    eta - (n, m) conversion efficiency
    """
    if (
        not isinstance(pulse_energy, np.ndarray) 
        or not isinstance(eta, np.ndarray)
    ):
        raise TypeError(f'expected argument type as ndarray')
    if eta.ndim != 2:
        raise ValueError(f'eta must be 2D, got {eta.ndim}D') 
    if eta.shape != (pulse_energy.shape[0], pulse_freq.shape[0]):
        raise ValueError(
            f"eta has shape {eta.shape}, expected "
            f"({pulse_energy.shape[0]}, {pulse_freq.shape[0]}) to match "
            f"pulse_energy and pulse_freq"
        )

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    for i, lam0 in enumerate(pulse_freq):
        ax.plot(pulse_energy * 1e6, eta[:, i]*100, '*', 
                  label=rf'$\lambda = {lam0:.0f}$ nm'
        )

    ax.set_title(r"Terahertz conversion efficiency")
    ax.set_xlabel(r"pulse energy $[\mu\,J]$")
    ax.set_ylabel(r"$\eta$ [%]")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    
    return fig


@dataclass
class ORgraphs:
    result: ORSimResult
    measure: Observable = field(init=False)

    def __post_init__(self):
        self.measure = Observable(self.result)
        self.lam = c_thz * 2*np.pi / self.result.model.pulse.w * 1e9

    def _intensity(self, field):
        return np.abs(field)**2 / (2*Z0) * 1e-6 * 1e-4

    def spectral_intensity(self):
        fig, ((ax, ax1), (ax2, ax3)) = plt.subplots(
        	figsize=(15, 10), nrows=2, ncols=2
        )
        self._thz_2d(ax)
        self._thz_1d(ax1)
        self._optical_2d(ax2)
        self._optical_1d(ax3)
        plt.tight_layout()
        return fig

    def _thz_2d(self, ax):
        mesh = ax.pcolormesh(
            self.result.model.Ω/(2*np.pi),
            self.result.model.pulse.r*1e3,
            self._intensity(self.result.EΩ),
            cmap="viridis", shading="auto"
        )
        ax.figure.colorbar(mesh, ax=ax, label=r"$I \quad [{\rm MW / cm^2 ps^2}]$")
        ax.set_xlabel(r"$\Omega \: / \: 2\pi \quad [THz]$")
        ax.set_ylabel('r [mm]')
        ax.set_title("Terahertz Spectra")

    def _thz_1d(self, ax):
        ax.plot(
            self.result.model.Ω/(2*np.pi),
            self._intensity(self.result.EΩ[0, :]),
            'r'
        )
        ax.set_title("Terahertz Spectrum at Peak Fluence r=0")
        ax.set_xlabel(r"$\Omega \: / \: 2\pi \quad [THz]$")
        ax.set_ylabel(r"$I \quad [{\rm MW / cm^2 ps^2}]$")
        ax.tick_params(axis='y', labelcolor='k')
        ax.grid(True)

    def _optical_2d(self, ax):
        lam = c_thz * 2*np.pi / self.result.model.pulse.w * 1e9
        mesh = ax.pcolormesh(
            lam,
            self.result.model.pulse.r*1e3,
            self._intensity(self.result.Ew),
            cmap="viridis", shading="auto"
        )
        ax.figure.colorbar(mesh, ax=ax, label=r"$I \quad [{\rm MW / cm^2 ps^2}]$")
        ax.set_xlabel(r"$\lambda \quad [nm]$")
        ax.set_ylabel('r [mm]')
        ax.set_title("Optical Input Spectra")

    def _optical_1d(self, ax):
        lam = c_thz * 2*np.pi / self.result.model.pulse.w * 1e9
        ax.plot(
            lam,
            np.abs(self.result.model.pulse.field_w()[0, :])**2/(2*Z0) * 1e-6*1e-4,
            '--', label=r'input pulse z=0'
        )
        ax.plot(
            lam,
            self._intensity(self.result.Ew[0, :]),
            '.r', alpha=0.4,
            label=r'input pulse z=$370 \: {\rm {\mu}m}$, OR_simulation'
        )
        ax.legend()
        ax.set_title("Optical Input Spectrum at Peak Fluence r=0")
        ax.set_xlabel(r"$\lambda \quad [nm]$")
        ax.set_ylabel(r"$I \quad [{\rm MW / cm^2 ps^2}]$")
        ax.tick_params(axis='y', labelcolor='k')
        ax.grid(True)