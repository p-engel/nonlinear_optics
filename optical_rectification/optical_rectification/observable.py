# measurement.py
from numpy import sum, abs
from dataclasses import dataclass

from .definitions import Gaussian, EPS0, c_thz
from .run import ORSimResult

@dataclass
class Observable:
    result: ORSimResult

    def power_spectrum(self, n=1):
        """n -- refractive index"""
        return [
            Gaussian().power_spectrum(self.result.EΩ, n=n),
            Gaussian().power_spectrum(self.result.Ew, n=n)
        ]

    # def energy_density(Ew, dw): 
    #     "spectral energy density [ J/m^3 * (rad/ps)^{-1} ]"
    #     return EPS0 * sum( abs(Ew)**2 ) * dw

    # def conversion_efficiency(
    #     f0: float = 203,         # [THz]
    #     t_fwhm: float = 75e-3,   # [ps]
    #     A: float = 5.4315e8,     # [V/m]
    #     gam3PA: float = 6e-26,   # [m^3/W^2]
    #     fluence: float = None,   # [J/m]
    #     cascade=True
    # ):  
    #     return