# strong coupling
import numpy as np
from typing import Union

# constants
c_thz = 299792458e-12  # [THz]

class FabryPerot():
    def __init__(
        self,
        reflectance: float = 0.0,
        freq: Union[float, np.ndarray] = 0.0,
        index: float = 0.0,
        length: Union[float, np.ndarray] = 0.0
    ):
        """
        Paramters
        ---------
        reflectance : [1]
        freq        : incident frequency [THz]
        index       : refractive index [1]
        length      : length of cavity  [m]
        """
        self.R = reflectance
        self.w = freq
        self.n = index
        self.L = length

    def fsr(self): return c_thz / (2 * self.n * self.L);  # FSR [THz]

    def fwhm(self): return 2 * (1 - self.R);

    def peak_bandwidth(self): return self.fwhm() / (2 * np.pi) * self.fsr();

    def T(self, phi=0):
        """transmittance"""
        delta = (4 * np.pi * self.n * self.L * self.w / c_thz) + (2 * phi)
 
        return (
            (1 - self.R)**2 /
            ( 1 + self.R**2 - 2*self.R*np.cos(delta) )
        )
