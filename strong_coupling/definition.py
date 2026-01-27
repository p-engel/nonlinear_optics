# strong coupling
import numpy as np
from typing import Union

# constants
c_thz = 299792458e-12  # [THz]

class FabryPerot():
    def __init__(
        self,
        reflectance: Union[float, np.ndarray] = 0.0,
        wavelen: Union[float, np.ndarray] = 0.0,
        index: Union[float, np.ndarray] = 0.0,
        length: Union[float, np.ndarray] = 0.0
    ):
        """
        Paramters
        ---------
        reflectance : [1]
        wavelen     : wavelength [m]
        index       : refractive index [1]
        length      : length of cavity  [m]
        """
        self.R = reflectance
        self.lam = wavelen
        self.n = index
        self.L = length
        # self.freq = 0 if wavelen is 0 else c_thz / wavelen;

    def fsr(self): return c_thz / (2 * self.n * self.L);  # FSR [THz]

    def fwhm(self): return 2 * (1 - self.R);

    def peak_bandwidth(self): return self.fwhm() / (2 * np.pi) * self.fsr();

