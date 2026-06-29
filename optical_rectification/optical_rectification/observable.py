# measurement.py
from numpy import sum, abs, array
from dataclasses import dataclass

from .definitions import Gaussian
from .run import ORSimResult

@dataclass
class Observable:
    result: ORSimResult

    def power_spectrum(self, field_w, n=1):
        """n -- refractive index"""
        return Gaussian().power_spectrum(field_w, n=n)  # [W * ps^2]

    def energy(self, power_spectrum):
        return self.result.model.dw * sum(power_spectrum)  # [J]

    def conversion_efficiency(self):
        ps_thz = self.power_spectrum(self.result.EΩ)
        ps_opt = self.power_spectrum(self.result.model.Ew0)
        return self.energy(ps_thz) / self.energy(ps_opt)