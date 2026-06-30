# measurement.py
from __future__ import annotations
from typing import TYPE_CHECKING
from numpy import sum, abs, array
from dataclasses import dataclass

from .definitions import Gaussian
if TYPE_CHECKING:
    from .run import ORSimResult

@dataclass
class Observable:
    result: ORSimResult

    def __post_init__(self):
        from .run import ORSimResult
        if not isinstance(self.result, ORSimResult):
            raise TypeError(f"initiate Observable with type ORSimResult")

    def power_spectrum(self, field_w, n=1):
        """n -- refractive index"""
        return Gaussian().power_spectrum(field_w, n=n)  # [W * ps^2]

    def energy(self, power_spectrum):
        return self.result.model.dw * sum(power_spectrum)

    def conversion_efficiency(self):
        ps_thz = self.power_spectrum(self.result.EΩ)
        ps_opt = self.power_spectrum(self.result.model.Ew0)
        return self.energy(ps_thz) / self.energy(ps_opt)