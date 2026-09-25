"""YAG-specific data with shared two-manifold Yb rate equations.

Runtime data were imported from origin/main (7aa9904), Yb-YAG/. No LuAG
cross sections, site density, lifetime or refractive indices are used here.
The source data and their licensing/provenance remain in data/provenance/.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np

from ybluag.model import YbLuAGMaterial, _positive
from . import material_data as data


@lru_cache(maxsize=1)
def _rt_spectra():
    rows = data._csv('spectra/yb_yag_293K_legacy.csv')
    return tuple(np.array([float(row[key]) for row in rows]) for key in
                 ('wavelength_nm', 'sigma_abs_cm2', 'sigma_em_cm2'))


@dataclass(frozen=True)
class YbYAGMaterial(YbLuAGMaterial):
    """RT Yb:YAG; inheritance reuses only generic two-manifold kinetics.

    The 0.95 ms lifetime is a literature model parameter, not tau(T,doping).
    Concentration scales ion density, not the per-ion spectral arrays.
    """
    yb_at_percent: float = 20.0
    pump_wavelength_nm: float = 969.0
    lifetime_s: float | None = .00095

    name = "Yb:YAG"
    supports_coupled_temperature = False
    spectral_range_nm = (905.0, 1095.0)

    def __post_init__(self):
        for key in ('yb_at_percent', 'temperature_K', 'pump_wavelength_nm',
                    'signal_wavelength_nm', 'lifetime_s'):
            if getattr(self, key) is None:
                raise ValueError('Yb:YAG requires an explicit positive lifetime')
            _positive(key, getattr(self, key))
        if self.yb_at_percent > 100:
            raise ValueError('Yb concentration exceeds Y-site occupancy')
        self.cross_sections_m2(self.pump_wavelength_nm)
        self.cross_sections_m2(self.signal_wavelength_nm)

    @property
    def number_density_m3(self):
        return float(data.yb_number_density_m3(self.yb_at_percent))

    @property
    def spectral_wavelengths_nm(self):
        return _rt_spectra()[0].copy()

    def cross_sections_m2(self, wavelength_nm):
        return self.local_cross_sections_m2(wavelength_nm, self.temperature_K)

    def local_cross_sections_m2(self, wavelength_nm, temperature_K):
        wl, temp = np.broadcast_arrays(np.asarray(wavelength_nm, float),
                                      np.asarray(temperature_K, float))
        if np.any(~np.isfinite(wl)) or np.any((wl < 905) | (wl > 1095)):
            raise ValueError('Yb:YAG room-temperature spectrum covers 905–1095 nm')
        if np.any(~np.isfinite(temp)) or np.any(abs(temp-293.15) > 1e-8):
            raise ValueError('Yb:YAG coupled hot gain is unavailable: temperature-dependent pump spectra are missing; RT data are restricted to 293.15 K')
        grid, absorption, emission = _rt_spectra()
        return np.interp(wl, grid, absorption)*1e-4, np.interp(wl, grid, emission)*1e-4

    def refractive_index(self):
        return float(data.n_yag(self.signal_wavelength_nm))

    @property
    def cavity_phase_index(self):
        return self.refractive_index()

    @property
    def cavity_group_index(self):
        wl = self.signal_wavelength_nm
        derivative = (data.n_yag(wl+.01)-data.n_yag(wl-.01))/.02
        return float(self.refractive_index()-wl*derivative)
