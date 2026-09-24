"""CW Yb:LuAG pump and signal propagation in SI units.

The spectral tables are figure-guided reconstructions, not author-supplied data.
This is a two-manifold, monochromatic, collinear model. It does not imply that
the Ho:YAG four-manifold or coupled resonator solvers apply to Yb:LuAG.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import io

import numpy as np

# NumPy 2 renamed trapz to trapezoid; keep NumPy 1.x working.
trapezoid = getattr(np, "trapezoid", None) or np.trapz

H = 6.62607015e-34
C = 299792458.0
SITE_DENSITY_M3 = 1.42e28  # Lu sites, Beil et al. 2010
K_B = 1.380649e-23
# Kramers-doublet Stark energies in cm^-1, Körner et al. Table 1.
GROUND_STARK_CM1 = (0.0, 600.0, 635.0, 762.0)
EXCITED_STARK_CM1 = (10330.0, 10645.0, 10900.0)
_ROOT = Path(__file__).resolve().parent / "data"
_STEM = "yb_luag_cross_sections_20_200C_reconstructed.npz.part"


@lru_cache(maxsize=1)
def _spectra():
    parts = [_ROOT / f"{_STEM}{i}" for i in range(5)]
    missing = [p for p in parts if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Yb:LuAG spectral archive is incomplete: {missing}")
    with np.load(io.BytesIO(b"".join(p.read_bytes() for p in parts))) as data:
        wavelength = data["wavelength_nm"].astype(float)
        temperature = data["temperature_C"].astype(float) + 273.15
        absorption = data["sigma_abs_cm2"].astype(float) * 1e-4
        emission = data["sigma_em_cm2"].astype(float) * 1e-4
    if (not np.all(np.diff(wavelength) > 0) or
            not np.all(np.diff(temperature) > 0) or
            absorption.shape != (len(temperature), len(wavelength)) or
            emission.shape != absorption.shape or
            not np.all(np.isfinite(absorption)) or
            not np.all(np.isfinite(emission)) or
            np.any(absorption < 0) or np.any(emission < 0)):
        raise ValueError("invalid Yb:LuAG spectral archive")
    return wavelength, temperature, absorption, emission


def _positive(name, value):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class YbLuAGMaterial:
    """A fixed-temperature Yb:LuAG sample and two optical wavelengths.

    The 0.965 ms default lifetime is Beil's pinhole measurement at 10 at.%.
    Other dopings require an explicitly supplied measured lifetime. The optical
    cross sections are reconstructed curves and need sample-specific validation.
    """

    yb_at_percent: float = 10.0
    temperature_K: float = 293.15
    pump_wavelength_nm: float = 940.0
    signal_wavelength_nm: float = 1030.0
    lifetime_s: float | None = None

    def __post_init__(self):
        for name in ("yb_at_percent", "temperature_K", "pump_wavelength_nm",
                     "signal_wavelength_nm"):
            _positive(name, getattr(self, name))
        if self.yb_at_percent > 100:
            raise ValueError("yb_at_percent exceeds Lu-site occupancy")
        wavelength, temperature, _, _ = _spectra()
        if not temperature[0] <= self.temperature_K <= temperature[-1]:
            raise ValueError("spectral temperature must be 293.15–473.15 K")
        for value in (self.pump_wavelength_nm, self.signal_wavelength_nm):
            if not wavelength[0] <= value <= wavelength[-1]:
                raise ValueError("optical wavelength must be 880–1150 nm")
        if self.lifetime_s is None:
            if self.yb_at_percent != 10.0 or self.temperature_K != 293.15:
                raise ValueError("supply a measured lifetime_s outside 10 at.% and 293.15 K")
            object.__setattr__(self, "lifetime_s", 0.965e-3)
        _positive("lifetime_s", self.lifetime_s)

    @property
    def number_density_m3(self):
        return SITE_DENSITY_M3 * self.yb_at_percent / 100.0

    def cross_sections_m2(self, wavelength_nm):
        """Return absorption and McCumber-consistent emission in m².

        The archived emission figure reconstruction violates detailed balance
        away from its peaks. Use the reconstructed absorption and the measured
        Stark energies to obtain emission by reciprocity instead.
        """
        _positive("wavelength_nm", wavelength_nm)
        wavelength, temperature, absorption, _ = _spectra()
        if not wavelength[0] <= wavelength_nm <= wavelength[-1]:
            raise ValueError("optical wavelength must be 880–1150 nm")
        at_nodes = [np.interp(wavelength_nm, wavelength, row) for row in absorption]
        sigma_abs = float(np.interp(self.temperature_K, temperature, at_nodes))
        kbt_cm1 = (K_B * self.temperature_K / (H * C)) / 100.0
        z_ground = sum(np.exp(-energy / kbt_cm1) for energy in GROUND_STARK_CM1)
        z_excited = sum(np.exp(-(energy - EXCITED_STARK_CM1[0]) / kbt_cm1)
                        for energy in EXCITED_STARK_CM1)
        photon_cm1 = 1e7 / wavelength_nm
        sigma_em = sigma_abs * (z_ground / z_excited) * np.exp(
            (EXCITED_STARK_CM1[0] - photon_cm1) / kbt_cm1)
        return sigma_abs, float(sigma_em)

    def rates_s1(self, pump_intensity_W_m2, signal_intensity_W_m2):
        """Per-ion total upward and downward rates including reabsorption."""
        pump = np.asarray(pump_intensity_W_m2, dtype=float)
        signal = np.asarray(signal_intensity_W_m2, dtype=float)
        if (np.any(~np.isfinite(pump)) or np.any(pump < 0) or
                np.any(~np.isfinite(signal)) or np.any(signal < 0)):
            raise ValueError("intensities must be finite and nonnegative")
        ap, ep = self.cross_sections_m2(self.pump_wavelength_nm)
        a_s, e_s = self.cross_sections_m2(self.signal_wavelength_nm)
        pump_flux = pump / (H * C / (self.pump_wavelength_nm * 1e-9))
        signal_flux = signal / (H * C / (self.signal_wavelength_nm * 1e-9))
        return ap * pump_flux + a_s * signal_flux, ep * pump_flux + e_s * signal_flux

    def excited_fraction_cw(self, pump_intensity_W_m2, signal_intensity_W_m2=0.0):
        """Steady solution of dβ/dt = (1-β)Wup - β(Wdown+1/τ)."""
        up, down = self.rates_s1(pump_intensity_W_m2, signal_intensity_W_m2)
        return up / (up + down + 1.0 / self.lifetime_s)

    def coefficients_m1(self, excited_fraction):
        beta = np.asarray(excited_fraction, dtype=float)
        if np.any(~np.isfinite(beta)) or np.any((beta < 0) | (beta > 1)):
            raise ValueError("excited fraction must be in [0, 1]")
        ap, ep = self.cross_sections_m2(self.pump_wavelength_nm)
        a_s, e_s = self.cross_sections_m2(self.signal_wavelength_nm)
        ground = self.number_density_m3 * (1.0 - beta)
        excited = self.number_density_m3 * beta
        return ap * ground - ep * excited, e_s * excited - a_s * ground

    def transparency_fraction(self):
        a_s, e_s = self.cross_sections_m2(self.signal_wavelength_nm)
        return a_s / (a_s + e_s)

    def refractive_index(self):
        """Undoped LuAG host dispersion at the signal wavelength (Hrabovský)."""
        wavelength_um = self.signal_wavelength_nm / 1000.0
        n_squared = (2.077 + 1.237 * wavelength_um**2 /
                     (wavelength_um**2 - 0.1376**2) -
                     0.0104 * wavelength_um**2)
        return float(np.sqrt(n_squared))


@dataclass(frozen=True)
class CWResult:
    pump_out_W_m2: np.ndarray
    signal_out_W_m2: np.ndarray
    absorbed_pump_W_m2: np.ndarray
    signal_change_W_m2: np.ndarray
    excited_fraction_by_step: np.ndarray
    fluorescence_W_m2: np.ndarray | None
    heat_W_m2: np.ndarray | None
    heat_W_m3_by_step: np.ndarray | None
    fluorescence_wavelength_nm_used: float | None
    scope: str = "collinear monochromatic CW, fixed-temperature, no diffraction or cavity feedback"


def propagate_cw(material: YbLuAGMaterial, thickness_m: float, steps: int,
                 pump_in_W_m2, signal_in_W_m2=0.0, *,
                 fluorescence_quantum_yield: float | None = None,
                 mean_fluorescence_wavelength_nm: float | None = None) -> CWResult:
    """Propagate co-directed pump and signal through one homogeneous disk.

    Inputs may be scalars or matching transverse intensity maps. At each z step,
    the local CW population is recomputed at midpoint intensity. Optional heat
    is an area-integrated steady-state first-law estimate. Its fluorescence
    yield and mean photon wavelength must be supplied for the sample.
    """
    _positive("thickness_m", thickness_m)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if fluorescence_quantum_yield is None and mean_fluorescence_wavelength_nm is not None:
        raise ValueError("supply fluorescence_quantum_yield with the wavelength")
    if fluorescence_quantum_yield is not None:
        if not np.isfinite(fluorescence_quantum_yield) or not 0 <= fluorescence_quantum_yield <= 1:
            raise ValueError("fluorescence_quantum_yield must be in [0, 1]")
        if mean_fluorescence_wavelength_nm is None:
            from .fluorescence import fluorescence_spectrum
            mean_fluorescence_wavelength_nm = fluorescence_spectrum(
                material).energy_equivalent_wavelength_nm
        else:
            _positive("mean_fluorescence_wavelength_nm", mean_fluorescence_wavelength_nm)
    pump = np.asarray(pump_in_W_m2, dtype=float)
    signal = np.asarray(signal_in_W_m2, dtype=float)
    if pump.shape != signal.shape or np.any(~np.isfinite(pump)) or np.any(~np.isfinite(signal)) or np.any(pump < 0) or np.any(signal < 0):
        raise ValueError("pump and signal must have matching, finite, nonnegative intensities")
    pump = pump.copy()
    signal = signal.copy()
    original_pump = pump.copy()
    original_signal = signal.copy()
    dz = thickness_m / steps
    fractions = []
    fluorescence = np.zeros_like(pump) if fluorescence_quantum_yield is not None else None
    heat_by_step = [] if fluorescence is not None else None
    for _ in range(steps):
        beta0 = material.excited_fraction_cw(pump, signal)
        alpha0, gain0 = material.coefficients_m1(beta0)
        pump_mid = pump * np.exp(-0.5 * alpha0 * dz)
        signal_mid = signal * np.exp(0.5 * gain0 * dz)
        beta = material.excited_fraction_cw(pump_mid, signal_mid)
        alpha, gain = material.coefficients_m1(beta)
        next_pump = pump * np.exp(-alpha * dz)
        next_signal = signal * np.exp(gain * dz)
        fractions.append(beta)
        if fluorescence is not None:
            photon_energy = H * C / (mean_fluorescence_wavelength_nm * 1e-9)
            emitted = (material.number_density_m3 * beta /
                       material.lifetime_s * fluorescence_quantum_yield *
                       photon_energy * dz)
            fluorescence += emitted
            heat_by_step.append(((pump - next_pump) - (next_signal - signal) - emitted) / dz)
        pump, signal = next_pump, next_signal
    absorbed = original_pump - pump
    signal_change = signal - original_signal
    heat = absorbed - signal_change - fluorescence if fluorescence is not None else None
    return CWResult(pump, signal, absorbed, signal_change,
                    np.stack(fractions, axis=0), fluorescence, heat,
                    np.stack(heat_by_step, axis=0) if heat_by_step is not None else None,
                    mean_fluorescence_wavelength_nm)
