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


def _interp_with_explicit_extrapolation(x, xp, fp, extrapolate):
    value = np.interp(x, xp, fp)
    if extrapolate:
        value = np.where(x < xp[0], fp[0] + (x-xp[0]) *
                         (fp[1]-fp[0])/(xp[1]-xp[0]), value)
        value = np.where(x > xp[-1], fp[-1] + (x-xp[-1]) *
                         (fp[-1]-fp[-2])/(xp[-1]-xp[-2]), value)
    return value


def spectral_cross_sections_m2(wavelength_nm, temperature_K, *,
                               extrapolate=False, dataset="canonical_mccumber"):
    """Canonical figure-guided absorption and derived emission (SI units).

    `archived_reconstruction` explicitly selects the independent historical
    emission trace. Outside the archive domain, linear extrapolation is a
    numerical option only and is not validated material behavior.
    """
    import warnings
    wl = np.asarray(wavelength_nm, dtype=float)
    temperature_K = float(temperature_K)
    wavelengths, temperatures, absorption, archived_emission = _spectra()
    if np.any(~np.isfinite(wl)) or not np.isfinite(temperature_K):
        raise ValueError("nonfinite spectral query")
    outside = (np.any((wl < wavelengths[0]) | (wl > wavelengths[-1])) or
               not temperatures[0] <= temperature_K <= temperatures[-1])
    if outside and not extrapolate:
        raise ValueError("spectral query outside 880–1150 nm or 293.15–473.15 K")
    if outside:
        warnings.warn("Yb:LuAG spectral extrapolation is unvalidated", RuntimeWarning,
                      stacklevel=2)
    if dataset not in ("canonical_mccumber", "archived_reconstruction"):
        raise ValueError("unknown Yb:LuAG spectral dataset")

    def sampled(table):
        at_nodes = np.stack([_interp_with_explicit_extrapolation(
            wl, wavelengths, row, extrapolate) for row in table])
        flat = at_nodes.reshape(len(temperatures), -1)
        values = np.array([_interp_with_explicit_extrapolation(
            temperature_K, temperatures, flat[:, j], extrapolate)
            for j in range(flat.shape[1])])
        return values.reshape(wl.shape)

    sigma_abs = sampled(absorption)
    if dataset == "archived_reconstruction":
        sigma_em = sampled(archived_emission)
    else:
        kbt_cm1 = (K_B * temperature_K / (H * C)) / 100.0
        z_ground = sum(np.exp(-energy / kbt_cm1) for energy in GROUND_STARK_CM1)
        z_excited = sum(np.exp(-(energy - EXCITED_STARK_CM1[0]) / kbt_cm1)
                        for energy in EXCITED_STARK_CM1)
        sigma_em = sigma_abs * (z_ground / z_excited) * np.exp(
            (EXCITED_STARK_CM1[0] - 1e7 / wl) / kbt_cm1)
    if np.any(sigma_abs < 0) or np.any(sigma_em < 0):
        raise ValueError("spectral extrapolation produced negative cross section")
    return (float(sigma_abs) if wl.ndim == 0 else sigma_abs,
            float(sigma_em) if wl.ndim == 0 else sigma_em)


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
        return spectral_cross_sections_m2(wavelength_nm, self.temperature_K)

    def local_cross_sections_m2(self, wavelength_nm, temperature_K):
        """Canonical absorption/emission on a local temperature field.

        Uses the reconstructed absorption grid and the same McCumber relation
        as ``cross_sections_m2``. The spatial field is never clamped to the
        archive range; callers must choose an explicit extrapolation model if
        they need one. Concentration changes ion density, not cross sections.
        """
        _positive("wavelength_nm", wavelength_nm)
        wl, temperatures, absorption, _ = _spectra()
        temp = np.asarray(temperature_K, dtype=float)
        if (np.any(~np.isfinite(temp)) or np.any(temp < temperatures[0]-1e-9) or
                np.any(temp > temperatures[-1]+1e-9) or
                not wl[0] <= wavelength_nm <= wl[-1]):
            raise ValueError("local spectral query outside 880–1150 nm or 293.15–473.15 K")
        # Interpolation at an exact archive boundary may differ by a few ulps.
        temp = np.clip(temp, temperatures[0], temperatures[-1])
        absorption_at_wavelength = np.array(
            [np.interp(wavelength_nm, wl, row) for row in absorption])
        sigma_abs = np.interp(temp, temperatures, absorption_at_wavelength)
        kbt_cm1 = (K_B * temp / (H * C)) / 100.0
        z_ground = sum(np.exp(-energy / kbt_cm1) for energy in GROUND_STARK_CM1)
        z_excited = sum(np.exp(-(energy - EXCITED_STARK_CM1[0]) / kbt_cm1)
                        for energy in EXCITED_STARK_CM1)
        sigma_em = sigma_abs * z_ground / z_excited * np.exp(
            (EXCITED_STARK_CM1[0] - 1e7 / wavelength_nm) / kbt_cm1)
        return sigma_abs, sigma_em

    def rates_s1(self, pump_intensity_W_m2, signal_intensity_W_m2,
                 temperature_K=None):
        """Per-ion total upward and downward rates including reabsorption."""
        pump = np.asarray(pump_intensity_W_m2, dtype=float)
        signal = np.asarray(signal_intensity_W_m2, dtype=float)
        if (np.any(~np.isfinite(pump)) or np.any(pump < 0) or
                np.any(~np.isfinite(signal)) or np.any(signal < 0)):
            raise ValueError("intensities must be finite and nonnegative")
        query = (self.cross_sections_m2 if temperature_K is None else
                 lambda wavelength: self.local_cross_sections_m2(wavelength, temperature_K))
        ap, ep = query(self.pump_wavelength_nm)
        a_s, e_s = query(self.signal_wavelength_nm)
        pump_flux = pump / (H * C / (self.pump_wavelength_nm * 1e-9))
        signal_flux = signal / (H * C / (self.signal_wavelength_nm * 1e-9))
        return ap * pump_flux + a_s * signal_flux, ep * pump_flux + e_s * signal_flux

    def excited_fraction_cw(self, pump_intensity_W_m2, signal_intensity_W_m2=0.0,
                            temperature_K=None):
        """Steady solution of dβ/dt = (1-β)Wup - β(Wdown+1/τ)."""
        up, down = self.rates_s1(pump_intensity_W_m2, signal_intensity_W_m2,
                                 temperature_K)
        return up / (up + down + 1.0 / self.lifetime_s)

    def coefficients_m1(self, excited_fraction, temperature_K=None):
        """Scalar-density compatibility API for pump absorption and signal gain."""
        return self.local_coefficients_m1(excited_fraction,
                                          self.number_density_m3,
                                          temperature_K)

    def local_coefficients_m1(self, excited_fraction, density_m3,
                              temperature_K=None):
        """Pump absorption and signal gain from local population, density and T.

        Concentration changes the ion density only. No unsupported direct
        concentration dependence of cross sections or refractive index is
        inferred here.
        """
        beta = np.asarray(excited_fraction, dtype=float)
        density = np.asarray(density_m3, dtype=float)
        if (np.any(~np.isfinite(beta)) or np.any((beta < 0) | (beta > 1)) or
                np.any(~np.isfinite(density)) or np.any(density < 0)):
            raise ValueError("invalid local population or concentration")
        query = (self.cross_sections_m2 if temperature_K is None else
                 lambda wavelength: self.local_cross_sections_m2(wavelength, temperature_K))
        ap, ep = query(self.pump_wavelength_nm)
        a_s, e_s = query(self.signal_wavelength_nm)
        ground = density * (1.0 - beta)
        excited = density * beta
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
