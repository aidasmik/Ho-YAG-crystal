"""Weak coherent signal through a CW pumped Yb:LuAG disk."""

from dataclasses import dataclass

import numpy as np

from hoyag.propagation import Grid2D, angular_spectrum_propagate, optical_power

from .model import YbLuAGMaterial, propagate_cw


@dataclass(frozen=True)
class StructuredSignalResult:
    field_out: np.ndarray
    input_power_W: float
    output_power_W: float
    excited_fraction_by_step: np.ndarray
    pump_out_W_m2: np.ndarray
    scope: str = "small-signal coherent field; pump-only CW population; no signal depletion"


def propagate_structured_small_signal(material: YbLuAGMaterial, field_in,
                                      grid: Grid2D, pump_in_W_m2,
                                      thickness_m: float, steps: int,
                                      *, include_diffraction: bool = True):
    """Split-step scalar field, with |field|² in W/m² and pump-only inversion.

    Uses the existing angular-spectrum kernel. The signal does not alter the
    population, so this must not be used for high-extraction predictions.
    """
    field = np.asarray(field_in, dtype=complex)
    pump = np.asarray(pump_in_W_m2, dtype=float)
    if field.shape != grid.shape or pump.shape != grid.shape:
        raise ValueError("field and pump maps must match the transverse grid")
    if np.any(~np.isfinite(field)) or np.any(~np.isfinite(pump)) or np.any(pump < 0):
        raise ValueError("field and pump must be finite; pump must be nonnegative")
    pump_result = propagate_cw(material, thickness_m, steps, pump, np.zeros_like(pump))
    initial_power = optical_power(field, grid)
    dz = thickness_m / steps
    wavelength_m = material.signal_wavelength_nm * 1e-9
    refractive_index = material.refractive_index()
    result = field.copy()
    for beta in pump_result.excited_fraction_by_step:
        if include_diffraction:
            result = angular_spectrum_propagate(result, grid, wavelength_m, dz / 2,
                                                refractive_index=refractive_index)
        _, gain = material.coefficients_m1(beta)
        result *= np.exp(0.5 * gain * dz)
        if include_diffraction:
            result = angular_spectrum_propagate(result, grid, wavelength_m, dz / 2,
                                                refractive_index=refractive_index)
    return StructuredSignalResult(result, initial_power, optical_power(result, grid),
                                  pump_result.excited_fraction_by_step,
                                  pump_result.pump_out_W_m2)
