"""Steady multipass pump transport for the Yb:LuAG thin-disk proposal."""

from dataclasses import dataclass

import numpy as np

from .model import YbLuAGMaterial


@dataclass(frozen=True)
class MultipassPumpState:
    excited_fraction_by_slice: np.ndarray
    total_midpoint_intensity_W_m2_by_slice: np.ndarray
    absorbed_pump_W_m2_by_slice: np.ndarray
    final_pump_W_m2: np.ndarray
    iterations: int
    residual: float


def transport_multipass_pump(material: YbLuAGMaterial, pump_in_W_m2,
                             density_scale_by_slice, thickness_m: float,
                             passes: int, excited_fraction_by_slice, *,
                             relay_efficiency: float = 1.0,
                             temperature_K_by_slice=None):
    """Alternate the axial direction of successive ideal reimaged pump passes."""
    scale = np.asarray(density_scale_by_slice, dtype=float)
    beta = np.asarray(excited_fraction_by_slice, dtype=float)
    pump = np.asarray(pump_in_W_m2, dtype=float)
    if (scale.ndim < 1 or beta.shape != scale.shape or
            pump.shape != scale.shape[1:] or
            np.any(~np.isfinite(scale)) or np.any(scale < 0) or
            np.any(~np.isfinite(beta)) or np.any((beta < 0) | (beta > 1)) or
            np.any(~np.isfinite(pump)) or np.any(pump < 0) or
            not np.isfinite(thickness_m) or thickness_m <= 0 or
            isinstance(passes, bool) or not isinstance(passes, int) or passes < 1 or
            not np.isfinite(relay_efficiency) or not 0 < relay_efficiency <= 1):
        raise ValueError("invalid multipass pump geometry or state")
    temperature = (None if temperature_K_by_slice is None else
                   np.broadcast_to(np.asarray(temperature_K_by_slice, dtype=float),
                                   scale.shape))
    dz = thickness_m / scale.shape[0]
    total_midpoint = np.zeros_like(scale)
    absorbed = np.zeros_like(scale)
    current = pump.copy()
    for ipass in range(passes):
        indices = range(scale.shape[0]) if ipass % 2 == 0 else range(scale.shape[0] - 1, -1, -1)
        for iz in indices:
            alpha, _ = material.coefficients_m1(
                beta[iz], None if temperature is None else temperature[iz])
            next_pump = current * np.exp(-alpha * scale[iz] * dz)
            total_midpoint[iz] += 0.5 * (current + next_pump)
            absorbed[iz] += current - next_pump
            current = next_pump
        if ipass < passes - 1:
            current *= relay_efficiency
    return total_midpoint, absorbed, current


def steady_multipass_pump(material: YbLuAGMaterial, pump_in_W_m2,
                          density_scale_by_slice, thickness_m: float,
                          passes: int, *, relay_efficiency: float = 1.0,
                          tolerance: float = 1e-8,
                          max_iterations: int = 200,
                          temperature_K_by_slice=None) -> MultipassPumpState:
    """Converge local CW Yb populations under all pump traversals together."""
    scale = np.asarray(density_scale_by_slice, dtype=float)
    pump = np.asarray(pump_in_W_m2, dtype=float)
    if scale.ndim < 1 or pump.shape != scale.shape[1:]:
        raise ValueError("pump and density-scale shapes disagree")
    beta = np.zeros_like(scale)
    for iteration in range(1, max_iterations + 1):
        midpoint, _, _ = transport_multipass_pump(
            material, pump, scale, thickness_m, passes, beta,
            relay_efficiency=relay_efficiency,
            temperature_K_by_slice=temperature_K_by_slice)
        target = material.excited_fraction_cw(midpoint, temperature_K=temperature_K_by_slice)
        following = 0.5 * beta + 0.5 * target
        residual = float(np.max(abs(following - beta)))
        beta = following
        if residual <= tolerance:
            break
    else:
        raise RuntimeError("multipass Yb pump population did not converge")
    midpoint, absorbed, final = transport_multipass_pump(
        material, pump, scale, thickness_m, passes, beta,
        relay_efficiency=relay_efficiency,
        temperature_K_by_slice=temperature_K_by_slice)
    return MultipassPumpState(beta, midpoint, absorbed, final, iteration, residual)
