"""Structured 2.09 um signal amplification and depletion for Stage 4.

Stage 4 propagates an arbitrary complex signal field through a stored Ho:YAG
population map. It supports:
  1. weak-signal propagation with fixed populations;
  2. finite-energy signal pulses with stimulated population depletion.
"""

from __future__ import annotations

from dataclasses import dataclass
from .population_state import PopulationField, validate_populations
import numpy as np

from .inhomogeneity import HoDensityField
from .populations import (
    C0,
    H,
    I7,
    I8,
    HoYAGFourLevelParams,
    four_level_rhs,
    scale_pulse_to_energy,
)
from .propagation import Grid2D, angular_spectrum_propagate, optical_power
from .temporal import TimeGrid, propagate_spatiotemporal, spatiotemporal_energy


DEFAULT_SIGNAL_REFRACTIVE_INDEX = 1.799104526293235
DEFAULT_SIGNAL_BETA2_S2_PER_M = -7.78456823900342e-26


def validate_population_field(populations_by_slice, density, grid):
    """Canonical manifold-first field; reject wrong local totals, never guess axes."""
    density.validate_grid(grid)
    if isinstance(populations_by_slice, PopulationField):
        if not np.array_equal(populations_by_slice.density_m3,density.values_m3):
            raise ValueError('PopulationField and geometry densities disagree')
        populations_by_slice=populations_by_slice.values_m3
    return validate_populations(populations_by_slice,density.values_m3)


def small_signal_gain_coefficient_m1(
    populations,
    params: HoYAGFourLevelParams | None = None,
) -> np.ndarray:
    """Return local 2.09 um intensity gain coefficient g in m^-1."""
    params = params or HoYAGFourLevelParams()
    state = np.asarray(populations, dtype=float)
    if state.shape[0] != 4:
        raise ValueError("population axis must have length 4")
    return (
        params.sigma_em_laser_m2 * state[I7]
        - params.sigma_abs_laser_m2 * state[I8]
    )


def laser_transparency_i7_fraction(
    params: HoYAGFourLevelParams | None = None,
) -> float:
    """I7 fraction required for transparency in an I7/I8-only population."""
    params = params or HoYAGFourLevelParams()
    return params.sigma_abs_laser_m2 / (
        params.sigma_abs_laser_m2 + params.sigma_em_laser_m2
    )


@dataclass
class SmallSignalResult:
    field_out: np.ndarray
    input_power: float
    output_power: float
    power_gain: float
    gain_coefficient_by_slice_m1: np.ndarray


def propagate_structured_signal_small_signal(
    field,
    grid: Grid2D,
    populations_by_slice,
    density: HoDensityField,
    params: HoYAGFourLevelParams | None = None,
    *,
    refractive_index: float = DEFAULT_SIGNAL_REFRACTIVE_INDEX,
    include_passive_propagation: bool = True,
) -> SmallSignalResult:
    """Propagate E(y,x) through a fixed population field.

    The material operator is purely real amplitude gain/loss:
        E -> E * exp(g*dz/2)
    so intensity obeys dI/dz = g I.
    """
    params = params or HoYAGFourLevelParams()
    state = validate_population_field(populations_by_slice, density, grid)

    arr = np.asarray(field, dtype=np.complex128)
    if arr.shape != grid.shape:
        raise ValueError(f"field shape {arr.shape} != {grid.shape}")

    input_power = optical_power(arr, grid)
    if input_power <= 0:
        raise ValueError("signal field has zero power")

    signal = arr.copy()
    gains = np.empty((density.nz, grid.ny, grid.nx), dtype=float)

    for iz in range(density.nz):
        if include_passive_propagation:
            signal = angular_spectrum_propagate(
                signal,
                grid,
                params.laser_wavelength_m,
                density.dz_m / 2.0,
                refractive_index=refractive_index,
            )

        g = small_signal_gain_coefficient_m1(state[:, iz], params)
        gains[iz] = g
        signal *= np.exp(0.5 * g * density.dz_m)

        if include_passive_propagation:
            signal = angular_spectrum_propagate(
                signal,
                grid,
                params.laser_wavelength_m,
                density.dz_m / 2.0,
                refractive_index=refractive_index,
            )

    output_power = optical_power(signal, grid)
    return SmallSignalResult(
        field_out=signal,
        input_power=float(input_power),
        output_power=float(output_power),
        power_gain=float(output_power / input_power),
        gain_coefficient_by_slice_m1=gains,
    )


def _physicalize_signal_populations(state,density_m3):
    return validate_populations(state,density_m3,error_type=FloatingPointError)


@dataclass
class SignalMaterialStepResult:
    field_out: np.ndarray
    final_populations: np.ndarray
    signal_energy_change_J: float
    minimum_gain_m1: float
    maximum_gain_m1: float


def signal_material_step(
    field,
    grid: Grid2D,
    time: TimeGrid,
    dz_m: float,
    density_m3,
    initial_populations,
    params: HoYAGFourLevelParams | None = None,
) -> SignalMaterialStepResult:
    """Finite-energy signal interaction for one Ho:YAG z slice."""
    if dz_m <= 0:
        raise ValueError("dz_m must be positive")
    params = params or HoYAGFourLevelParams()

    density = np.asarray(density_m3, dtype=float)
    if density.shape != grid.shape:
        raise ValueError(f"density shape {density.shape} != {grid.shape}")
    if np.any(density < 0):
        raise ValueError("density cannot be negative")

    arr = np.asarray(field, dtype=np.complex128)
    expected = (time.nt, grid.ny, grid.nx)
    if arr.shape != expected:
        raise ValueError(f"field shape {arr.shape} != {expected}")

    state = _physicalize_signal_populations(initial_populations, density)
    photon_energy = H * C0 / params.laser_wavelength_m
    out = np.empty_like(arr)

    min_gain = np.inf
    max_gain = -np.inf

    def derivative_and_gain(s, input_intensity):
        g = small_signal_gain_coefficient_m1(s, params)
        midpoint_intensity = input_intensity * np.exp(g * dz_m / 2.0)
        photon_flux = midpoint_intensity / photon_energy
        Wa = params.sigma_abs_laser_m2 * photon_flux
        We = params.sigma_em_laser_m2 * photon_flux
        derivative = four_level_rhs(
            s,
            params,
            laser_abs_rate_s1=Wa,
            laser_em_rate_s1=We,
        )
        return derivative, g

    for i in range(time.nt):
        I0 = np.abs(arr[i]) ** 2
        k1, g = derivative_and_gain(state, I0)
        out[i] = arr[i] * np.exp(0.5 * g * dz_m)

        min_gain = min(min_gain, float(np.min(g)))
        max_gain = max(max_gain, float(np.max(g)))

        if i == time.nt - 1:
            break

        I1 = np.abs(arr[i + 1]) ** 2
        Ih = 0.5 * (I0 + I1)
        dt = time.dt

        k2, _ = derivative_and_gain(state + 0.5 * dt * k1, Ih)
        k3, _ = derivative_and_gain(state + 0.5 * dt * k2, Ih)
        k4, _ = derivative_and_gain(state + dt * k3, I1)

        state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        state = _physicalize_signal_populations(state, density)

    ein = spatiotemporal_energy(arr, grid, time)
    eout = spatiotemporal_energy(out, grid, time)
    return SignalMaterialStepResult(
        field_out=out,
        final_populations=state,
        signal_energy_change_J=float(eout - ein),
        minimum_gain_m1=float(min_gain),
        maximum_gain_m1=float(max_gain),
    )


@dataclass
class SaturatedSignalResult:
    field_out: np.ndarray
    input_energy_J: float
    output_energy_J: float
    energy_gain: float
    signal_energy_change_by_slice_J: np.ndarray
    final_populations_by_slice: np.ndarray
    small_signal_power_gain_reference: float


def propagate_structured_signal_saturated(
    field,
    grid: Grid2D,
    time: TimeGrid,
    pulse_energy_J: float,
    populations_by_slice,
    density: HoDensityField,
    params: HoYAGFourLevelParams | None = None,
    *,
    refractive_index: float = DEFAULT_SIGNAL_REFRACTIVE_INDEX,
    beta2_s2_per_m: float = DEFAULT_SIGNAL_BETA2_S2_PER_M,
    include_passive_propagation: bool = True,
) -> SaturatedSignalResult:
    """Propagate a finite-energy structured signal pulse with gain depletion."""
    if pulse_energy_J <= 0:
        raise ValueError("pulse_energy_J must be positive")
    params = params or HoYAGFourLevelParams()
    density.validate_grid(grid)
    state = validate_population_field(populations_by_slice, density, grid)

    signal = scale_pulse_to_energy(field, grid, time, pulse_energy_J)
    input_energy = spatiotemporal_energy(signal, grid, time)

    # Use the SAME complex pulse, phase and GVD, freezing populations only.
    # Taking sqrt(fluence) would erase vortex phase and space-time correlations.
    small_ref = propagate_structured_signal_frozen(
        signal, grid, time, pulse_energy_J, state, density, params,
        refractive_index=refractive_index, beta2_s2_per_m=beta2_s2_per_m,
        include_passive_propagation=include_passive_propagation,
    )

    energy_changes = np.zeros(density.nz)
    final = np.empty_like(state)

    for iz in range(density.nz):
        if include_passive_propagation:
            signal = propagate_spatiotemporal(
                signal,
                grid,
                time,
                params.laser_wavelength_m,
                density.dz_m / 2.0,
                refractive_index=refractive_index,
                beta2_s2_per_m=beta2_s2_per_m,
            )

        step = signal_material_step(
            signal,
            grid,
            time,
            density.dz_m,
            density.values_m3[iz],
            state[:, iz],
            params,
        )
        signal = step.field_out
        final[:, iz] = step.final_populations
        energy_changes[iz] = step.signal_energy_change_J

        if include_passive_propagation:
            signal = propagate_spatiotemporal(
                signal,
                grid,
                time,
                params.laser_wavelength_m,
                density.dz_m / 2.0,
                refractive_index=refractive_index,
                beta2_s2_per_m=beta2_s2_per_m,
            )

    output_energy = spatiotemporal_energy(signal, grid, time)
    return SaturatedSignalResult(
        field_out=signal,
        input_energy_J=float(input_energy),
        output_energy_J=float(output_energy),
        energy_gain=float(output_energy / input_energy),
        signal_energy_change_by_slice_J=energy_changes,
        final_populations_by_slice=final,
        small_signal_power_gain_reference=float(small_ref.energy_gain),
    )


@dataclass
class FrozenSignalResult:
    field_out: np.ndarray
    input_energy_J: float
    output_energy_J: float
    energy_gain: float


def propagate_structured_signal_frozen(field, grid, time, pulse_energy_J,
        populations_by_slice, density, params=None, *,
        refractive_index=DEFAULT_SIGNAL_REFRACTIVE_INDEX,
        beta2_s2_per_m=DEFAULT_SIGNAL_BETA2_S2_PER_M, include_passive_propagation=True):
    """Full complex E(t,y,x) weak-signal reference with frozen populations.

    Preserves arbitrary spatial phase, temporal phase and space-time correlations.
    Identical diffraction, GVD, slice geometry and normalization to the saturated
    path. Only the stimulated population update is disabled.
    """
    params=params or HoYAGFourLevelParams()
    state=validate_population_field(populations_by_slice,density,grid)
    signal=scale_pulse_to_energy(field,grid,time,pulse_energy_J)
    ein=spatiotemporal_energy(signal,grid,time)
    for iz in range(density.nz):
        if include_passive_propagation:
            signal=propagate_spatiotemporal(signal,grid,time,params.laser_wavelength_m,density.dz_m/2,
                         refractive_index=refractive_index,beta2_s2_per_m=beta2_s2_per_m)
        gain=small_signal_gain_coefficient_m1(state[:,iz],params)
        signal*=np.exp(.5*gain[None]*density.dz_m)
        if include_passive_propagation:
            signal=propagate_spatiotemporal(signal,grid,time,params.laser_wavelength_m,density.dz_m/2,
                         refractive_index=refractive_index,beta2_s2_per_m=beta2_s2_per_m)
    eout=spatiotemporal_energy(signal,grid,time)
    return FrozenSignalResult(signal,float(ein),float(eout),float(eout/ein))
