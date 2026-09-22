"""Repetitive picosecond-pump population accumulation for Stage 2R."""

from __future__ import annotations

from dataclasses import dataclass
from .population_state import validate_populations
import numpy as np

from .populations import (
    HoYAGFourLevelParams,
    I7,
    ground_state_populations,
    pump_material_step,
    relax_populations_dark,
    scale_pulse_to_energy,
)
from .propagation import Grid2D
from .spectroscopy import effective_pump_absorption_cross_section_295K
from .temporal import TimeGrid, propagate_spatiotemporal, spatiotemporal_energy


@dataclass
class PulseTrainResult:
    population_axes = ("manifold", "z", "y", "x")
    converged: bool
    pulses_simulated: int
    repetition_rate_Hz: float
    period_s: float
    dark_time_s: float
    effective_sigma_abs_m2: float
    transmission_history: np.ndarray
    absorbed_energy_history_J: np.ndarray
    pre_pulse_peak_I7_fraction_history: np.ndarray
    post_pulse_peak_I7_fraction_history: np.ndarray
    convergence_history: np.ndarray
    pre_pulse_populations_by_slice: np.ndarray
    post_pulse_populations_by_slice: np.ndarray
    last_field_out: np.ndarray


def _initial_slice_populations(
    nz: int,
    grid: Grid2D,
    params: HoYAGFourLevelParams,
    initial_populations_by_slice,
) -> np.ndarray:
    expected = (4, nz, grid.ny, grid.nx)
    if initial_populations_by_slice is None:
        state = ground_state_populations(params, (nz, grid.ny, grid.nx))
    else:
        state = np.asarray(initial_populations_by_slice, dtype=float).copy()
        if state.shape != expected:
            raise ValueError(
                f"initial_populations_by_slice shape {state.shape} != {expected}"
            )
        state = validate_populations(state, np.full(expected[1:], params.N_total_m3))
    return state


def simulate_pulse_train_hoyag(
    field,
    grid: Grid2D,
    time: TimeGrid,
    pulse_energy_J: float,
    length_m: float,
    nz: int,
    repetition_rate_Hz: float,
    params: HoYAGFourLevelParams | None = None,
    *,
    refractive_index: float = 1.8018686989409411,
    beta2_s2_per_m: float = -4.460681783044079e-26,
    spectral_absorption: bool = True,
    include_passive_propagation: bool = True,
    initial_populations_by_slice=None,
    max_pulses: int = 2000,
    min_pulses: int = 2,
    convergence_tolerance: float = 1e-6,
    max_dark_step_s: float | None = None,
) -> PulseTrainResult:
    """Iterate identical pump pulses to a periodic population steady state.

    State convention:
      pre-pulse state -> pump pulse -> post-pulse state -> dark relaxation
      -> next pre-pulse state.

    Convergence is the maximum absolute population change between consecutive
    pre-pulse states, normalized by the Ho number density.
    """
    if pulse_energy_J <= 0:
        raise ValueError("pulse_energy_J must be positive")
    if length_m <= 0:
        raise ValueError("length_m must be positive")
    if nz < 1:
        raise ValueError("nz must be >= 1")
    if repetition_rate_Hz <= 0:
        raise ValueError("repetition_rate_Hz must be positive")
    if max_pulses < 1:
        raise ValueError("max_pulses must be >= 1")
    if min_pulses < 1:
        raise ValueError("min_pulses must be >= 1")
    if convergence_tolerance <= 0:
        raise ValueError("convergence_tolerance must be positive")

    params = params or HoYAGFourLevelParams()
    incident = scale_pulse_to_energy(field, grid, time, pulse_energy_J)
    incident_energy = spatiotemporal_energy(incident, grid, time)

    period_s = 1.0 / repetition_rate_Hz
    simulated_pulse_span_s = (time.nt - 1) * time.dt
    dark_time_s = period_s - simulated_pulse_span_s
    if dark_time_s < 0:
        raise ValueError(
            "pulse simulation time window is longer than the pulse repetition period"
        )

    if spectral_absorption:
        sigma_abs = effective_pump_absorption_cross_section_295K(
            incident, time, params.pump_wavelength_m
        )
    else:
        sigma_abs = params.sigma_abs_pump_m2

    state = _initial_slice_populations(
        nz, grid, params, initial_populations_by_slice
    )
    dz = length_m / nz

    transmission_history = []
    absorbed_history = []
    pre_i7_history = []
    post_i7_history = []
    convergence_history = []

    converged = False
    last_post = state.copy()
    last_field_out = incident.copy()

    for pulse_index in range(max_pulses):
        state_before = state.copy()
        pre_i7_history.append(
            float(np.max(state_before[I7]) / params.N_total_m3)
        )

        pulse = incident.copy()
        post = np.empty_like(state_before)

        for iz in range(nz):
            if include_passive_propagation:
                pulse = propagate_spatiotemporal(
                    pulse,
                    grid,
                    time,
                    params.pump_wavelength_m,
                    dz / 2.0,
                    refractive_index=refractive_index,
                    beta2_s2_per_m=beta2_s2_per_m,
                )

            step = pump_material_step(
                pulse,
                grid,
                time,
                dz,
                params,
                sigma_abs_m2=sigma_abs,
                sigma_em_m2=params.sigma_em_pump_m2,
                initial_populations=state_before[:, iz],
            )
            pulse = step.field_out
            post[:, iz] = step.final_populations

            if include_passive_propagation:
                pulse = propagate_spatiotemporal(
                    pulse,
                    grid,
                    time,
                    params.pump_wavelength_m,
                    dz / 2.0,
                    refractive_index=refractive_index,
                    beta2_s2_per_m=beta2_s2_per_m,
                )

        output_energy = spatiotemporal_energy(pulse, grid, time)
        transmission_history.append(float(output_energy / incident_energy))
        absorbed_history.append(float(incident_energy - output_energy))
        post_i7_history.append(
            float(np.max(post[I7]) / params.N_total_m3)
        )

        next_state = relax_populations_dark(
            post,
            dark_time_s,
            params,
            max_step_s=max_dark_step_s,
        )
        residual = float(
            np.max(np.abs(next_state - state_before)) / params.N_total_m3
        )
        convergence_history.append(residual)

        state = next_state
        last_post = post
        last_field_out = pulse

        if (
            pulse_index + 1 >= min_pulses
            and residual <= convergence_tolerance
        ):
            converged = True
            break

    n = len(transmission_history)
    return PulseTrainResult(
        converged=converged,
        pulses_simulated=n,
        repetition_rate_Hz=float(repetition_rate_Hz),
        period_s=float(period_s),
        dark_time_s=float(dark_time_s),
        effective_sigma_abs_m2=float(sigma_abs),
        transmission_history=np.asarray(transmission_history),
        absorbed_energy_history_J=np.asarray(absorbed_history),
        pre_pulse_peak_I7_fraction_history=np.asarray(pre_i7_history),
        post_pulse_peak_I7_fraction_history=np.asarray(post_i7_history),
        convergence_history=np.asarray(convergence_history),
        pre_pulse_populations_by_slice=state,
        post_pulse_populations_by_slice=last_post,
        last_field_out=last_field_out,
    )
