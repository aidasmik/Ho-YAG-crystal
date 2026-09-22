"""Spatially inhomogeneous Ho:YAG concentration model for Stage 3.

The local Ho density is represented as N_Ho(z, y, x). Spectroscopic cross
sections and the split ETU/cross-relaxation coefficients remain at the Rupp
1.1 at.% baseline values unless explicitly replaced in a later stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from .population_state import validate_populations
from .density_statistics import bounded_mean_rms, density_statistics
import numpy as np

from .populations import (
    H,
    C0,
    I7,
    I8,
    HoYAGFourLevelParams,
    MaterialStepResult,
    four_level_rhs,
    pump_absorption_coefficient_m1,
    scale_pulse_to_energy,
)
from .propagation import Grid2D
from .geometry import ThinDiskGeometry, DEFAULT_THIN_DISK_GEOMETRY
from .pulse_train import PulseTrainResult
from .spectroscopy import effective_pump_absorption_cross_section_295K
from .temporal import TimeGrid, propagate_spatiotemporal, spatiotemporal_energy


@dataclass(frozen=True)
class HoDensityField:
    """Voxelized Ho number-density field with shape (nz, ny, nx)."""

    values_m3: np.ndarray
    length_m: float

    def __post_init__(self) -> None:
        values = np.asarray(self.values_m3, dtype=float)
        if values.ndim != 3:
            raise ValueError("values_m3 must have shape (nz, ny, nx)")
        if values.shape[0] < 1:
            raise ValueError("density field must have at least one z slice")
        if self.length_m <= 0:
            raise ValueError("length_m must be positive")
        if np.any(~np.isfinite(values)):
            raise ValueError("density field contains non-finite values")
        if np.any(values < 0):
            raise ValueError("Ho density cannot be negative")
        object.__setattr__(self, "values_m3", values.copy())

    @property
    def nz(self) -> int:
        return int(self.values_m3.shape[0])

    @property
    def dz_m(self) -> float:
        return self.length_m / self.nz

    @property
    def mean_density_m3(self) -> float:
        return float(np.mean(self.values_m3))

    @property
    def min_density_m3(self) -> float:
        return float(np.min(self.values_m3))

    @property
    def max_density_m3(self) -> float:
        return float(np.max(self.values_m3))

    def validate_grid(self, grid: Grid2D) -> None:
        if self.values_m3.shape[1:] != grid.shape:
            raise ValueError(
                f"density transverse shape {self.values_m3.shape[1:]} "
                f"does not match grid {grid.shape}"
            )


def ho_density_statistics(density: HoDensityField):
    """Achieved density statistics, not merely generator input settings."""
    return density_statistics(density.values_m3)


def uniform_ho_density_field(
    grid: Grid2D,
    nz: int,
    length_m: float,
    density_m3: float,
) -> HoDensityField:
    if nz < 1:
        raise ValueError("nz must be >= 1")
    if density_m3 < 0:
        raise ValueError("density_m3 must be nonnegative")
    values = np.full((nz, grid.ny, grid.nx), density_m3, dtype=float)
    return HoDensityField(values, length_m)



def thin_disk_ho_density_field(
    grid: Grid2D,
    nz: int,
    density_m3: float,
    geometry: ThinDiskGeometry = DEFAULT_THIN_DISK_GEOMETRY,
) -> HoDensityField:
    """Uniform Ho density inside a circular thin disk, zero outside.

    The returned field has shape (nz, ny, nx) and length equal to the physical
    disk thickness. The transverse numerical domain may be larger than the disk.
    """
    if nz < 1:
        raise ValueError("nz must be >= 1")
    if density_m3 < 0:
        raise ValueError("density_m3 must be nonnegative")

    mask = geometry.aperture_mask(grid)
    values_2d = np.where(mask, density_m3, 0.0)
    values = np.broadcast_to(
        values_2d[None, :, :], (nz, grid.ny, grid.nx)
    ).copy()
    return HoDensityField(values, geometry.thickness_m)


def axial_linear_ho_density_field(
    grid: Grid2D,
    nz: int,
    length_m: float,
    mean_density_m3: float,
    relative_end_to_end: float,
) -> HoDensityField:
    """Linear z-gradient while preserving the requested discrete mean density.

    relative_end_to_end=0.2 gives approximately 0.9*N at the entrance and
    1.1*N at the exit.
    """
    if nz < 1:
        raise ValueError("nz must be >= 1")
    if not np.isfinite(mean_density_m3) or mean_density_m3 < 0:
        raise ValueError("mean_density_m3 must be nonnegative")
    z = (np.arange(nz) + 0.5) / nz - 0.5
    multiplier = 1.0 + relative_end_to_end * z
    if np.min(multiplier) < 0:
        raise ValueError("requested gradient creates negative Ho density")
    values = mean_density_m3 * multiplier[:, None, None]
    values = np.broadcast_to(values, (nz, grid.ny, grid.nx)).copy()
    return HoDensityField(values, length_m)


def gaussian_ho_density_perturbation(
    grid: Grid2D,
    nz: int,
    length_m: float,
    background_density_m3: float,
    relative_amplitude: float,
    sigma_xy_m: float,
    sigma_z_m: float,
    *,
    x0_m: float = 0.0,
    y0_m: float = 0.0,
    z0_m: float | None = None,
) -> HoDensityField:
    """Add a 3-D Gaussian dopant-rich (>0) or dopant-poor (<0) region."""
    if nz < 1:
        raise ValueError("nz must be >= 1")
    if background_density_m3 < 0:
        raise ValueError("background_density_m3 must be nonnegative")
    if sigma_xy_m <= 0 or sigma_z_m <= 0:
        raise ValueError("Gaussian widths must be positive")

    x, y = grid.mesh
    z = (np.arange(nz) + 0.5) * (length_m / nz)
    if z0_m is None:
        z0_m = length_m / 2.0

    transverse = np.exp(
        -((x - x0_m) ** 2 + (y - y0_m) ** 2) / (2.0 * sigma_xy_m**2)
    )
    axial = np.exp(-(z - z0_m) ** 2 / (2.0 * sigma_z_m**2))
    perturbation = axial[:, None, None] * transverse[None, :, :]
    values = background_density_m3 * (1.0 + relative_amplitude * perturbation)
    if np.min(values) < 0:
        raise ValueError("relative_amplitude creates negative Ho density")
    return HoDensityField(values, length_m)


def smooth_random_ho_density_field(
    grid: Grid2D,
    nz: int,
    length_m: float,
    mean_density_m3: float,
    relative_rms: float,
    *,
    seed: int = 0,
    correlation_fraction: float = 0.15,
    minimum_fraction: float = 0.0,
) -> HoDensityField:
    """Generate a reproducible smooth 3-D random concentration map.

    A Gaussian low-pass filter is applied in Fourier-index space. The map is
    normalized to zero mean/unit RMS before scaling by relative_rms.
    """
    if nz < 2:
        raise ValueError("nz must be >= 2 for a smooth random 3-D field")
    if not np.isfinite(mean_density_m3) or mean_density_m3 < 0:
        raise ValueError("mean_density_m3 must be nonnegative")
    if not np.isfinite(relative_rms) or relative_rms < 0:
        raise ValueError("relative_rms must be nonnegative")
    if not (0 < correlation_fraction <= 1):
        raise ValueError("correlation_fraction must be in (0, 1]")
    if not np.isfinite(minimum_fraction) or not 0 <= minimum_fraction <= 1:
        raise ValueError("minimum_fraction must be nonnegative")

    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(nz, grid.ny, grid.nx))

    kz = np.fft.fftfreq(nz)
    ky = np.fft.fftfreq(grid.ny)
    kx = np.fft.fftfreq(grid.nx)
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
    k2 = KZ**2 + KY**2 + KX**2

    cutoff = max(correlation_fraction, 1e-6)
    filt = np.exp(-0.5 * k2 / cutoff**2)
    smooth = np.fft.ifftn(np.fft.fftn(noise) * filt).real
    smooth -= np.mean(smooth)
    rms = float(np.sqrt(np.mean(smooth**2)))
    if rms == 0:
        raise FloatingPointError("random-field RMS unexpectedly vanished")
    smooth /= rms

    values = bounded_mean_rms(smooth, mean_density_m3, relative_rms, minimum_fraction)
    return HoDensityField(values, length_m)


def _density_slice(density_m3, grid: Grid2D) -> np.ndarray:
    density = np.asarray(density_m3, dtype=float)
    if density.shape != grid.shape:
        raise ValueError(f"density slice shape {density.shape} != {grid.shape}")
    if np.any(~np.isfinite(density)) or np.any(density < 0):
        raise ValueError("density slice must be finite and nonnegative")
    return density


def _ground_state_from_density(density_m3) -> np.ndarray:
    density = np.asarray(density_m3, dtype=float)
    state = np.zeros((4, *density.shape), dtype=float)
    state[I8] = density
    return state


def _physicalize_to_density(state,density_m3):
    return validate_populations(state,density_m3,error_type=FloatingPointError)


def _max_level_fraction(state, density_m3, level: int) -> float:
    density = np.asarray(density_m3, dtype=float)
    active = density > 0
    if not np.any(active):
        return 0.0
    fraction = np.zeros_like(density, dtype=float)
    fraction[active] = state[level][active] / density[active]
    return float(np.max(fraction))


def _recommended_dark_step_for_density(
    params: HoYAGFourLevelParams,
    max_density_m3: float,
) -> float:
    N = max(float(max_density_m3), 0.0)
    rates = (
        params.M56_s1 + 1.0 / params.tau5_s + params.C57_m3_s * N,
        params.M67_s1 + 1.0 / params.tau6_s + params.C67_m3_s * N,
        params.M78_s1 + 1.0 / params.tau7_s,
        2.0 * (params.k75_m3_s + params.k76_m3_s) * N,
        2.0 * (params.C57_m3_s + params.C67_m3_s) * N,
    )
    return 0.4 / max(rates)


def relax_inhomogeneous_populations_dark(
    initial_populations,
    density_m3,
    duration_s: float,
    params: HoYAGFourLevelParams | None = None,
    *,
    max_step_s: float | None = None,
) -> np.ndarray:
    """Dark relaxation while conserving a local N_Ho map exactly."""
    if duration_s < 0:
        raise ValueError("duration_s must be nonnegative")
    params = params or HoYAGFourLevelParams()

    density = np.asarray(density_m3, dtype=float)
    state = np.asarray(initial_populations, dtype=float).copy()
    if state.shape != (4, *density.shape):
        raise ValueError("population and density shapes do not match")
    state = _physicalize_to_density(state, density)

    if duration_s == 0:
        return state

    recommended = _recommended_dark_step_for_density(
        params, float(np.max(density)) if density.size else 0.0
    )
    if max_step_s is None:
        max_step_s = recommended
    if max_step_s <= 0:
        raise ValueError("max_step_s must be positive")
    max_step_s = min(max_step_s, recommended)

    n_steps = max(1, int(np.ceil(duration_s / max_step_s)))
    dt = duration_s / n_steps

    for _ in range(n_steps):
        k1 = four_level_rhs(state, params)
        k2 = four_level_rhs(state + 0.5 * dt * k1, params)
        k3 = four_level_rhs(state + 0.5 * dt * k2, params)
        k4 = four_level_rhs(state + dt * k3, params)
        state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        state = _physicalize_to_density(state, density)

    return state


def pump_material_step_inhomogeneous(
    field,
    grid: Grid2D,
    time: TimeGrid,
    dz_m: float,
    density_m3,
    params: HoYAGFourLevelParams | None = None,
    *,
    sigma_abs_m2: float | None = None,
    sigma_em_m2: float | None = None,
    initial_populations=None,
) -> MaterialStepResult:
    """Apply one z slice with arbitrary transverse Ho density N_Ho(y,x)."""
    if dz_m <= 0:
        raise ValueError("dz_m must be positive")
    params = params or HoYAGFourLevelParams()
    density = _density_slice(density_m3, grid)

    arr = np.asarray(field, dtype=np.complex128)
    expected = (time.nt, grid.ny, grid.nx)
    if arr.shape != expected:
        raise ValueError(f"field shape {arr.shape} != {expected}")

    sigma_abs_m2 = (
        params.sigma_abs_pump_m2 if sigma_abs_m2 is None else sigma_abs_m2
    )
    sigma_em_m2 = (
        params.sigma_em_pump_m2 if sigma_em_m2 is None else sigma_em_m2
    )

    if initial_populations is None:
        state = _ground_state_from_density(density)
    else:
        state = np.asarray(initial_populations, dtype=float).copy()
        if state.shape != (4, grid.ny, grid.nx):
            raise ValueError("initial population shape mismatch")
        state = _physicalize_to_density(state, density)

    photon_energy = H * C0 / params.pump_wavelength_m
    out = np.empty_like(arr)
    peak_i7 = _max_level_fraction(state, density, I7)

    def derivative_and_alpha(s, input_intensity):
        alpha = pump_absorption_coefficient_m1(
            s, sigma_abs_m2, sigma_em_m2
        )
        midpoint_intensity = input_intensity * np.exp(-alpha * dz_m / 2.0)
        photon_flux = midpoint_intensity / photon_energy
        Wa = sigma_abs_m2 * photon_flux
        We = sigma_em_m2 * photon_flux
        return four_level_rhs(s, params, Wa, We), alpha

    for i in range(time.nt):
        I0 = np.abs(arr[i]) ** 2
        k1, alpha = derivative_and_alpha(state, I0)
        out[i] = arr[i] * np.exp(-0.5 * alpha * dz_m)

        if i == time.nt - 1:
            break

        I1 = np.abs(arr[i + 1]) ** 2
        Ih = 0.5 * (I0 + I1)
        dt = time.dt

        k2, _ = derivative_and_alpha(state + 0.5 * dt * k1, Ih)
        k3, _ = derivative_and_alpha(state + 0.5 * dt * k2, Ih)
        k4, _ = derivative_and_alpha(state + dt * k3, I1)

        state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        state = _physicalize_to_density(state, density)
        peak_i7 = max(peak_i7, _max_level_fraction(state, density, I7))

    input_energy = spatiotemporal_energy(arr, grid, time)
    output_energy = spatiotemporal_energy(out, grid, time)

    return MaterialStepResult(
        field_out=out,
        final_populations=state,
        absorbed_energy_J=float(input_energy - output_energy),
        peak_I7_fraction=peak_i7,
    )


@dataclass
class InhomogeneousPumpResult:
    """Population output order is (manifold,z,y,x)."""
    population_axes = ("manifold", "z", "y", "x")
    field_out: np.ndarray
    input_energy_J: float
    output_energy_J: float
    transmission: float
    effective_sigma_abs_m2: float
    absorbed_energy_by_slice_J: np.ndarray
    peak_I7_fraction_by_slice: np.ndarray
    final_populations_by_slice: np.ndarray | None


def propagate_single_pulse_inhomogeneous(
    field,
    grid: Grid2D,
    time: TimeGrid,
    pulse_energy_J: float,
    density: HoDensityField,
    params: HoYAGFourLevelParams | None = None,
    *,
    refractive_index: float = 1.8018686989409411,
    beta2_s2_per_m: float = -4.460681783044079e-26,
    spectral_absorption: bool = True,
    include_passive_propagation: bool = True,
    store_full_populations: bool = False,
) -> InhomogeneousPumpResult:
    """Single-pulse Stage 2P propagation through N_Ho(z,y,x)."""
    params = params or HoYAGFourLevelParams()
    density.validate_grid(grid)

    pulse = scale_pulse_to_energy(field, grid, time, pulse_energy_J)
    input_energy = spatiotemporal_energy(pulse, grid, time)

    if spectral_absorption:
        sigma_abs = effective_pump_absorption_cross_section_295K(
            pulse, time, params.pump_wavelength_m
        )
    else:
        sigma_abs = params.sigma_abs_pump_m2

    absorbed = np.zeros(density.nz)
    peak_i7 = np.zeros(density.nz)
    full = (
        np.empty((4, density.nz, grid.ny, grid.nx), dtype=float)
        if store_full_populations
        else None
    )

    for iz in range(density.nz):
        if include_passive_propagation:
            pulse = propagate_spatiotemporal(
                pulse, grid, time, params.pump_wavelength_m, density.dz_m / 2,
                refractive_index=refractive_index,
                beta2_s2_per_m=beta2_s2_per_m,
            )

        step = pump_material_step_inhomogeneous(
            pulse, grid, time, density.dz_m, density.values_m3[iz], params,
            sigma_abs_m2=sigma_abs,
            sigma_em_m2=params.sigma_em_pump_m2,
        )
        pulse = step.field_out
        absorbed[iz] = step.absorbed_energy_J
        peak_i7[iz] = step.peak_I7_fraction
        if full is not None:
            full[:, iz] = step.final_populations

        if include_passive_propagation:
            pulse = propagate_spatiotemporal(
                pulse, grid, time, params.pump_wavelength_m, density.dz_m / 2,
                refractive_index=refractive_index,
                beta2_s2_per_m=beta2_s2_per_m,
            )

    output_energy = spatiotemporal_energy(pulse, grid, time)
    return InhomogeneousPumpResult(
        field_out=pulse,
        input_energy_J=float(input_energy),
        output_energy_J=float(output_energy),
        transmission=float(output_energy / input_energy),
        effective_sigma_abs_m2=float(sigma_abs),
        absorbed_energy_by_slice_J=absorbed,
        peak_I7_fraction_by_slice=peak_i7,
        final_populations_by_slice=full,
    )


def _population_field_from_density(density: HoDensityField) -> np.ndarray:
    state = np.zeros((4, *density.values_m3.shape), dtype=float)
    state[I8] = density.values_m3
    return state


def _normalize_initial_state_to_density(state, density: HoDensityField) -> np.ndarray:
    state = np.asarray(state, dtype=float).copy()
    expected = (4, *density.values_m3.shape)
    if state.shape != expected:
        raise ValueError(f"initial population shape {state.shape} != {expected}")
    return _physicalize_to_density(state, density.values_m3)


def _max_relative_change(a, b, density_m3) -> float:
    density = np.asarray(density_m3, dtype=float)
    active = density > 0
    if not np.any(active):
        return 0.0
    delta = np.max(np.abs(a - b), axis=0)
    return float(np.max(delta[active] / density[active]))


def simulate_inhomogeneous_pulse_train(
    field,
    grid: Grid2D,
    time: TimeGrid,
    pulse_energy_J: float,
    density: HoDensityField,
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
    """Stage 2R pulse train with a full N_Ho(z,y,x) concentration field."""
    if pulse_energy_J <= 0:
        raise ValueError("pulse_energy_J must be positive")
    if repetition_rate_Hz <= 0:
        raise ValueError("repetition_rate_Hz must be positive")
    if max_pulses < 1 or min_pulses < 1:
        raise ValueError("pulse counts must be >= 1")
    if convergence_tolerance <= 0:
        raise ValueError("convergence_tolerance must be positive")

    params = params or HoYAGFourLevelParams()
    density.validate_grid(grid)

    incident = scale_pulse_to_energy(field, grid, time, pulse_energy_J)
    incident_energy = spatiotemporal_energy(incident, grid, time)

    period_s = 1.0 / repetition_rate_Hz
    pulse_span_s = (time.nt - 1) * time.dt
    dark_time_s = period_s - pulse_span_s
    if dark_time_s < 0:
        raise ValueError("pulse time window exceeds repetition period")

    if spectral_absorption:
        sigma_abs = effective_pump_absorption_cross_section_295K(
            incident, time, params.pump_wavelength_m
        )
    else:
        sigma_abs = params.sigma_abs_pump_m2

    if initial_populations_by_slice is None:
        state = _population_field_from_density(density)
    else:
        state = _normalize_initial_state_to_density(
            initial_populations_by_slice, density
        )

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
            _max_level_fraction(state_before, density.values_m3, I7)
        )

        pulse = incident.copy()
        post = np.empty_like(state_before)

        for iz in range(density.nz):
            if include_passive_propagation:
                pulse = propagate_spatiotemporal(
                    pulse, grid, time, params.pump_wavelength_m,
                    density.dz_m / 2,
                    refractive_index=refractive_index,
                    beta2_s2_per_m=beta2_s2_per_m,
                )

            step = pump_material_step_inhomogeneous(
                pulse,
                grid,
                time,
                density.dz_m,
                density.values_m3[iz],
                params,
                sigma_abs_m2=sigma_abs,
                sigma_em_m2=params.sigma_em_pump_m2,
                initial_populations=state_before[:, iz],
            )
            pulse = step.field_out
            post[:, iz] = step.final_populations

            if include_passive_propagation:
                pulse = propagate_spatiotemporal(
                    pulse, grid, time, params.pump_wavelength_m,
                    density.dz_m / 2,
                    refractive_index=refractive_index,
                    beta2_s2_per_m=beta2_s2_per_m,
                )

        output_energy = spatiotemporal_energy(pulse, grid, time)
        transmission_history.append(float(output_energy / incident_energy))
        absorbed_history.append(float(incident_energy - output_energy))
        post_i7_history.append(
            _max_level_fraction(post, density.values_m3, I7)
        )

        next_state = relax_inhomogeneous_populations_dark(
            post,
            density.values_m3,
            dark_time_s,
            params,
            max_step_s=max_dark_step_s,
        )
        residual = _max_relative_change(
            next_state, state_before, density.values_m3
        )
        convergence_history.append(residual)

        state = next_state
        last_post = post
        last_field_out = pulse

        if pulse_index + 1 >= min_pulses and residual <= convergence_tolerance:
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
