"""Transient four-manifold Ho:YAG pump/population model for Stage 2P."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .population_state import validate_populations
from .propagation import Grid2D
from .temporal import TimeGrid, propagate_spatiotemporal, spatiotemporal_energy
from .spectroscopy import effective_pump_absorption_cross_section_295K

C0 = 299_792_458.0
H = 6.62607015e-34

I5, I6, I7, I8 = 0, 1, 2, 3


@dataclass(frozen=True)
class HoYAGFourLevelParams:
    """Rupp et al. 1.1 at.% Ho:YAG baseline in SI units.

    Spectroscopic rates are fixed baseline values; a hot temperature field does
    not by itself establish temperature-dependent cross sections or lifetimes.
    """

    N_total_m3: float = 1.52e26

    tau5_s: float = 4.4e-3
    tau6_s: float = 3.5e-3
    tau7_s: float = 7.9e-3

    beta56: float = 0.064
    beta57: float = 0.542
    beta58: float = 0.394
    beta67: float = 0.142
    beta68: float = 0.858
    beta78: float = 1.0

    M56_s1: float = 7.6e5
    M67_s1: float = 2.2e4
    M78_s1: float = 20.9

    k75_m3_s: float = 3.8e-24
    k76_m3_s: float = 4.0e-25
    C57_m3_s: float = 1.1e-23
    C67_m3_s: float = 2.6e-24

    pump_wavelength_m: float = 1.9077e-6
    sigma_abs_pump_m2: float = 1.2e-24
    sigma_em_pump_m2: float = 7.7e-25

    laser_wavelength_m: float = 2.0903e-6
    sigma_abs_laser_m2: float = 2.1e-25
    sigma_em_laser_m2: float = 1.2e-24

    def __post_init__(self) -> None:
        for name in ('N_total_m3','tau5_s','tau6_s','tau7_s','pump_wavelength_m','laser_wavelength_m'):
            value=getattr(self,name)
            if not np.isscalar(value) or not np.isfinite(value) or value<=0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('M56_s1','M67_s1','M78_s1','k75_m3_s','k76_m3_s','C57_m3_s','C67_m3_s',
                     'sigma_abs_pump_m2','sigma_em_pump_m2','sigma_abs_laser_m2','sigma_em_laser_m2'):
            value=getattr(self,name)
            if not np.isscalar(value) or not np.isfinite(value) or value<0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for name in ('beta56','beta57','beta58','beta67','beta68','beta78'):
            value=getattr(self,name)
            if not np.isscalar(value) or not np.isfinite(value) or not 0<=value<=1:
                raise ValueError(f'{name} must be finite and in [0,1]')
        if not np.isclose(self.beta56+self.beta57+self.beta58,1.,rtol=0,atol=1e-12):
            raise ValueError('I5 branching ratios must sum to one')
        if not np.isclose(self.beta67+self.beta68,1.,rtol=0,atol=1e-12):
            raise ValueError('I6 branching ratios must sum to one')
        if not np.isclose(self.beta78,1.,rtol=0,atol=1e-12):
            raise ValueError('this four-manifold RHS requires beta78=1')

    @classmethod
    def from_stage0_dict(cls, data: dict) -> "HoYAGFourLevelParams":
        s = data["spectroscopy"]
        return cls(
            N_total_m3=data["composition"]["ho_number_density_m-3"],
            tau5_s=s["spontaneous_lifetimes_s"]["I5"],
            tau6_s=s["spontaneous_lifetimes_s"]["I6"],
            tau7_s=s["spontaneous_lifetimes_s"]["I7"],
            beta56=s["branching_ratios"]["beta_I5_to_I6"],
            beta57=s["branching_ratios"]["beta_I5_to_I7"],
            beta58=s["branching_ratios"]["beta_I5_to_I8"],
            beta67=s["branching_ratios"]["beta_I6_to_I7"],
            beta68=s["branching_ratios"]["beta_I6_to_I8"],
            beta78=s["branching_ratios"]["beta_I7_to_I8"],
            M56_s1=s["multiphonon_relaxation_rates_s-1"]["M_I5_to_I6"],
            M67_s1=s["multiphonon_relaxation_rates_s-1"]["M_I6_to_I7"],
            M78_s1=s["multiphonon_relaxation_rates_s-1"]["M_I7_to_I8"],
            k75_m3_s=s["etu_and_cross_relaxation_m3_s"]["k_I7_to_I5"],
            k76_m3_s=s["etu_and_cross_relaxation_m3_s"]["k_I7_to_I6"],
            C57_m3_s=s["etu_and_cross_relaxation_m3_s"]["C_I5_to_I7"],
            C67_m3_s=s["etu_and_cross_relaxation_m3_s"]["C_I6_to_I7"],
            pump_wavelength_m=data["operating_point"]["pump_wavelength_m"],
            sigma_abs_pump_m2=s["cross_sections_m2"]["sigma_absorption_pump"],
            sigma_em_pump_m2=s["cross_sections_m2"]["sigma_emission_pump"],
            laser_wavelength_m=data["operating_point"]["laser_wavelength_m"],
            sigma_abs_laser_m2=s["cross_sections_m2"]["sigma_absorption_laser"],
            sigma_em_laser_m2=s["cross_sections_m2"]["sigma_emission_laser"],
        )


def ground_state_populations(params: HoYAGFourLevelParams, shape=()) -> np.ndarray:
    populations = np.zeros((4, *tuple(shape)), dtype=float)
    populations[I8] = params.N_total_m3
    return populations


def stimulated_rates_from_intensity(intensity_W_m2, wavelength_m, sigma_abs_m2, sigma_em_m2):
    """Local nonnegative stimulated rates in s^-1 from physical intensity."""
    if not np.isfinite(wavelength_m) or wavelength_m<=0:
        raise ValueError('wavelength must be finite and positive')
    for sigma in (sigma_abs_m2,sigma_em_m2):
        if not np.isfinite(sigma) or sigma<0:
            raise ValueError('cross sections must be finite and nonnegative')
    intensity=np.asarray(intensity_W_m2,float)
    if np.any(~np.isfinite(intensity)) or np.any(intensity<0):
        raise ValueError('intensity must be finite and nonnegative')
    flux=intensity/(H*C0/wavelength_m)
    return sigma_abs_m2*flux,sigma_em_m2*flux


def four_level_rhs(
    populations,
    params: HoYAGFourLevelParams,
    pump_abs_rate_s1=0.0,
    pump_em_rate_s1=0.0,
    laser_abs_rate_s1=0.0,
    laser_em_rate_s1=0.0,
) -> np.ndarray:
    """Rupp et al. Eqs. (4)-(7), vectorized over all trailing dimensions."""
    n = np.asarray(populations, dtype=float)
    if n.shape[0] != 4:
        raise ValueError("population axis must have length 4")

    N5, N6, N7, N8 = n
    Wabs = np.asarray(pump_abs_rate_s1) + np.asarray(laser_abs_rate_s1)
    Wemi = np.asarray(pump_em_rate_s1) + np.asarray(laser_em_rate_s1)

    d5 = (
        N7**2 * params.k75_m3_s
        - N5 * (params.C57_m3_s * N8 + params.M56_s1 + 1.0 / params.tau5_s)
    )

    d6 = (
        N7**2 * params.k76_m3_s
        - N6 * (params.C67_m3_s * N8 + params.M67_s1 + 1.0 / params.tau6_s)
        + N5 * (params.M56_s1 + params.beta56 / params.tau5_s)
    )

    d7 = (
        -2.0 * N7**2 * (params.k76_m3_s + params.k75_m3_s)
        - N7 * (params.M78_s1 + 1.0 / params.tau7_s)
        + N6
        * (
            2.0 * params.C67_m3_s * N8
            + params.M67_s1
            + params.beta67 / params.tau6_s
        )
        + N5 * (2.0 * params.C57_m3_s * N8 + params.beta57 / params.tau5_s)
        + N8 * Wabs
        - N7 * Wemi
    )

    d8 = (
        N7**2 * (params.k76_m3_s + params.k75_m3_s)
        - N8 * (params.C57_m3_s * N5 + params.C67_m3_s * N6)
        + N7 * (params.M78_s1 + 1.0 / params.tau7_s)
        + N6 * params.beta68 / params.tau6_s
        + N5 * params.beta58 / params.tau5_s
        - N8 * Wabs
        + N7 * Wemi
    )

    return np.stack((d5, d6, d7, d8), axis=0)


def pump_absorption_coefficient_m1(
    populations,
    sigma_abs_m2: float,
    sigma_em_m2: float,
):
    n = np.asarray(populations, dtype=float)
    return sigma_abs_m2 * n[I8] - sigma_em_m2 * n[I7]


def laser_gain_coefficient_m1(populations, params: HoYAGFourLevelParams):
    n = np.asarray(populations, dtype=float)
    return params.sigma_em_laser_m2 * n[I7] - params.sigma_abs_laser_m2 * n[I8]


def scale_pulse_to_energy(
    field,
    grid: Grid2D,
    time: TimeGrid,
    pulse_energy_J: float,
) -> np.ndarray:
    """Scale E(tau,y,x) so |E|^2 is W/m^2 and integrates to pulse energy."""
    if not np.isfinite(pulse_energy_J) or pulse_energy_J <= 0:
        raise ValueError("pulse_energy_J must be positive")
    arr = np.asarray(field, dtype=np.complex128)
    current = spatiotemporal_energy(arr, grid, time)
    if current <= 0:
        raise ValueError("field has zero energy")
    return arr * np.sqrt(pulse_energy_J / current)


def _physicalize_populations(state, params):
    n=np.asarray(state,float)
    if n.ndim<1:
        raise FloatingPointError('population axis is missing')
    density=np.full(n.shape[1:],params.N_total_m3)
    return validate_populations(n,density,error_type=FloatingPointError)


def _rk4_population_step(
    state,
    I0,
    I1,
    dt,
    params,
    wavelength_m,
    sigma_abs_m2,
    sigma_em_m2,
):
    Imid = 0.5 * (I0 + I1)

    def derivative(s, intensity):
        Wa, We = stimulated_rates_from_intensity(
            intensity, wavelength_m, sigma_abs_m2, sigma_em_m2
        )
        return four_level_rhs(s, params, Wa, We)

    k1 = derivative(state, I0)
    k2 = derivative(state + 0.5 * dt * k1, Imid)
    k3 = derivative(state + 0.5 * dt * k2, Imid)
    k4 = derivative(state + dt * k3, I1)
    updated = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return _physicalize_populations(updated, params)


def integrate_populations(
    intensity_W_m2,
    time: TimeGrid,
    params: HoYAGFourLevelParams | None = None,
    *,
    wavelength_m: float | None = None,
    sigma_abs_m2: float | None = None,
    sigma_em_m2: float | None = None,
    initial_populations=None,
) -> np.ndarray:
    """Integrate local four-manifold populations over one pulse."""
    params = params or HoYAGFourLevelParams()
    intensity = np.asarray(intensity_W_m2, dtype=float)
    if intensity.shape[0] != time.nt:
        raise ValueError("first intensity axis must match time.nt")
    if np.any(~np.isfinite(intensity)) or np.any(intensity < 0):
        raise ValueError("intensity must be nonnegative")

    spatial_shape = intensity.shape[1:]
    if initial_populations is None:
        state = ground_state_populations(params, spatial_shape)
    else:
        state = np.asarray(initial_populations, dtype=float).copy()
        if state.shape != (4, *spatial_shape):
            raise ValueError("initial population shape mismatch")

    wavelength_m = params.pump_wavelength_m if wavelength_m is None else wavelength_m
    sigma_abs_m2 = params.sigma_abs_pump_m2 if sigma_abs_m2 is None else sigma_abs_m2
    sigma_em_m2 = params.sigma_em_pump_m2 if sigma_em_m2 is None else sigma_em_m2

    state = _physicalize_populations(state, params)

    for i in range(time.nt - 1):
        state = _rk4_population_step(
            state,
            intensity[i],
            intensity[i + 1],
            time.dt,
            params,
            wavelength_m,
            sigma_abs_m2,
            sigma_em_m2,
        )
    return state


def recommended_dark_relaxation_step_s(params: HoYAGFourLevelParams) -> float:
    """Conservative explicit-RK4 step for zero-light Ho:YAG relaxation.

    The estimate includes the fastest linear decay/cross-relaxation scale at
    N_total and the largest quadratic ETU scale. It is a numerical safety
    estimate, not an additional physical parameter.
    """
    N = params.N_total_m3
    rates = (
        params.M56_s1 + 1.0 / params.tau5_s + params.C57_m3_s * N,
        params.M67_s1 + 1.0 / params.tau6_s + params.C67_m3_s * N,
        params.M78_s1 + 1.0 / params.tau7_s,
        2.0 * (params.k75_m3_s + params.k76_m3_s) * N,
        2.0 * (params.C57_m3_s + params.C67_m3_s) * N,
    )
    return 0.4 / max(rates)


def relax_populations_dark(
    initial_populations,
    duration_s: float,
    params: HoYAGFourLevelParams | None = None,
    *,
    max_step_s: float | None = None,
) -> np.ndarray:
    """Evolve Ho populations with no pump or laser field.

    The routine is vectorized over every trailing dimension, so a full
    population field with shape (4, nz, ny, nx) can be relaxed at once.
    """
    if duration_s < 0:
        raise ValueError("duration_s must be nonnegative")

    params = params or HoYAGFourLevelParams()
    state = np.asarray(initial_populations, dtype=float).copy()
    if state.shape[0] != 4:
        raise ValueError("population axis must have length 4")
    state = _physicalize_populations(state, params)

    if duration_s == 0:
        return state

    recommended_step = recommended_dark_relaxation_step_s(params)
    if max_step_s is None:
        max_step_s = recommended_step
    if max_step_s <= 0:
        raise ValueError("max_step_s must be positive")

    # A caller may request a smaller step for accuracy, but never a larger
    # step than the conservative baseline stability estimate.
    max_step_s = min(max_step_s, recommended_step)

    n_steps = max(1, int(np.ceil(duration_s / max_step_s)))
    dt = duration_s / n_steps

    for _ in range(n_steps):
        k1 = four_level_rhs(state, params)
        k2 = four_level_rhs(state + 0.5 * dt * k1, params)
        k3 = four_level_rhs(state + 0.5 * dt * k2, params)
        k4 = four_level_rhs(state + dt * k3, params)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        state = _physicalize_populations(state, params)

    return state


@dataclass
class MaterialStepResult:
    field_out: np.ndarray
    final_populations: np.ndarray
    absorbed_energy_J: float
    peak_I7_fraction: float


def pump_material_step(
    field,
    grid: Grid2D,
    time: TimeGrid,
    dz_m: float,
    params: HoYAGFourLevelParams | None = None,
    *,
    sigma_abs_m2: float | None = None,
    sigma_em_m2: float | None = None,
    initial_populations=None,
) -> MaterialStepResult:
    """Apply one homogeneous Ho:YAG slice to a physical pump envelope."""
    if dz_m <= 0:
        raise ValueError("dz_m must be positive")
    params = params or HoYAGFourLevelParams()
    arr = np.asarray(field, dtype=np.complex128)
    expected = (time.nt, grid.ny, grid.nx)
    if arr.shape != expected:
        raise ValueError(f"field shape {arr.shape} != {expected}")

    sigma_abs_m2 = params.sigma_abs_pump_m2 if sigma_abs_m2 is None else sigma_abs_m2
    sigma_em_m2 = params.sigma_em_pump_m2 if sigma_em_m2 is None else sigma_em_m2

    if initial_populations is None:
        state = ground_state_populations(params, grid.shape)
    else:
        state = np.asarray(initial_populations, dtype=float).copy()
        if state.shape != (4, grid.ny, grid.nx):
            raise ValueError("initial population shape mismatch")

    state = _physicalize_populations(state, params)

    photon_energy = H * C0 / params.pump_wavelength_m
    out = np.empty_like(arr)
    peak_i7_fraction = float(np.max(state[I7]) / params.N_total_m3)

    def derivative_and_alpha(s, input_intensity):
        alpha = pump_absorption_coefficient_m1(s, sigma_abs_m2, sigma_em_m2)
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

        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        state = _physicalize_populations(state, params)
        peak_i7_fraction = max(
            peak_i7_fraction,
            float(np.max(state[I7]) / params.N_total_m3),
        )

    input_energy = spatiotemporal_energy(arr, grid, time)
    output_energy = spatiotemporal_energy(out, grid, time)

    return MaterialStepResult(
        field_out=out,
        final_populations=state,
        absorbed_energy_J=float(input_energy - output_energy),
        peak_I7_fraction=peak_i7_fraction,
    )


@dataclass
class PumpPropagationResult:
    """Population output order is (manifold,z,y,x), not the legacy z-major order."""
    population_axes = ("manifold", "z", "y", "x")
    field_out: np.ndarray
    input_energy_J: float
    output_energy_J: float
    transmission: float
    effective_sigma_abs_m2: float
    absorbed_energy_by_slice_J: np.ndarray
    peak_I7_fraction_by_slice: np.ndarray
    final_populations_by_slice: np.ndarray | None


def propagate_single_pulse_hoyag(
    field,
    grid: Grid2D,
    time: TimeGrid,
    pulse_energy_J: float,
    length_m: float,
    nz: int,
    params: HoYAGFourLevelParams | None = None,
    *,
    refractive_index: float = 1.8018686989409411,
    beta2_s2_per_m: float = -4.460681783044079e-26,
    spectral_absorption: bool = True,
    include_passive_propagation: bool = True,
    store_full_populations: bool = False,
) -> PumpPropagationResult:
    """Propagate one pulse through homogeneous Ho:YAG with transient saturation.

    spectral_absorption=True uses a spectrum-weighted effective sigma_a.
    This captures bandwidth-dependent overlap but not frequency-resolved
    spectral reshaping inside the saturated crystal.
    """
    if length_m <= 0:
        raise ValueError("length_m must be positive")
    if nz < 1:
        raise ValueError("nz must be >= 1")

    params = params or HoYAGFourLevelParams()
    pulse = scale_pulse_to_energy(field, grid, time, pulse_energy_J)
    input_energy = spatiotemporal_energy(pulse, grid, time)

    if spectral_absorption:
        sigma_abs = effective_pump_absorption_cross_section_295K(
            pulse, time, params.pump_wavelength_m
        )
    else:
        sigma_abs = params.sigma_abs_pump_m2

    dz = length_m / nz
    absorbed = np.zeros(nz)
    peak_i7 = np.zeros(nz)
    full = (
        np.empty((4, nz, grid.ny, grid.nx), dtype=float)
        if store_full_populations
        else None
    )

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
        )
        pulse = step.field_out
        absorbed[iz] = step.absorbed_energy_J
        peak_i7[iz] = step.peak_I7_fraction
        if full is not None:
            full[:, iz] = step.final_populations

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
    return PumpPropagationResult(
        field_out=pulse,
        input_energy_J=float(input_energy),
        output_energy_J=float(output_energy),
        transmission=float(output_energy / input_energy),
        effective_sigma_abs_m2=float(sigma_abs),
        absorbed_energy_by_slice_J=absorbed,
        peak_I7_fraction_by_slice=peak_i7,
        final_populations_by_slice=full,
    )
