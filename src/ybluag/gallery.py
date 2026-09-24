"""Separate Yb:LuAG structured-field calculations for the local browser app.

The beam and ideal phase-mask generators are shared with the Ho gallery; all
gain, population, and heat quantities here use the Yb two-manifold model.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from hoyag.propagation import Grid2D, angular_spectrum_propagate, optical_power
from hoyag.resonator import ThinDiskResonator
from hoyag.structured_beam_gallery import (BEAM_NAMES, PHASE_MASKS, GallerySettings,
                                            _cartesian_to_polar,
                                            nonuniform_density, phase_pattern)
from hoyag.thermal import DiskThermalMesh

from .fluorescence import fluorescence_spectrum
from .model import YbLuAGMaterial
from .assembly import (solve_yb_assembly, solve_yb_cooler_temperature,
                       yb_cooler_solver)
from .pulsed import propagate_pulse
from .multipass_pump import steady_multipass_pump, transport_multipass_pump
from .beam_shaping import gaussian_seed_and_target_mask


@dataclass(frozen=True)
class YbGallerySettings:
    pump_power_W: float = 5.0
    pump_radius_m: float = 0.6e-3
    thickness_m: float = 150e-6
    input_power_W: float = 1.0
    waist_m: float = 0.408e-3
    post_disk_distance_m: float = 0.25
    slm_to_disk_distance_m: float = 0.25
    phase_mask_name: str = "none"
    phase_strength_rad: float = math.pi
    density_seed: int = 17
    cluster_count: int = 24
    cluster_contrast: float = 0.27
    cluster_radius_min_m: float = 0.20e-3
    cluster_radius_max_m: float = 1.25e-3
    solver_mode: str = "weak_probe"
    fluorescence_escape_yield: float = 0.0
    grid_n: int = 64
    field_size_m: float = 8e-3
    z_steps: int = 4

    def __post_init__(self):
        if self.phase_mask_name not in PHASE_MASKS:
            raise ValueError("unknown phase mask")
        if self.solver_mode not in ("weak_probe", "saturated_cw", "modal_cw"):
            raise ValueError("unknown Yb structured solver mode")
        for name in ("pump_power_W", "pump_radius_m", "thickness_m", "input_power_W",
                     "waist_m"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (not np.isfinite(self.post_disk_distance_m) or self.post_disk_distance_m < 0 or
                not np.isfinite(self.slm_to_disk_distance_m) or
                self.slm_to_disk_distance_m <= 0 or
                not np.isfinite(self.phase_strength_rad) or
                not 0 <= self.fluorescence_escape_yield <= 1 or
                not 0 <= self.cluster_contrast <= 1 or
                self.cluster_count < 2 or self.cluster_count > 64 or
                self.cluster_radius_min_m <= 0 or
                self.cluster_radius_max_m < self.cluster_radius_min_m or
                self.grid_n not in (64, 96) or
                not 8e-3 <= self.field_size_m <= 16e-3 or
                not 1 <= self.z_steps <= 16):
            raise ValueError("invalid structured-gallery settings")


def _assembly_configuration(material: YbLuAGMaterial, thickness_m: float):
    config_path = Path(__file__).resolve().parents[2] / "config" / "ybluag_10at_assembly.json"
    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    if material.yb_at_percent == 12.0:
        configuration["material"] = "12 at.% Yb:LuAG proposal crystal on generic copper"
        configuration["geometry"]["disk_thickness_m"] = thickness_m
        configuration["thermal"]["disk"]["conductivity_W_mK"] = 7.2
        configuration["sources"]["conductivity"] = "Korner et al. 2012 12 at.% crystal parameter"
    return configuration


def _thermal_payload(assembly, scope: str):
    screens = assembly.screens
    return {
        "status": "computed", "material_range_valid": True,
        "input_heat_W": assembly.temperature.input_heat_W,
        "balance_error_W": assembly.temperature.balance_error_W,
        "disk_temperature_min_C": float(np.min(assembly.temperature.disk_temperature_K) - 273.15),
        "disk_temperature_max_C": float(np.max(assembly.temperature.disk_temperature_K) - 273.15),
        "scalar_roundtrip_opd_nm": screens.mean_roundtrip_opd_m * 1e9,
        "front_displacement_nm": screens.front_uz_m * 1e9,
        "rear_displacement_nm": screens.rear_uz_m * 1e9,
        "scope": scope + " Within the currently supported 20–26.85 °C thermo-mechanical parameter range; this is not a crystal survival limit.",
    }


def _thermal_or_design_reference(mesh, heat_polar, grid, configuration, scope):
    try:
        return _thermal_payload(solve_yb_assembly(mesh, heat_polar, grid, configuration),
                                scope)
    except ValueError as exc:
        if "outside 293.15–300 K" not in str(exc):
            raise
    screen = solve_yb_cooler_temperature(mesh, heat_polar, configuration)
    coolant = configuration["thermal"]["coolant_temperature_K"]
    max_rise = float(np.max(screen.disk_temperature_K) - coolant)
    if max_rise <= 0:
        raise ValueError("cooler temperature screen has no positive rise")
    # Scale the *same spatial heat map* into the proposal's 5 K design range.
    # Mechanical deformation is solved only for this in-range design case.
    factor = min(1.0, 5.0 / max_rise)
    design = solve_yb_assembly(mesh, heat_polar * factor, grid, configuration)
    payload = _thermal_payload(design, scope)
    payload.update(
        status="design_reference",
        material_range_valid=False,
        actual_heat_W=float(screen.input_heat_W),
        actual_constant_property_max_C=float(np.max(screen.disk_temperature_K) - 273.15),
        design_heat_scale=factor,
        scope=(scope + " The 5 K surfaces are an in-range design reference. "
               "Thermal optical feedback is not applied to the beam fields."))
    return payload


def _conservative_heat_polar(heat_slices, grid, mesh):
    """Preserve integrated heat after Cartesian-to-polar interpolation."""
    polar = _cartesian_to_polar(heat_slices, grid, mesh)
    target_W = float(np.sum(heat_slices * np.diff(mesh.z_edges_m)[:, None, None]) *
                     grid.dx * grid.dy)
    mapped_W = float(np.sum(polar * mesh.volumes_m3))
    if not np.isfinite(mapped_W) or mapped_W <= 0 or target_W <= 0:
        raise ValueError("thermal heat mapping requires positive finite power")
    return polar * (target_W / mapped_W)


def _pulsed_thermal_timeline(mesh, heat_polar, grid, configuration, duration_s,
                             cooling_mode="fixed", cooling_target_C=40.0,
                             cooling_h_max_W_m2K=100000.0):
    """Finite-capacity startup with a bounded coolant-side flow-control proxy."""
    h_min = configuration["thermal"]["coolant_conductance_W_m2K"]
    bath = configuration["thermal"]["coolant_temperature_K"]
    gain_h_per_K = 3000.0  # Assumed controller slope; no measured flow curve.

    def solver_at(h):
        controlled = deepcopy(configuration)
        controlled["thermal"]["coolant_conductance_W_m2K"] = float(h)
        return yb_cooler_solver(mesh, controlled)

    def command_h(max_disk_K):
        if cooling_mode == "fixed":
            return h_min
        return float(np.clip(h_min + gain_h_per_K *
                             (max_disk_K - 273.15 - cooling_target_C),
                             h_min, cooling_h_max_W_m2K))

    # A steady solution exists because the bath has fixed temperature and
    # positive conductance. Solve the controller's fixed point at steady state.
    if cooling_mode == "feedback":
        low, high = h_min, cooling_h_max_W_m2K
        for _ in range(20):
            mid = 0.5 * (low + high)
            candidate = solver_at(mid).steady(heat_polar)
            if mid < command_h(float(np.max(candidate.disk_temperature_K))):
                low = mid
            else:
                high = mid
        steady_h = 0.5 * (low + high)
    else:
        steady_h = h_min
    steady = solver_at(steady_h).steady(heat_polar)
    steady_disk_max_C = float(np.max(steady.disk_temperature_K) - 273.15)
    tolerance_K = max(0.05, 0.005 * max(steady_disk_max_C - (bath - 273.15), 0))
    solver = solver_at(h_min)
    disk = np.full(mesh.shape, bath)
    plate = np.full(solver.plate.shape, bath)
    x, y = grid.mesh
    inside = x*x + y*y <= (5e-3)**2
    startup = np.geomspace(min(2.5e-5, duration_s / 1000), duration_s, 16)
    end_s = max(30.0, min(300.0, 10 * duration_s))
    continuation = np.geomspace(duration_s * 1.5, end_s, 10) if end_s > duration_s * 1.5 else np.array([])
    times = np.r_[0.0, startup, continuation]
    max_temperature = [bath - 273.15]
    opd_pv = [0.0]
    front_pv = [0.0]
    valid = [True]
    conductance = [h_min]
    balance = []
    latest_valid = None
    latest_valid_time = 0.0
    requested = None
    stabilized_at = None
    for previous, now in zip(times[:-1], times[1:]):
        h = command_h(float(np.max(disk)))
        if h != solver.coolant.conductance_W_m2K:
            solver = solver_at(h)
        step = solver.advance(disk, plate, heat_polar, float(now - previous))
        disk, plate = step.disk_temperature_K, step.plate_temperature_K
        assembly = solve_yb_assembly(mesh, heat_polar, grid, configuration,
                                     temperature=step, allow_extrapolation=True)
        screen = assembly.screens.mean_roundtrip_opd_m[inside]
        face = assembly.screens.front_uz_m[inside]
        max_temperature.append(float(np.max(disk) - 273.15))
        opd_pv.append(float(np.ptp(screen) * 1e9))
        front_pv.append(float(np.ptp(face) * 1e9))
        valid.append(assembly.material_range_valid)
        conductance.append(h)
        balance.append(step.relative_balance_error)
        if np.isclose(now, duration_s, rtol=0, atol=1e-12):
            requested = assembly
        if assembly.material_range_valid:
            latest_valid = assembly
            latest_valid_time = float(now)
        if (now >= duration_s and
                np.max(np.abs(disk - steady.disk_temperature_K)) <= tolerance_K and
                np.max(np.abs(plate - steady.plate_temperature_K)) <= tolerance_K and
                abs(h - steady_h) <= max(1.0, 0.01 * steady_h)):
            stabilized_at = float(now)
            break
    times = times[:len(max_temperature)]
    requested_valid = requested.material_range_valid
    return {
        "time_s": times, "disk_max_C": np.asarray(max_temperature),
        "roundtrip_opd_pv_nm": np.asarray(opd_pv),
        "front_displacement_pv_nm": np.asarray(front_pv),
        "material_range_valid": valid,
        "coolant_conductance_W_m2K": np.asarray(conductance),
        "cooling_mode": cooling_mode, "cooling_target_C": cooling_target_C,
        "cooling_h_max_W_m2K": cooling_h_max_W_m2K,
        "requested_time_s": duration_s,
        "requested_disk_max_C": float(np.max(requested.temperature.disk_temperature_K) - 273.15),
        "requested_material_range_valid": requested_valid,
        "steady_disk_max_C": steady_disk_max_C,
        "steady_coolant_conductance_W_m2K": steady_h,
        "stabilization_tolerance_K": tolerance_K,
        "stabilization_time_s": stabilized_at,
        "stabilized": stabilized_at is not None,
        "energy_balance_relative_max": float(max(balance)),
        "final_roundtrip_opd_m": requested.screens.mean_roundtrip_opd_m,
        "latest_valid_time_s": latest_valid_time,
        "latest_valid_roundtrip_opd_m": (latest_valid.screens.mean_roundtrip_opd_m
                                          if latest_valid is not None else np.zeros(grid.shape)),
        "final_front_displacement_nm": requested.screens.front_uz_m * 1e9,
        "final_rear_displacement_nm": requested.screens.rear_uz_m * 1e9,
        "scope": ("Finite startup from uniform coolant temperature with cycle-averaged periodic pulse heat "
                  "applied from t=0; the first population-recovery interval is approximated. "
                  "Disk, contact, and copper heat capacities are integrated by backward Euler; each recorded "
                  "temperature drives a bonded elastic solve and round-trip OPD. "
                  "The optional flow controller changes only the water-side conductance; its slope and "
                  "instantaneous response are assumptions. The fixed-temperature coolant bath is idealized. "
                  "Results outside the currently supported 20–26.85 °C thermo-mechanical parameter "
                  "range are unvalidated extrapolations, not crystal failure claims or device predictions."),
    }


def _modal_cw_background(material, settings, grid, density_scale, pump):
    """Fixed Gaussian cavity mode with two-manifold CW gain saturation."""
    cavity = ThinDiskResonator(
        disk_diameter_m=0.010, disk_thickness_m=settings.thickness_m,
        air_gap_m=0.200, output_mirror_radius_m=0.500,
        output_transmission=0.015, disk_hr_reflectivity=0.9995,
        other_roundtrip_loss=0.005,
        wavelength_m=material.signal_wavelength_nm * 1e-9,
        host_index=material.refractive_index(),
        host_group_index=material.refractive_index())
    if not cavity.stable:
        raise ValueError("assumed Yb cavity is unstable")
    x, y = grid.mesh
    mode = np.exp(-2 * (x*x + y*y) / cavity.waist_m**2)
    mode /= float(np.sum(mode) * grid.dx * grid.dy)
    dz = settings.thickness_m / settings.z_steps
    fluorescence_energy = fluorescence_spectrum(material).mean_photon_energy_J

    def state(intracavity_W, keep=False):
        p = pump.copy()
        integrated_gain = 0.0
        betas = []
        heat_slices = []
        for scale in density_scale:
            signal_intensity = 2 * intracavity_W * mode
            beta0 = material.excited_fraction_cw(p, signal_intensity)
            alpha0, _ = material.coefficients_m1(beta0)
            p_mid = p * np.exp(-0.5 * alpha0 * scale * dz)
            beta = material.excited_fraction_cw(p_mid, signal_intensity)
            alpha, gain = material.coefficients_m1(beta)
            alpha *= scale
            gain *= scale
            p_next = p * np.exp(-alpha * dz)
            integrated_gain += float(np.sum(gain * mode) * grid.dx * grid.dy * dz)
            if keep:
                fluorescence = (material.number_density_m3 * scale * beta /
                                material.lifetime_s * settings.fluorescence_escape_yield *
                                fluorescence_energy * dz)
                heat_slices.append(((p - p_next) - gain * signal_intensity * dz -
                                    fluorescence) / dz)
                betas.append(beta)
            p = p_next
        return 2 * integrated_gain + math.log(cavity.passive_power_retention), betas, heat_slices

    small_gain = state(0)[0]
    intracavity = 0.0
    if small_gain > 0:
        high = 1.0
        while state(high)[0] > 0:
            high *= 2
            if high > 1e8:
                raise RuntimeError("could not bracket Yb modal CW intensity")
        low = 0.0
        for _ in range(35):
            middle = 0.5 * (low + high)
            if state(middle)[0] > 0:
                low = middle
            else:
                high = middle
        intracavity = 0.5 * (low + high)
    residual, betas, heat = state(intracavity, keep=True)
    return {
        "beta_by_slice": np.stack(betas), "heat_W_m3_by_slice": np.stack(heat),
        "cavity_waist_m": cavity.waist_m,
        "intracavity_one_direction_W": intracavity,
        "output_coupler_power_W": intracavity * cavity.output_transmission,
        "small_signal_roundtrip_log_margin": small_gain,
        "roundtrip_log_residual": residual,
        "scope": "Fixed Gaussian CW mode, equal counterpropagating intensities, 0.2 m plane-concave cavity, 1.5% output coupler, 0.05% disk HR loss and 0.5% other round-trip loss. No transverse eigenmode or thermal feedback."}


def simulate_structured_gallery(material: YbLuAGMaterial,
                                settings: YbGallerySettings, *,
                                selected_beam: str = "Gaussian TEM00",
                                compute_thermal: bool = True):
    """Calculate one selected field with spatial Yb concentration and local CW rates.

    In weak mode the pump-only population is reused for each field. Saturated
    mode recomputes the two-manifold population from each signal intensity.
    The signal is transported coherently and diffracted between slices.
    """
    if selected_beam not in BEAM_NAMES:
        raise ValueError("unknown selected beam")
    grid = Grid2D.square(settings.grid_n, settings.field_size_m)
    common = GallerySettings(mean_ho_density_m3=material.number_density_m3,
                             density_seed=settings.density_seed,
                             cluster_count=settings.cluster_count,
                             cluster_contrast=settings.cluster_contrast,
                             cluster_radius_min_m=settings.cluster_radius_min_m,
                             cluster_radius_max_m=settings.cluster_radius_max_m,
                             waist_m=settings.waist_m,
                             input_power_W=settings.input_power_W,
                             phase_mask_name=settings.phase_mask_name,
                             phase_strength_rad=settings.phase_strength_rad)
    z_edges = np.linspace(0, settings.thickness_m, settings.z_steps + 1)
    density = nonuniform_density(grid, z_edges, 5e-3, common).values_m3
    density_scale = density / material.number_density_m3
    aberration_phase = phase_pattern(grid, common)
    source, target_phase = gaussian_seed_and_target_mask(
        grid, settings.waist_m, settings.input_power_W, selected_beam,
        material.signal_wavelength_nm * 1e-9, settings.slm_to_disk_distance_m)
    phase = np.mod(target_phase + aberration_phase, 2 * np.pi)
    x, y = grid.mesh
    pump = np.exp(-2 * (x*x + y*y) / settings.pump_radius_m**2)
    pump *= settings.pump_power_W / (float(pump.sum()) * grid.dx * grid.dy)
    dz = settings.thickness_m / settings.z_steps
    wavelength = material.signal_wavelength_nm * 1e-9
    n = material.refractive_index()
    fluorescence_energy = fluorescence_spectrum(material).mean_photon_energy_J
    modal = (_modal_cw_background(material, settings, grid, density_scale, pump)
             if settings.solver_mode == "modal_cw" else None)
    outcomes = {}
    reference_heat_slices = None
    for name in (selected_beam,):
        field_in = source
        field = angular_spectrum_propagate(
            source * np.exp(1j * phase), grid, wavelength,
            settings.slm_to_disk_distance_m)
        disk_field_in = field.copy()
        pump_step = pump.copy()
        heat = np.zeros(grid.shape)
        heat_slices = []
        beta_sum = np.zeros(grid.shape)
        for iz, scale in enumerate(density_scale):
            field = angular_spectrum_propagate(field, grid, wavelength, dz / 2,
                                               refractive_index=n)
            signal = abs(field)**2
            beta = (modal["beta_by_slice"][iz] if modal is not None else
                    material.excited_fraction_cw(
                        pump_step, signal if settings.solver_mode == "saturated_cw" else 0))
            alpha, gain = material.coefficients_m1(beta)
            alpha *= scale
            gain *= scale
            pump_next = pump_step * np.exp(-alpha * dz)
            field_next = field * np.exp(0.5 * gain * dz)
            if settings.solver_mode == "saturated_cw":
                beta = material.excited_fraction_cw(
                    0.5 * (pump_step + pump_next),
                    0.5 * (signal + abs(field_next)**2))
                alpha, gain = material.coefficients_m1(beta)
                alpha *= scale
                gain *= scale
                pump_next = pump_step * np.exp(-alpha * dz)
                field_next = field * np.exp(0.5 * gain * dz)
            emitted = np.zeros_like(pump_step)
            if settings.fluorescence_escape_yield:
                emitted = (material.number_density_m3 * scale * beta /
                           material.lifetime_s * settings.fluorescence_escape_yield *
                           fluorescence_energy * dz)
            # The weak probe does not deplete inversion, so its gain cannot be
            # subtracted from the pump-only heat balance.
            heat_sheet = ((pump_step - pump_next) - emitted if settings.solver_mode == "weak_probe"
                          else (pump_step - pump_next) - (abs(field_next)**2 - signal) - emitted)
            heat += heat_sheet
            heat_slices.append(heat_sheet / dz)
            beta_sum += beta
            pump_step = pump_next
            field = angular_spectrum_propagate(field_next, grid, wavelength, dz / 2,
                                               refractive_index=n)
        disk_out = optical_power(field, grid)
        if settings.post_disk_distance_m:
            field = angular_spectrum_propagate(field, grid, wavelength,
                                               settings.post_disk_distance_m)
        outcomes[name] = {
            "input_intensity": abs(field_in)**2,
            "disk_input_intensity": abs(disk_field_in)**2,
            "output_intensity": abs(field)**2,
            "input_phase": np.angle(field_in),
            "disk_input_phase": np.angle(disk_field_in),
            "output_phase": np.angle(field),
            "input_profile": abs(field_in[grid.ny // 2])**2,
            "output_profile": abs(field[grid.ny // 2])**2,
            "input_power_W": optical_power(field_in, grid),
            "disk_output_power_W": disk_out,
            "output_power_W": optical_power(field, grid),
            "mean_excited_fraction": float(np.mean(beta_sum / settings.z_steps)),
            "pump_absorbed_W": float(np.sum(pump - pump_step) * grid.dx * grid.dy),
            "net_heat_W_upper_or_assumed": float(
                np.sum(modal["heat_W_m3_by_slice"]) * dz * grid.dx * grid.dy
                if modal is not None else np.sum(heat) * grid.dx * grid.dy),
        }
        reference_heat_slices = (modal["heat_W_m3_by_slice"] if modal is not None
                                 else np.stack(heat_slices))
    thermal = {"status": "not_requested", "reason": "Thermal calculation disabled."}
    if compute_thermal:
        nominal_thickness = 100e-6 if material.yb_at_percent == 12 else 150e-6
        if not np.isclose(settings.thickness_m, nominal_thickness, atol=1e-12):
            thermal = {"status": "out_of_scope", "reason": f"The assembly configuration is for a {nominal_thickness * 1e6:g} µm disk."}
        else:
            configuration = _assembly_configuration(material, settings.thickness_m)
            mesh = DiskThermalMesh.disk(nr=8, nz=settings.z_steps, nphi=12,
                                        radius_m=5e-3, thickness_m=settings.thickness_m)
            heat_polar = _conservative_heat_polar(reference_heat_slices, grid, mesh)
            try:
                thermal = _thermal_or_design_reference(
                    mesh, heat_polar, grid, configuration,
                    ("Fixed Gaussian cavity heat" if modal is not None else
                     "Pump-only weak-probe heat" if settings.solver_mode == "weak_probe" else
                     f"{selected_beam} saturated-CW heat") +
                    " on generic C10100 copper; scalar optical path and surface deformation; photoelasticity omitted.")
            except ValueError as exc:
                thermal = {"status": "out_of_scope", "reason": str(exc)}
    if modal is not None:
        modal = {key: value for key, value in modal.items()
                 if key not in ("beta_by_slice", "heat_W_m3_by_slice")}
    return {"grid": grid, "phase_mask": phase,
            "target_phase_mask": np.mod(target_phase, 2 * np.pi),
            "aberration_phase_mask": np.mod(aberration_phase, 2 * np.pi),
            "yb_density_m3": density,
            "pump_intensity_W_m2": pump, "outcomes": outcomes,
            "thermal": thermal,
            "resonator": modal,
            "fluorescence_escape_yield_assumed": settings.fluorescence_escape_yield,
            "scope": ("One selected CW single-pass Yb:LuAG calculation. The clustered Yb map is "
                      "synthetic, not a measured crystal. The weak probe does not deplete "
                      "inversion; saturated CW includes local signal depletion. Modal CW "
                      "uses one fixed Gaussian cavity background and probes each shape "
                      "without depletion. Weak-probe heat is a pump-only estimate; saturated CW heat includes signal extraction. "
                      "Modal CW uses its fixed Gaussian cavity heat. A Gaussian source receives a phase-only "
                      "target mask plus optional correction before free-space propagation to the disk. "
                      "All modes screen a "
                      "copper-cooled assembly within its material range. Free-space diffraction "
                      "follows the disk. Fluorescence escape yield is assumed by the user.")}


def simulate_pulsed_seed(material: YbLuAGMaterial, settings: YbGallerySettings,
                         selected_beam: str, seed_energy_J: float,
                         seed_fwhm_s: float, repetition_rate_Hz: float,
                         signal_traversals: int, pump_passes: int = 10,
                         *, compute_thermal: bool = True,
                         operation_duration_s: float = 0.0,
                         cooling_mode: str = "fixed", cooling_target_C: float = 40.0,
                         cooling_h_max_W_m2K: float = 100000.0):
    """Periodic pulsed seed with fixed pump-only CW profile and ideal relays.

    The population is iterated from pulse to pulse. During the short seed the
    pump is neglected, while between seeds the local pump rates restore the
    population. The fixed pump profile ignores pulse-induced pump saturation.
    """
    if selected_beam not in BEAM_NAMES:
        raise ValueError("unknown seed beam")
    if (not np.isfinite(seed_energy_J) or seed_energy_J <= 0 or
            not np.isfinite(seed_fwhm_s) or seed_fwhm_s <= 0 or
            not np.isfinite(repetition_rate_Hz) or repetition_rate_Hz <= 0 or
            isinstance(signal_traversals, bool) or not 1 <= signal_traversals <= 10 or
            isinstance(pump_passes, bool) or not isinstance(pump_passes, int) or
            not 1 <= pump_passes <= 48 or
            not np.isfinite(operation_duration_s) or
            not 0 <= operation_duration_s <= 120 or
            cooling_mode not in ("fixed", "feedback") or
            not np.isfinite(cooling_target_C) or not 20 <= cooling_target_C <= 250 or
            not np.isfinite(cooling_h_max_W_m2K) or
            not 10000 <= cooling_h_max_W_m2K <= 200000):
        raise ValueError("invalid pulsed seed settings")
    grid = Grid2D.square(settings.grid_n, settings.field_size_m)
    common = GallerySettings(mean_ho_density_m3=material.number_density_m3,
                             density_seed=settings.density_seed,
                             cluster_count=settings.cluster_count,
                             cluster_contrast=settings.cluster_contrast,
                             cluster_radius_min_m=settings.cluster_radius_min_m,
                             cluster_radius_max_m=settings.cluster_radius_max_m,
                             waist_m=settings.waist_m, input_power_W=1.0,
                             phase_mask_name=settings.phase_mask_name,
                             phase_strength_rad=settings.phase_strength_rad)
    density = nonuniform_density(grid, np.linspace(0, settings.thickness_m,
                         settings.z_steps + 1), 5e-3, common).values_m3
    scale = density / material.number_density_m3
    source, target_phase = gaussian_seed_and_target_mask(
        grid, settings.waist_m, 1.0, selected_beam,
        material.signal_wavelength_nm * 1e-9, settings.slm_to_disk_distance_m)
    aberration_phase = phase_pattern(grid, common)
    phase = np.mod(target_phase + aberration_phase, 2 * np.pi)
    seed = angular_spectrum_propagate(
        source * np.exp(1j * phase), grid,
        material.signal_wavelength_nm * 1e-9, settings.slm_to_disk_distance_m)
    x, y = grid.mesh
    pump = np.exp(-2 * (x*x + y*y) / settings.pump_radius_m**2)
    pump *= settings.pump_power_W / (float(pump.sum()) * grid.dx * grid.dy)
    dz = settings.thickness_m / settings.z_steps
    pump_state = steady_multipass_pump(
        material, pump, scale, settings.thickness_m, pump_passes)
    beta_steady = pump_state.excited_fraction_by_slice
    up, down = material.rates_s1(pump_state.total_midpoint_intensity_W_m2_by_slice, 0)
    recovery_rate = up + down + 1 / material.lifetime_s
    time = np.linspace(-3 * seed_fwhm_s, 3 * seed_fwhm_s, 41)
    pulse_shape = np.exp(-4 * np.log(2) * (time / seed_fwhm_s)**2)
    pulse_shape /= np.trapezoid(pulse_shape, time)
    input_fluence = seed_energy_J * abs(source)**2
    disk_input_fluence = seed_energy_J * abs(seed)**2
    initial_signal = pulse_shape[:, None, None] * disk_input_fluence[None]
    dark_time = 1 / repetition_rate_Hz - (time[-1] - time[0])
    if dark_time <= 0:
        raise ValueError("pulse window exceeds repetition period")
    beta_before = beta_steady.copy()
    cycles = 0
    residual = math.inf
    for cycles in range(1, 101):
        signal = initial_signal
        beta = beta_before.copy()
        signal_gain_fluence = np.zeros_like(beta)
        for _ in range(signal_traversals):
            pulse = propagate_pulse(material, time, np.zeros_like(signal), signal,
                                    settings.thickness_m, settings.z_steps,
                                    initial_excited_fraction=beta,
                                    density_scale_by_slice=scale)
            signal = pulse.signal_out_W_m2
            signal_gain_fluence += pulse.signal_fluence_change_J_m2_by_slice
            beta = pulse.final_excited_fraction_by_slice
        following = beta_steady + (beta - beta_steady) * np.exp(-recovery_rate * dark_time)
        residual = float(np.max(abs(following - beta_before)))
        beta_before = following
        if residual <= 1e-6:
            break
    else:
        raise RuntimeError("pulsed Yb population did not converge within 100 cycles")
    beta_dark_integral = (beta_steady * dark_time +
                          (beta - beta_steady) *
                          (-np.expm1(-recovery_rate * dark_time)) / recovery_rate)
    beta_average = beta_dark_integral * repetition_rate_Hz
    _, pump_absorbed, _ = transport_multipass_pump(
        material, pump, scale, settings.thickness_m, pump_passes, beta_average)
    fluorescence = (density * beta_average / material.lifetime_s *
                    settings.fluorescence_escape_yield *
                    fluorescence_spectrum(material).mean_photon_energy_J * dz)
    signal_gain = signal_gain_fluence * repetition_rate_Hz
    heat_slices = (pump_absorbed - signal_gain - fluorescence) / dz
    pixel_area = grid.dx * grid.dy
    heat_W = float(np.sum(heat_slices) * dz * pixel_area)
    absorbed_W = float(np.sum(pump_absorbed) * pixel_area)
    signal_gain_W = float(np.sum(signal_gain) * pixel_area)
    fluorescence_W = float(np.sum(fluorescence) * pixel_area)
    nominal_thickness = 100e-6 if material.yb_at_percent == 12 else 150e-6
    thermal = {"status": "out_of_scope", "reason": f"The assembly configuration is for a {nominal_thickness * 1e6:g} µm disk."}
    if not compute_thermal:
        thermal = {"status": "not_requested", "reason": "Thermal calculation was shared from the Gaussian reference case."}
    timeline = None
    if compute_thermal and np.isclose(settings.thickness_m, nominal_thickness, atol=1e-12):
        configuration = _assembly_configuration(material, settings.thickness_m)
        mesh = DiskThermalMesh.disk(nr=8, nz=settings.z_steps, nphi=12,
                                    radius_m=5e-3, thickness_m=settings.thickness_m)
        heat_polar = _conservative_heat_polar(heat_slices, grid, mesh)
        try:
            thermal = _thermal_or_design_reference(
                mesh, heat_polar, grid, configuration,
                "Fixed pump-profile periodic heat on generic C10100 copper; scalar optical path and surface deformation; photoelasticity omitted.")
        except ValueError as exc:
            thermal = {"status": "out_of_scope", "reason": str(exc)}
        if operation_duration_s:
            timeline = _pulsed_thermal_timeline(mesh, heat_polar, grid,
                                                 configuration, operation_duration_s,
                                                 cooling_mode, cooling_target_C,
                                                 cooling_h_max_W_m2K)
    fluence_out = np.trapezoid(signal, time, axis=0)
    disk_output_J = float(np.sum(fluence_out) * grid.dx * grid.dy)
    gain_amplitude = np.sqrt(np.maximum(fluence_out, 0) / np.maximum(disk_input_fluence, 1e-30))
    field_out = seed * gain_amplitude
    thermal_feedback_applied = bool(timeline is not None and
                                    timeline["requested_material_range_valid"])
    if thermal_feedback_applied:
        # An ideal image relay returns the same transverse coordinate to the
        # same physical disk. Two thickness traversals make one active-mirror
        # round trip; this phase is applied at each encounter in that model.
        phase_screen = (np.pi * signal_traversals /
                        (material.signal_wavelength_nm * 1e-9) *
                        timeline["final_roundtrip_opd_m"])
        field_out *= np.exp(1j * phase_screen)
    if settings.post_disk_distance_m:
        field_out = angular_spectrum_propagate(field_out, grid,
                                               material.signal_wavelength_nm * 1e-9,
                                               settings.post_disk_distance_m)
    return {
        "grid": grid, "phase_mask": phase,
        "target_phase_mask": np.mod(target_phase, 2 * np.pi),
        "aberration_phase_mask": np.mod(aberration_phase, 2 * np.pi),
        "yb_density_m3": density,
        "selected_beam": selected_beam,
        "input_fluence_J_m2": input_fluence,
        "disk_input_fluence_J_m2": disk_input_fluence,
        "output_fluence_J_m2": abs(field_out)**2 * seed_energy_J,
        "input_phase": np.angle(source),
        "disk_input_phase": np.angle(seed),
        "output_phase": np.angle(field_out),
        "time_ps": (time * 1e12),
        "input_power_trace_W": np.sum(initial_signal, axis=(1, 2)) * grid.dx * grid.dy,
        "output_power_trace_W": np.sum(signal, axis=(1, 2)) * grid.dx * grid.dy,
        "input_energy_J": seed_energy_J, "disk_output_energy_J": disk_output_J,
        "output_energy_J": optical_power(field_out, grid) * seed_energy_J,
        "cycles": cycles, "residual": residual,
        "pump_passes": pump_passes,
        "pump_steady_iterations": pump_state.iterations,
        "cycle_average_heat_W_upper_or_assumed": heat_W,
        "cycle_average_pump_absorbed_W": absorbed_W,
        "cycle_average_signal_gain_W": signal_gain_W,
        "cycle_average_escaping_fluorescence_W": fluorescence_W,
        "fluorescence_escape_yield_assumed": settings.fluorescence_escape_yield,
        "thermal": thermal,
        "thermal_timeline": timeline,
        "thermal_feedback_applied": thermal_feedback_applied,
        "thermal_feedback_time_s": (operation_duration_s if thermal_feedback_applied else None),
        "mean_excited_fraction_before_pulse": float(np.mean(beta_before)),
        "scope": (f"Gaussian TEM00 source shaped by a phase-only target mask plus optional "
                  "added phase, then propagated to the disk. LG, HG, Bessel and flat-top names "
                  "describe approximate targets, not guaranteed pure modes. The flat-top mask "
                  "uses a 48-step scalar alternating-projection design. "
                  f"Periodic two-manifold Yb:LuAG population with {pump_passes} alternating "
                  "CW pump traversals, short pulse gain depletion, and ideal image relays "
                  "between signal traversals. Retarded-time intensity transport omits "
                  "GVD, Kerr phase, walkoff, coherent diffraction within each pulse pass, "
                  "and gain feedback from thermal beam reshaping. The recorded thermal OPD "
                  "is applied per ideal-relayed disk encounter before output-plane diffraction "
                  "only when the requested operating time remains inside the 20–26.85 C "
                  "thermo-mechanical parameter range, which is not a crystal survival limit. "
                  "Cycle-averaged heat uses the fixed pump profile "
                  "and an assumed fluorescence escape yield; the generic copper assembly "
                  "runs only inside its material temperature range. The output spatial "
                  "phase retains the shaped disk-incident phase "
                  "and includes free-space propagation; high-extraction phase accuracy "
                  "requires a coupled space-time field solver.")
    }
