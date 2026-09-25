"""Separate Yb:LuAG structured-field calculations for the local browser app.

The beam and ideal phase-mask generators are shared with the Ho gallery; all
gain, population, and heat quantities here use the Yb two-manifold model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import map_coordinates

from hoyag.propagation import Grid2D, angular_spectrum_propagate, optical_power
from hoyag.resonator import ThinDiskResonator
from hoyag.structured_beam_gallery import (BEAM_NAMES, PHASE_MASKS, GallerySettings,
                                            _cartesian_to_polar,
                                            nonuniform_density, phase_pattern)
from hoyag.thermal import DiskThermalMesh
from hoyag.cooling_plate import sample_temperature

from .fluorescence import fluorescence_spectrum
from .model import H, C, YbLuAGMaterial, trapezoid
from .assembly import (solve_yb_assembly, solve_yb_cooler_temperature,
                       yb_cooler_solver)
from .pulsed import propagate_pulse
from .multipass_pump import recover_pumped_population, steady_multipass_pump, transport_multipass_pump
from .beam_shaping import gaussian_seed_and_target_mask
from .regenerative import RegenerativeCavity, amplify_regenerative
from .field_metrics import coherent_overlap, intensity_overlap as field_intensity_overlap
from .sensors import ProbeArray, default_five_probes


@dataclass(frozen=True)
class YbGallerySettings:
    pump_power_W: float = 5.0
    pump_radius_m: float = 0.6e-3
    thickness_m: float = 150e-6
    disk_radius_m: float = 5e-3
    assembly_property_model: str = "reference_10at"
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
    thermal_nr: int = 8
    thermal_nphi: int = 12
    thermal_nz: int = 4
    ideal_relay_power_retention: float = 1.0

    def __post_init__(self):
        if self.phase_mask_name not in PHASE_MASKS:
            raise ValueError("unknown phase mask")
        if self.solver_mode not in ("weak_probe", "saturated_cw", "modal_cw"):
            raise ValueError("unknown Yb structured solver mode")
        for name in ("pump_power_W", "pump_radius_m", "thickness_m", "disk_radius_m", "input_power_W",
                     "waist_m"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (not np.isfinite(self.post_disk_distance_m) or self.post_disk_distance_m < 0 or
                self.assembly_property_model not in ("reference_10at", "proposal_12at", "yag_rt_proxy") or
                not np.isfinite(self.slm_to_disk_distance_m) or
                self.slm_to_disk_distance_m <= 0 or
                not np.isfinite(self.phase_strength_rad) or
                not 0 <= self.fluorescence_escape_yield <= 1 or
                not 0 <= self.cluster_contrast <= 1 or
                self.cluster_count < 2 or self.cluster_count > 64 or
                self.cluster_radius_min_m <= 0 or
                self.cluster_radius_max_m < self.cluster_radius_min_m or
                isinstance(self.grid_n, bool) or not isinstance(self.grid_n, int) or
                not 32 <= self.grid_n <= 768 or
                not 8e-3 <= self.field_size_m <= 24e-3 or
                not 1 <= self.z_steps <= 16 or
                not 4 <= self.thermal_nr <= 48 or
                not 4 <= self.thermal_nphi <= 96 or
                not 1 <= self.thermal_nz <= 24 or
                not np.isfinite(self.ideal_relay_power_retention) or
                not 0 < self.ideal_relay_power_retention <= 1):
            raise ValueError("invalid structured-gallery settings")


def _assembly_configuration(material: YbLuAGMaterial, settings: YbGallerySettings):
    config_path = Path(__file__).resolve().parents[2] / "config" / "ybluag_10at_assembly.json"
    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    if material.name == "Yb:YAG":
        from ybyag.assembly import configuration as yag_configuration
        return yag_configuration(material, settings, configuration)
    if settings.assembly_property_model == "yag_rt_proxy":
        raise ValueError("YAG assembly cannot be used for Yb:LuAG")
    configuration["geometry"]["disk_radius_m"] = settings.disk_radius_m
    configuration["geometry"]["disk_thickness_m"] = settings.thickness_m
    configuration["material"] = (f"{material.yb_at_percent:g} at.% Yb:LuAG with "
                                  f"{settings.assembly_property_model} property assumption")
    configuration["sources"]["selected_property_model"] = settings.assembly_property_model
    if settings.assembly_property_model == "proposal_12at":
        configuration["thermal"]["disk"]["conductivity_W_mK"] = 7.2
        configuration["sources"]["conductivity"] = ("Korner et al. 2012 12 at.% crystal parameter; "
                                                  "explicit proxy at other concentrations")
    return configuration


def _optical_temperature_field(mesh, disk_temperature_K, grid, optical_depth_cells,
                               reference_temperature_K):
    """Register the one physical thermal disk on all optical depth cells."""
    x, y = grid.mesh
    active = np.hypot(x, y) <= mesh.r_edges_m[-1]
    xy = np.column_stack((x[active], y[active]))
    field = np.full((optical_depth_cells, *grid.shape), reference_temperature_K)
    for iz in range(optical_depth_cells):
        z = (iz + .5) * mesh.z_edges_m[-1] / optical_depth_cells
        points = np.column_stack((xy, np.full(len(xy), z)))
        field[iz, active] = sample_temperature(mesh, disk_temperature_K, points)
    return field


def _coupled_regenerative_steady(material, settings, grid, seed, seed_energy_J,
                                  pump, density_scale, pump_passes,
                                  repetition_rate_Hz, cavity_settings,
                                  fluorescence_energy_J, cooling_mode,
                                  cooling_target_C, cooling_h_max_W_m2K,
                                  temperature_probes, optical_solver=None):
    """Iterate periodic optical state, heat, disk/plate, and encounter optics.

    The thermo-optic coefficient and elastic constants come from the selected
    uncalibrated assembly property assumption. Out-of-range temperatures stop
    this mode; no heat normalization or silent extrapolation is applied.
    """
    configuration = _assembly_configuration(material, settings)
    mesh = DiskThermalMesh.disk(nr=settings.thermal_nr, nz=settings.thermal_nz,
                                nphi=settings.thermal_nphi,
                                radius_m=settings.disk_radius_m,
                                thickness_m=settings.thickness_m)
    temperature = np.full(mesh.shape, configuration["thermal"]["coolant_temperature_K"])
    plate_temperature = None
    screen_opd = np.zeros(grid.shape)
    previous_heat = None
    previous_output = None
    previous_field = None
    cold_reference_field = None
    history = []
    relaxation = .6
    nominal_probes = None
    if cooling_mode == "sensor_feedback":
        initial_cooler = yb_cooler_solver(mesh, configuration)
        nominal_probes = ProbeArray(
            default_five_probes(mesh.z_edges_m[-1], mesh.r_edges_m[-1])
            if temperature_probes is None else temperature_probes,
            mesh, initial_cooler.plate,
            initial_temperature_K=configuration["thermal"]["coolant_temperature_K"])

    def controlled_steady(heat_polar):
        h_min = configuration["thermal"]["coolant_conductance_W_m2K"]
        def at(h):
            controlled = deepcopy(configuration)
            controlled["thermal"]["coolant_conductance_W_m2K"] = float(h)
            return solve_yb_cooler_temperature(mesh, heat_polar, controlled)
        if cooling_mode == "fixed":
            return at(h_min), h_min
        low, high = h_min, cooling_h_max_W_m2K
        for _ in range(20):
            midpoint = .5*(low+high)
            trial = at(midpoint)
            if cooling_mode == "feedback":
                observation = float(np.max(trial.disk_temperature_K))
            else:
                readings = [nominal_probes._local_average(
                    probe, trial.disk_temperature_K,
                    trial.plate_temperature_K) + probe.bias_K
                    for probe in nominal_probes.probes
                    if probe.region == "disk" and not probe.missing]
                observation = max(readings) if readings else configuration["thermal"]["coolant_temperature_K"]
            requested_h = float(np.clip(
                h_min + 3000*(observation-273.15-cooling_target_C),
                h_min, cooling_h_max_W_m2K))
            if midpoint < requested_h:
                low = midpoint
            else:
                high = midpoint
        h = .5*(low+high)
        return at(h), h

    for iteration in range(1, 13):
        local_temperature = _optical_temperature_field(
            mesh, temperature, grid, density_scale.shape[0], material.temperature_K)
        regen = (optical_solver(local_temperature, screen_opd)
                 if optical_solver is not None else amplify_regenerative(
                     material, grid, seed, seed_energy_J, pump, density_scale,
                     settings.thickness_m, pump_passes, repetition_rate_Hz,
                     cavity_settings, fluorescence_energy_J,
                     settings.fluorescence_escape_yield,
                     temperature_K_by_slice=local_temperature,
                     encounter_opd_m=screen_opd))
        if cold_reference_field is None:
            cold_reference_field = regen["output_field"].copy() / math.sqrt(seed_energy_J)
        heat = regen["heat_W_m3_by_slice"]
        heat_polar = _conservative_heat_polar(heat, grid, mesh)
        steady, controlled_h = controlled_steady(heat_polar)
        if (np.min(steady.disk_temperature_K) < 293.15 or
                np.max(steady.disk_temperature_K) > 300.0):
            raise ValueError(
                "coupled steady state unavailable: disk temperature exceeds "
                "293.15–300 K assembly-property range; select an explicit "
                "unvalidated material extrapolation model to extend it")
        new_temperature = (temperature + relaxation *
                           (steady.disk_temperature_K - temperature))
        new_plate = (steady.plate_temperature_K if plate_temperature is None else
                     plate_temperature + relaxation *
                     (steady.plate_temperature_K - plate_temperature))
        relaxed = replace(steady, disk_temperature_K=new_temperature,
                          plate_temperature_K=new_plate)
        assembly = solve_yb_assembly(mesh, heat_polar, grid, configuration,
                                     temperature=relaxed)
        new_opd = (assembly.screens.thermal_single_pass_opd_m +
                   .5 * assembly.screens.geometry_roundtrip_opd_m)
        temperature_error = float(np.max(np.abs(new_temperature-temperature)))
        heat_error = (float(np.max(np.abs(heat-previous_heat)) /
                            max(float(np.max(np.abs(heat))), 1.0))
                      if previous_heat is not None else float("inf"))
        output_error = (abs(regen["output_energy_J"]-previous_output) /
                        max(regen["output_energy_J"], 1e-30)
                        if previous_output is not None else float("inf"))
        wavefront_error = float(np.max(np.abs(new_opd-screen_opd)) *
                                2*np.pi/(material.signal_wavelength_nm*1e-9))
        field_error = (1.0 - coherent_overlap(previous_field, regen["output_field"])
                       if previous_field is not None else float("inf"))
        history.append({"iteration": iteration,
                        "coolant_conductance_W_m2K": controlled_h,
                        "temperature_max_delta_K": temperature_error,
                        "heat_relative_delta": (heat_error if np.isfinite(heat_error) else None),
                        "output_energy_relative_delta": (output_error if np.isfinite(output_error) else None),
                        "encounter_wavefront_max_delta_rad": wavefront_error,
                        "coherent_field_overlap_error": (field_error if np.isfinite(field_error) else None)})
        temperature, plate_temperature, screen_opd = new_temperature, new_plate, new_opd
        previous_heat = heat.copy()
        previous_output = regen["output_energy_J"]
        previous_field = regen["output_field"].copy()
        if (temperature_error <= .01 and heat_error <= 1e-3 and
                output_error <= 1e-3 and wavefront_error <= 1e-3 and
                field_error <= 1e-6):
            # Publish optics evaluated at the accepted thermal/OPD state,
            # rather than the previous outer iterate.
            accepted_temperature = _optical_temperature_field(
                mesh, new_temperature, grid, density_scale.shape[0],
                material.temperature_K)
            accepted = (optical_solver(accepted_temperature, new_opd)
                        if optical_solver is not None else amplify_regenerative(
                            material, grid, seed, seed_energy_J, pump,
                            density_scale, settings.thickness_m, pump_passes,
                            repetition_rate_Hz, cavity_settings,
                            fluorescence_energy_J,
                            settings.fluorescence_escape_yield,
                            temperature_K_by_slice=accepted_temperature,
                            encounter_opd_m=new_opd))
            accepted_heat = accepted["heat_W_m3_by_slice"]
            accepted_heat_error = float(np.max(np.abs(accepted_heat-heat)) /
                                        max(float(np.max(np.abs(accepted_heat))), 1.0))
            history[-1]["accepted_heat_relative_delta"] = accepted_heat_error
            if accepted_heat_error <= 1e-3:
                return accepted, assembly, mesh, history, cold_reference_field
    raise RuntimeError("Yb coupled regenerative steady state did not converge in 12 iterations")


def _thermal_payload(assembly, scope: str):
    screens = assembly.screens
    return {
        "status": "computed", "material_range_valid": True,
        "validity": "extrapolated_unvalidated",
        "validity_reason": "Temperature is within the available material range, but the generic assembly and proxy optical coefficients are not experimentally calibrated.",
        "input_heat_W": assembly.temperature.input_heat_W,
        "balance_error_W": assembly.temperature.balance_error_W,
        "disk_temperature_min_C": float(np.min(assembly.temperature.disk_temperature_K) - 273.15),
        "disk_temperature_max_C": float(np.max(assembly.temperature.disk_temperature_K) - 273.15),
        "disk_temperature_K": assembly.temperature.disk_temperature_K,
        "plate_temperature_K": assembly.temperature.plate_temperature_K,
        "scalar_roundtrip_opd_nm": screens.mean_roundtrip_opd_m * 1e9,
        "front_displacement_nm": screens.front_uz_m * 1e9,
        "rear_displacement_nm": screens.rear_uz_m * 1e9,
        "photoelastic_retardance_rad": None,
        "photoelastic_reason": "The scalar amplifier does not apply photoelasticity; a validated material tensor, crystal orientation and vector optical coupling are required.",
        "scope": scope + " Within the currently supported 20–26.85 °C thermo-mechanical parameter range; this is not a crystal survival limit.",
    }


def _empty_thermal(status, reason):
    return {"status": status, "validity": "not_calculated", "reason": reason,
            "scalar_roundtrip_opd_nm": None,
            "front_displacement_nm": None,
            "rear_displacement_nm": None,
            "photoelastic_retardance_rad": None,
            "photoelastic_reason": "Photoelasticity is not applied in the scalar amplifier."}


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
    design_maps = {key: payload[key] for key in (
        "scalar_roundtrip_opd_nm", "front_displacement_nm",
        "rear_displacement_nm")}
    payload.update(
        status="design_reference",
        material_range_valid=False,
        validity="extrapolated_unvalidated",
        validity_reason="Actual-operation deformation and OPD are outside supported material properties.",
        design_reference_maps=design_maps,
        scalar_roundtrip_opd_nm=None,
        front_displacement_nm=None,
        rear_displacement_nm=None,
        actual_heat_W=float(screen.input_heat_W),
        actual_constant_property_max_C=float(np.max(screen.disk_temperature_K) - 273.15),
        design_heat_scale=factor,
        scope=(scope + " The 5 K surfaces are an in-range design reference. "
               "Thermal optical feedback is not applied to the beam fields."))
    return payload


def _conservative_heat_polar(heat_slices, grid, mesh):
    """Preserve integrated heat after Cartesian-to-polar interpolation."""
    source = np.asarray(heat_slices, dtype=float)
    source_edges = np.linspace(0, mesh.z_edges_m[-1], source.shape[0]+1)
    if source.shape[0] != mesh.nz:
        resampled = np.empty((mesh.nz, *grid.shape))
        for j, (left, right) in enumerate(zip(mesh.z_edges_m[:-1], mesh.z_edges_m[1:])):
            overlap = np.maximum(0, np.minimum(source_edges[1:], right) -
                                 np.maximum(source_edges[:-1], left))
            resampled[j] = np.tensordot(overlap, source, axes=(0, 0))/(right-left)
    else:
        resampled = source
    polar = _cartesian_to_polar(resampled, grid, mesh)
    target_W = float(np.sum(source * np.diff(source_edges)[:, None, None]) *
                     grid.dx * grid.dy)
    mapped_W = float(np.sum(polar * mesh.volumes_m3))
    if target_W == 0 and np.all(source == 0):
        return np.zeros(mesh.shape)
    if not np.isfinite(mapped_W) or mapped_W <= 0 or target_W <= 0:
        raise ValueError("thermal heat mapping requires positive finite power")
    return polar * (target_W / mapped_W)


def _fixed_heat_cooler_sensitivity(mesh, heat_polar, configuration):
    """Steady temperature sensitivity with the optical heat held fixed."""
    base = configuration["thermal"]
    rows = []
    for interface_factor, coolant_factor in ((1, 1), (.5, 1), (2, 1),
                                              (1, .5), (1, 2)):
        cfg = deepcopy(configuration)
        cfg["thermal"]["interface_conductance_W_m2K"] = (
            base["interface_conductance_W_m2K"]*interface_factor)
        cfg["thermal"]["coolant_conductance_W_m2K"] = (
            base["coolant_conductance_W_m2K"]*coolant_factor)
        temperature = solve_yb_cooler_temperature(mesh, heat_polar, cfg)
        maximum = float(np.max(temperature.disk_temperature_K))
        rows.append({"interface_factor": interface_factor,
                     "coolant_factor": coolant_factor,
                     "disk_max_C": maximum-273.15,
                     "material_range_valid": 293.15 <= maximum <= 300.0})
    return {"status": "fixed_heat_constant_property_screen",
            "rows": rows,
            "scope": "Steady cooler with unchanged optical heat; outside 20–26.85 °C the constant-property result is unvalidated."}


def _pulsed_thermal_timeline(mesh, heat_polar, grid, configuration, duration_s,
                             cooling_mode="fixed", cooling_target_C=40.0,
                             cooling_h_max_W_m2K=100000.0,
                             internal_max_step_s=10.0, temperature_probes=None,
                             probe_seed=0, initial_temperature_fields=None):
    """Thermal internal steps independent of the scheduled output times.

    Optical heat is an explicitly quasi-steady periodic source. The initial
    Yb excitation buildup is not resolved by this thermal-only comparison.
    """
    if not np.isfinite(internal_max_step_s) or internal_max_step_s <= 0:
        raise ValueError("thermal internal maximum step must be positive")
    h_min = configuration["thermal"]["coolant_conductance_W_m2K"]
    bath = configuration["thermal"]["coolant_temperature_K"]
    gain_h_per_K = 3000.0  # Assumed controller slope; no measured flow curve.
    initial_solver = yb_cooler_solver(mesh, configuration)
    probes = ProbeArray(default_five_probes(mesh.z_edges_m[-1], mesh.r_edges_m[-1])
                        if temperature_probes is None else temperature_probes,
                        mesh, initial_solver.plate, initial_temperature_K=bath,
                        seed=probe_seed)

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

    def control_observation(disk_temperature, plate_temperature):
        if cooling_mode == "feedback":
            return float(np.max(disk_temperature))
        disk_readings = [probes._local_average(probe, disk_temperature,
                                               plate_temperature) + probe.bias_K
                         for probe in probes.probes
                         if probe.region == "disk" and not probe.missing]
        return max(disk_readings) if disk_readings else bath

    # The sensor steady reference uses noiseless settled readings with fixed
    # calibration bias. Transient control below uses only delivered readings.
    if cooling_mode in ("feedback", "sensor_feedback"):
        low, high = h_min, cooling_h_max_W_m2K
        for _ in range(20):
            mid = 0.5 * (low + high)
            candidate = solver_at(mid).steady(heat_polar)
            observation = control_observation(candidate.disk_temperature_K,
                                              candidate.plate_temperature_K)
            if mid < command_h(observation):
                low = mid
            else:
                high = mid
        steady_h = 0.5 * (low + high)
    else:
        steady_h = h_min
    steady = solver_at(steady_h).steady(heat_polar)
    steady_disk_max_C = float(np.max(steady.disk_temperature_K) - 273.15)
    tolerance_K = max(0.05, 0.005 * max(steady_disk_max_C - (bath - 273.15), 0))
    solver = initial_solver
    if initial_temperature_fields is None:
        disk = np.full(mesh.shape, bath)
        plate = np.full(solver.plate.shape, bath)
    else:
        disk = np.asarray(initial_temperature_fields[0],float).copy()
        plate = np.asarray(initial_temperature_fields[1],float).copy()
        if (disk.shape != mesh.shape or plate.shape != solver.plate.shape or
                not np.all(np.isfinite(disk)) or not np.all(np.isfinite(plate))):
            raise ValueError("initial disk/plate temperatures must match the thermal meshes")
    probe_samples = probes.initial_samples(disk, plate)
    pending_samples = list(probe_samples)
    delivered_disk_K = {}
    x, y = grid.mesh
    inside = x*x + y*y <= configuration["geometry"]["disk_radius_m"]**2
    startup = np.geomspace(min(2.5e-5, duration_s / 1000), duration_s, 16)
    end_s = max(30.0, min(300.0, 10 * duration_s))
    continuation = np.geomspace(duration_s * 1.5, end_s, 10) if end_s > duration_s * 1.5 else np.array([])
    times = np.r_[0.0, startup, continuation]
    max_temperature = [float(np.max(disk)-273.15)]
    initial_assembly=solve_yb_assembly(mesh,heat_polar,grid,configuration,
        temperature=SimpleNamespace(disk_temperature_K=disk,
                                    plate_temperature_K=plate),allow_extrapolation=True)
    opd_pv = [float(np.ptp(initial_assembly.screens.mean_roundtrip_opd_m[inside])*1e9)]
    front_pv = [float(np.ptp(initial_assembly.screens.front_uz_m[inside])*1e9)]
    valid = [initial_assembly.material_range_valid]
    conductance = [h_min]
    balance = []
    internal_steps = 0
    largest_internal_step = 0.0
    latest_valid = initial_assembly if initial_assembly.material_range_valid else None
    latest_valid_time = 0.0
    requested = None
    stabilized_at = None
    for previous, now in zip(times[:-1], times[1:]):
        subdivisions = max(1, math.ceil(float(now - previous) / internal_max_step_s))
        dt = float(now - previous) / subdivisions
        for substep in range(subdivisions):
            current_time = float(previous+substep*dt)
            delivered = [event for event in pending_samples
                         if event["available_time_s"] <= current_time+1e-12]
            pending_samples = [event for event in pending_samples
                               if event["available_time_s"] > current_time+1e-12]
            for event in delivered:
                if event["region"] == "disk" and event["valid"]:
                    delivered_disk_K[event["name"]] = event["measured_K"]
            observation = (max(delivered_disk_K.values()) if delivered_disk_K
                           else bath) if cooling_mode == "sensor_feedback" else float(np.max(disk))
            h = command_h(observation)
            if h != solver.coolant.conductance_W_m2K:
                solver = solver_at(h)
            step = solver.advance(disk, plate, heat_polar, dt)
            disk, plate = step.disk_temperature_K, step.plate_temperature_K
            new_samples = probes.advance(float(previous+(substep+1)*dt),
                                         disk, plate)
            probe_samples.extend(new_samples)
            pending_samples.extend(new_samples)
            internal_steps += 1
            largest_internal_step = max(largest_internal_step, dt)
            balance.append(step.relative_balance_error)
        assembly = solve_yb_assembly(mesh, heat_polar, grid, configuration,
                                     temperature=step, allow_extrapolation=True)
        screen = assembly.screens.mean_roundtrip_opd_m[inside]
        face = assembly.screens.front_uz_m[inside]
        max_temperature.append(float(np.max(disk) - 273.15))
        opd_pv.append(float(np.ptp(screen) * 1e9))
        front_pv.append(float(np.ptp(face) * 1e9))
        valid.append(assembly.material_range_valid)
        conductance.append(h)
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
        "initial_state_kind": ("uniform_coolant" if initial_temperature_fields is None
                               else "previous_solved_state"),
        "requested_disk_max_C": float(np.max(requested.temperature.disk_temperature_K) - 273.15),
        "requested_disk_temperature_K": requested.temperature.disk_temperature_K,
        "requested_plate_temperature_K": requested.temperature.plate_temperature_K,
        "requested_material_range_valid": requested_valid,
        "steady_disk_max_C": steady_disk_max_C,
        "steady_coolant_conductance_W_m2K": steady_h,
        "stabilization_tolerance_K": tolerance_K,
        "stabilization_time_s": stabilized_at,
        "stabilized": stabilized_at is not None,
        "energy_balance_relative_max": float(max(balance)),
        "internal_steps": internal_steps,
        "internal_max_step_s": internal_max_step_s,
        "largest_internal_step_s": largest_internal_step,
        "temperature_probes": {
            "status": "synthetic_measurements",
            "seed": probe_seed,
            "specifications": [vars(probe) for probe in probes.probes],
            "samples": sorted(probe_samples,
                              key=lambda event: (event["available_time_s"], event["name"])),
            "scope": ("Five probes sample the same disk and plate thermal fields as the optics. Bias is fixed per probe; noise is readout-only. "
                      + ("Sensor feedback uses only delivered, valid disk readings."
                         if cooling_mode == "sensor_feedback" else
                         "Full-state feedback uses the exact disk maximum as an ideal benchmark."))},
        "final_roundtrip_opd_m": requested.screens.mean_roundtrip_opd_m,
        "latest_valid_time_s": latest_valid_time,
        "latest_valid_roundtrip_opd_m": (latest_valid.screens.mean_roundtrip_opd_m
                                          if latest_valid is not None else np.zeros(grid.shape)),
        "final_front_displacement_nm": requested.screens.front_uz_m * 1e9,
        "final_rear_displacement_nm": requested.screens.rear_uz_m * 1e9,
        "final_disk_displacement_m": requested.displacement.disk_u_m,
        "final_plate_displacement_m": requested.displacement.plate_u_m,
        "scope": ("Thermal startup from uniform coolant temperature with a quasi-steady "
                  "periodic pulse heat source applied after the fast Yb population transient; "
                  "excitation buildup and its short initial heat are not resolved. "
                  "Disk, contact, and copper heat capacities are integrated by backward Euler; each recorded "
                  "state uses internal steps independent of the displayed times. "
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
    density = nonuniform_density(grid, z_edges, settings.disk_radius_m, common).values_m3
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
        target_at_output = angular_spectrum_propagate(
            disk_field_in, grid, wavelength, settings.thickness_m,
            refractive_index=n)
        if settings.post_disk_distance_m:
            field = angular_spectrum_propagate(field, grid, wavelength,
                                               settings.post_disk_distance_m)
            target_at_output = angular_spectrum_propagate(
                target_at_output, grid, wavelength, settings.post_disk_distance_m)
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
            "target_vs_cold_coherent_overlap": coherent_overlap(target_at_output, field),
            "target_vs_cold_intensity_overlap": field_intensity_overlap(target_at_output, field),
            "target_overlap_scope": "Mask-generated cold target propagated to the same output plane, not an ideal pure mode; hot optical feedback unavailable.",
            "mean_excited_fraction": float(np.mean(beta_sum / settings.z_steps)),
            "pump_absorbed_W": float(np.sum(pump - pump_step) * grid.dx * grid.dy),
            "net_heat_W_upper_or_assumed": float(
                np.sum(modal["heat_W_m3_by_slice"]) * dz * grid.dx * grid.dy
                if modal is not None else np.sum(heat) * grid.dx * grid.dy),
        }
        reference_heat_slices = (modal["heat_W_m3_by_slice"] if modal is not None
                                 else np.stack(heat_slices))
    thermal = _empty_thermal("not_requested", "Thermal calculation disabled.")
    if compute_thermal:
        configuration = _assembly_configuration(material, settings)
        mesh = DiskThermalMesh.disk(nr=settings.thermal_nr, nz=settings.thermal_nz,
                                    nphi=settings.thermal_nphi,
                                    radius_m=settings.disk_radius_m, thickness_m=settings.thickness_m)
        heat_polar = _conservative_heat_polar(reference_heat_slices, grid, mesh)
        try:
            thermal = _thermal_or_design_reference(
                mesh, heat_polar, grid, configuration,
                ("Fixed Gaussian cavity heat" if modal is not None else
                 "Pump-only weak-probe heat" if settings.solver_mode == "weak_probe" else
                 f"{selected_beam} saturated-CW heat") +
                " on generic C10100 copper; scalar optical path and surface deformation; photoelasticity omitted.")
        except ValueError as exc:
            thermal = _empty_thermal("out_of_scope", str(exc))
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


def _periodic_ideal_multipass(material, settings, grid, seed, seed_energy_J,
                              pump, scale, pump_passes, repetition_rate_Hz,
                              signal_traversals, time, pulse_shape,
                              fluorescence_energy_J, *,
                              temperature_K_by_slice=None, encounter_opd_m=None):
    """One shared inversion and one coherent field through ideal 1:1 relays.

    Retarded-time intensity transport resolves saturation. The relay has unit
    magnification and no assumed loss. Thin-disk phase is imposed after every
    encounter; the temporal envelope is not assigned a fabricated chirp.
    """
    temperature = temperature_K_by_slice
    pump_state = steady_multipass_pump(
        material, pump, scale, settings.thickness_m, pump_passes,
        temperature_K_by_slice=temperature)
    beta_steady = pump_state.excited_fraction_by_slice
    dark_time = 1/repetition_rate_Hz - signal_traversals*(time[-1]-time[0])
    if dark_time <= 0:
        raise ValueError("pulse window exceeds repetition period")
    disk_input_fluence = seed_energy_J*abs(seed)**2
    initial_signal = pulse_shape[:, None, None]*disk_input_fluence[None]
    opd = (np.zeros(grid.shape) if encounter_opd_m is None else
           np.asarray(encounter_opd_m, float))
    if opd.shape != grid.shape or not np.all(np.isfinite(opd)):
        raise ValueError("encounter OPD must be a finite optical-grid map")
    encounter_phase = np.exp(2j*np.pi*opd/(material.signal_wavelength_nm*1e-9))
    beta_before = beta_steady.copy()
    for cycles in range(1, 101):
        initial_beta = beta_before.copy()
        beta = beta_before.copy()
        signal = initial_signal
        field = np.asarray(seed, complex)*np.sqrt(seed_energy_J)
        signal_gain_fluence = np.zeros_like(beta)
        pulse_beta_integral = np.zeros_like(beta)
        relay_loss_J = 0.0
        encounter_energies_J = []
        for encounter in range(signal_traversals):
            incident_fluence = trapezoid(signal, time, axis=0)
            pulse = propagate_pulse(
                material, time, np.zeros_like(signal), signal,
                settings.thickness_m, settings.z_steps,
                initial_excited_fraction=beta,
                density_scale_by_slice=scale,
                temperature_K_by_slice=temperature)
            signal = pulse.signal_out_W_m2
            outgoing_fluence = trapezoid(signal, time, axis=0)
            field *= np.sqrt(np.divide(
                outgoing_fluence, incident_fluence,
                out=np.ones_like(outgoing_fluence), where=incident_fluence > 0))
            field *= encounter_phase
            signal_gain_fluence += pulse.signal_fluence_change_J_m2_by_slice
            pulse_beta_integral += pulse.excited_fraction_time_integral_s_by_slice
            beta = pulse.final_excited_fraction_by_slice
            after_disk = float(np.sum(outgoing_fluence)*grid.dx*grid.dy)
            encounter_energies_J.append(after_disk)
            if encounter < signal_traversals-1:
                relay_loss_J += after_disk*(1-settings.ideal_relay_power_retention)
                field *= math.sqrt(settings.ideal_relay_power_retention)
                signal = signal*settings.ideal_relay_power_retention
        following, absorbed_integral, recovery_beta_integral = recover_pumped_population(
            material, pump, scale, settings.thickness_m, pump_passes, beta,
            dark_time, 8, temperature_K_by_slice=temperature)
        residual = float(np.max(abs(following-beta_before)))
        beta_before = following
        if residual <= 1e-6:
            break
    else:
        raise RuntimeError("ideal multipass Yb population did not converge")
    beta_average = (pulse_beta_integral+recovery_beta_integral)*repetition_rate_Hz
    pump_absorbed = absorbed_integral*repetition_rate_Hz
    dz = settings.thickness_m/settings.z_steps
    density = material.number_density_m3*scale
    fluorescence_potential = (density*beta_average/material.lifetime_s*
                              fluorescence_energy_J*dz)
    fluorescence = settings.fluorescence_escape_yield*fluorescence_potential
    signal_gain = signal_gain_fluence*repetition_rate_Hz
    storage = density*dz*(following-initial_beta)*repetition_rate_Hz*fluorescence_energy_J
    heat = (pump_absorbed-signal_gain-fluorescence-storage)/dz
    pump_photons = pump_absorbed/(H*C/(material.pump_wavelength_nm*1e-9))
    signal_photons = signal_gain/(H*C/(material.signal_wavelength_nm*1e-9))
    decay_photons = density*dz*beta_average/material.lifetime_s
    storage_photons = density*dz*(following-initial_beta)*repetition_rate_Hz
    balance = pump_photons-signal_photons-decay_photons-storage_photons
    balance_scale = np.sum(abs(pump_photons)+abs(signal_photons)+decay_photons+abs(storage_photons))
    photon_error = float(np.sum(abs(balance))/balance_scale) if balance_scale > 0 else 0.0
    energy_error = (optical_power(seed, grid)*seed_energy_J +
                    float(np.sum(signal_gain_fluence)*grid.dx*grid.dy) -
                    optical_power(field, grid)-relay_loss_J)
    return {
        "output_field": field,
        "output_energy_J": optical_power(field, grid),
        "stored_energy_J": optical_power(field, grid),
        "output_power_trace_W": np.sum(signal, axis=(1, 2))*grid.dx*grid.dy,
        "heat_W_m3_by_slice": heat,
        "pump_absorbed_W_m2_by_slice": pump_absorbed,
        "signal_gain_W_m2_by_slice": signal_gain,
        "escaping_fluorescence_W_m2_by_slice": fluorescence,
        "fluorescence_potential_W_m2_by_slice": fluorescence_potential,
        "excitation_storage_change_W_m2_by_slice": storage,
        "excited_fraction_before_pulse_by_slice": beta_before.copy(),
        "mean_excited_fraction_before_pulse": float(np.mean(beta_before)),
        "local_temperature_K_by_slice": (None if temperature is None else
                                           np.asarray(temperature).copy()),
        "pump_steady_iterations": pump_state.iterations,
        "cycles": cycles, "residual": residual,
        "population_photon_balance_relative_L1": photon_error,
        "optical_energy_balance_residual_J": energy_error,
        "recovery_substeps": 8,
        "recovery_scope": "Pump bleaching recomputed by exponential midpoint recovery; pump during picosecond signal windows omitted. Refine recovery, temporal and axial grids to assess error.",
        "ideal_relay": "unit_magnification_phase_preserving",
        "ideal_relay_power_retention": settings.ideal_relay_power_retention,
        "ideal_relay_loss_J": relay_loss_J,
        "encounter_exit_energies_J": encounter_energies_J,
    }


def simulate_pulsed_seed(material: YbLuAGMaterial, settings: YbGallerySettings,
                         selected_beam: str, seed_energy_J: float,
                         seed_fwhm_s: float, repetition_rate_Hz: float,
                         signal_traversals: int, pump_passes: int = 10,
                         *, compute_thermal: bool = True,
                         operation_duration_s: float = 0.0,
                         cooling_mode: str = "fixed", cooling_target_C: float = 25.0,
                         cooling_h_max_W_m2K: float = 100000.0,
                         architecture: str = "ideal_multipass",
                         regenerative_cavity: RegenerativeCavity | None = None,
                         thermal_optical_mode: str = "lumped_phase",
                         thermal_internal_max_step_s: float = 10.0,
                         temperature_probes=None, probe_seed: int = 0,
                         static_cold_phase_rad=None,
                         slm_correction_phase_rad=None,
                         dataset_physical=None):
    """Periodic pulsed seed through ideal relays or a regenerative cavity.

    Both paths iterate one physical disk population from pulse to pulse.
    Pumping during each short signal pulse is neglected. Both paths recompute
    pump transport and rates as inversion recovers between injected pulses.
    """
    if compute_thermal and thermal_optical_mode == "coupled_steady" and not material.supports_coupled_temperature:
        raise ValueError(f"{material.name}: coupled hot gain is unavailable because temperature-dependent pump spectra are missing. Select cold or the explicitly approximate lumped_phase calculation.")
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
            cooling_mode not in ("fixed", "feedback", "sensor_feedback") or
            not np.isfinite(cooling_target_C) or not 20 <= cooling_target_C <= 250 or
            not np.isfinite(cooling_h_max_W_m2K) or
            not 10000 <= cooling_h_max_W_m2K <= 200000 or
            architecture not in ("ideal_multipass", "regenerative") or
            thermal_optical_mode not in ("cold", "lumped_phase", "coupled_steady") or
            not np.isfinite(thermal_internal_max_step_s) or
            thermal_internal_max_step_s <= 0 or
            not isinstance(probe_seed, int)):
        raise ValueError("invalid pulsed seed settings")
    if (architecture == "regenerative" and regenerative_cavity is not None and
            not np.isclose(regenerative_cavity.disk_diameter_m,
                           2*settings.disk_radius_m, rtol=0, atol=1e-12)):
        raise ValueError("regenerative aperture and physical disk diameter disagree")
    cavity = regenerative_cavity or RegenerativeCavity(
        disk_diameter_m=2*settings.disk_radius_m)
    grid = Grid2D.square(settings.grid_n, settings.field_size_m)
    x, y = grid.mesh
    disk_mask = x*x+y*y <= settings.disk_radius_m**2
    perturb = {} if dataset_physical is None else dict(dataset_physical)
    if perturb and material.name != "Yb:YAG":
        raise ValueError("dataset physical perturbations currently require Yb:YAG")
    def optical_map(name, default=1.0):
        value = np.broadcast_to(np.asarray(perturb.get(name, default), float), grid.shape)
        if not np.all(np.isfinite(value)) or np.any(value <= 0):
            raise ValueError(f"{name} must be finite and positive")
        return value
    def thermal_configuration():
        configuration = _assembly_configuration(material, settings)
        if "yb_concentration_scale" in perturb:
            from ybyag import material_data as yag_data
            yb_scale = optical_map("yb_concentration_scale")
            radii=(np.arange(settings.thermal_nr)+.5)*settings.disk_radius_m/settings.thermal_nr
            azimuths=(np.arange(settings.thermal_nphi)+.5)*2*np.pi/settings.thermal_nphi
            rr,pp=np.meshgrid(radii,azimuths,indexing="ij")
            xq=rr*np.cos(pp); yq=rr*np.sin(pp)
            polar_yb=map_coordinates(yb_scale,
                [(yq-grid.y[0])/grid.dy,(xq-grid.x[0])/grid.dx],
                order=1,mode="nearest")
            concentration=material.yb_at_percent*polar_yb
            if material.yb_at_percent < 14.5:
                conductivity=yag_data.thermal_conductivity_doped(
                    293.15,concentration,family="CT",interpolate_doping=True)
            elif material.yb_at_percent <= 15:
                # CT is tabulated only through 15 at.%. Preserve the CT
                # nominal value; use the measured HT-family concentration
                # trend at 300 K only as an explicit local-slope proxy.
                reference=yag_data.thermal_conductivity_doped(
                    300.,material.yb_at_percent,family="HT",interpolate_doping=True)
                local=yag_data.thermal_conductivity_doped(
                    300.,concentration,family="HT",interpolate_doping=True)
                conductivity=(configuration["thermal"]["disk"]["conductivity_W_mK"]*
                              local/reference)
            else:
                conductivity=yag_data.thermal_conductivity_doped(
                    300.,concentration,family="HT",interpolate_doping=True)
            configuration["thermal"]["disk"]["conductivity_W_mK"]=np.broadcast_to(
                conductivity,(settings.thermal_nz,settings.thermal_nr,
                              settings.thermal_nphi)).copy()
        if "contact_scale_polar" in perturb:
            contact = np.asarray(perturb["contact_scale_polar"], float)
            if contact.shape != (settings.thermal_nr, settings.thermal_nphi) or (
                    not np.all(np.isfinite(contact)) or np.any(contact <= 0)):
                raise ValueError("contact_scale_polar must match the thermal contact mesh")
            configuration["thermal"]["interface_conductance_W_m2K"] *= contact
        if "coolant_temperature_K" in perturb:
            coolant = float(perturb["coolant_temperature_K"])
            if not np.isfinite(coolant) or not 273.15 <= coolant <= 300:
                raise ValueError("dataset coolant temperature outside supported range")
            configuration["thermal"]["coolant_temperature_K"] = coolant
        return configuration
    static_phase = (np.zeros(grid.shape) if static_cold_phase_rad is None else
                    np.asarray(static_cold_phase_rad, dtype=float))
    if static_phase.shape != grid.shape or not np.all(np.isfinite(static_phase)):
        raise ValueError("static cold phase must be a finite disk-plane map")
    if "thickness_scale" in perturb:
        delta_thickness = settings.thickness_m*(optical_map("thickness_scale")-1)
        static_phase = static_phase + disk_mask*(2*np.pi/(material.signal_wavelength_nm*1e-9)*
                                       2*(material.cavity_phase_index-1)*delta_thickness)
    if "surface_figure_m" in perturb:
        height = np.asarray(perturb["surface_figure_m"], float)
        if height.shape != grid.shape or not np.all(np.isfinite(height)):
            raise ValueError("surface figure must match optical grid in metres")
        static_phase = static_phase + disk_mask*4*np.pi*height/(material.signal_wavelength_nm*1e-9)
    static_opd = static_phase*(material.signal_wavelength_nm*1e-9)/(2*np.pi)
    correction_phase = (np.zeros(grid.shape) if slm_correction_phase_rad is None else
                        np.asarray(slm_correction_phase_rad, dtype=float))
    if correction_phase.shape != grid.shape or not np.all(np.isfinite(correction_phase)):
        raise ValueError("SLM correction must be a finite SLM-plane map")
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
                         settings.z_steps + 1), settings.disk_radius_m, common).values_m3
    true_density = density * optical_map("yb_concentration_scale")
    thickness_scale = optical_map("thickness_scale")
    # First-order thin-disk column-depth approximation. The mechanical mesh
    # retains nominal thickness; the surface-height contribution is explicit.
    scale = true_density / material.number_density_m3 * thickness_scale
    source, target_phase = gaussian_seed_and_target_mask(
        grid, settings.waist_m, 1.0, selected_beam,
        material.signal_wavelength_nm * 1e-9, settings.slm_to_disk_distance_m)
    if any(key in perturb for key in ("seed_center_m", "seed_angle_rad",
                                      "seed_ellipticity", "seed_aperture")):
        sx, sy = perturb.get("seed_center_m", (0.0, 0.0))
        ax, ay = perturb.get("seed_angle_rad", (0.0, 0.0))
        ellipticity = float(perturb.get("seed_ellipticity", 1.0))
        if not all(np.isfinite(v) for v in (sx, sy, ax, ay, ellipticity)) or ellipticity <= 0:
            raise ValueError("invalid seed position, angle or ellipticity")
        wx, wy = settings.waist_m*ellipticity**.5, settings.waist_m/ellipticity**.5
        source = np.exp(-((x-sx)**2/wx**2+(y-sy)**2/wy**2)).astype(complex)
        source *= np.exp(1j*2*np.pi/(material.signal_wavelength_nm*1e-9)*(ax*x+ay*y))
        if "seed_aperture" in perturb:
            aperture = perturb["seed_aperture"]
            cx, cy = aperture["center_m"]
            source *= ((x-cx)**2+(y-cy)**2 <= float(aperture["radius_m"])**2)
        norm = float(np.sum(abs(source)**2)*grid.dx*grid.dy)
        if norm <= 0:
            raise ValueError("seed aperture removed all light")
        source /= np.sqrt(norm)
    aberration_phase = phase_pattern(grid, common)
    requested_phase = target_phase + aberration_phase + correction_phase
    if "slm_actual_phase_rad" in perturb:
        applied = np.asarray(perturb["slm_actual_phase_rad"], float)
        if applied.shape != grid.shape or not np.all(np.isfinite(applied)):
            raise ValueError("actual SLM phase must match optical grid")
        external = np.asarray(perturb.get("external_phase_rad", 0),float)
        if external.shape not in ((),grid.shape) or not np.all(np.isfinite(external)):
            raise ValueError("external phase must match optical grid")
        phase = np.mod(applied+external, 2*np.pi)
    else:
        phase = np.mod(requested_phase, 2 * np.pi)
    seed = angular_spectrum_propagate(
        source * np.exp(1j * phase), grid,
        material.signal_wavelength_nm * 1e-9, settings.slm_to_disk_distance_m)
    px, py = perturb.get("pump_center_m", (0.0, 0.0))
    pump = np.exp(-2 * ((x-px)**2 + (y-py)**2) / settings.pump_radius_m**2)
    pump *= settings.pump_power_W / (float(pump.sum()) * grid.dx * grid.dy)
    background_alpha = np.broadcast_to(np.asarray(
        perturb.get("background_absorption_m1", 0), float), grid.shape)
    if np.any(~np.isfinite(background_alpha)) or np.any(background_alpha < 0):
        raise ValueError("background absorption must be finite and nonnegative")
    background_absorbed = disk_mask*pump * -np.expm1(
        -background_alpha*settings.thickness_m*thickness_scale)
    pump = pump-background_absorbed
    dz = settings.thickness_m / settings.z_steps
    time = np.linspace(-3 * seed_fwhm_s, 3 * seed_fwhm_s, 41)
    pulse_shape = np.exp(-4 * np.log(2) * (time / seed_fwhm_s)**2)
    pulse_shape /= trapezoid(pulse_shape, time)
    input_fluence = seed_energy_J * abs(source)**2
    disk_input_fluence = seed_energy_J * abs(seed)**2
    initial_signal = pulse_shape[:, None, None] * disk_input_fluence[None]
    regen = None
    coupled_assembly = None
    coupled_history = None
    coupled_mesh = None
    coupled_cold_field = None
    if architecture == "regenerative":
        def regen_optics(local_temperature, thermal_opd):
            return amplify_regenerative(
                material, grid, seed, seed_energy_J, pump, scale,
                settings.thickness_m, pump_passes, repetition_rate_Hz,
                cavity, fluorescence_spectrum(material).mean_photon_energy_J,
                settings.fluorescence_escape_yield,
                temperature_K_by_slice=local_temperature,
                encounter_opd_m=static_opd+thermal_opd)
        if compute_thermal and thermal_optical_mode == "coupled_steady":
            regen, coupled_assembly, coupled_mesh, coupled_history, coupled_cold_field = (
                _coupled_regenerative_steady(
                    material, settings, grid, seed, seed_energy_J, pump,
                    scale, pump_passes, repetition_rate_Hz,
                    cavity,
                    fluorescence_spectrum(material).mean_photon_energy_J,
                    cooling_mode, cooling_target_C, cooling_h_max_W_m2K,
                     temperature_probes, optical_solver=regen_optics))
        else:
            regen = regen_optics(None, np.zeros(grid.shape))
        fluence_out = abs(regen["output_field"])**2
        signal = None  # Fluence-only map cannot predict a temporal output trace.
        heat_slices = regen["heat_W_m3_by_slice"]
        pump_absorbed = regen["pump_absorbed_W_m2_by_slice"]
        signal_gain = regen["signal_gain_W_m2_by_slice"]
        fluorescence = regen["escaping_fluorescence_W_m2_by_slice"]
        fluorescence_potential = regen["fluorescence_potential_W_m2_by_slice"]
        excitation_storage = regen["excitation_storage_change_W_m2_by_slice"]
        cycles, residual = regen["cycles"], regen["residual"]
        mean_beta_before = regen["mean_excited_fraction_before_pulse"]
        pre_pulse_beta = regen["excited_fraction_before_pulse_by_slice"]
        optical_temperature = regen["local_temperature_K_by_slice"]
        pump_iterations = regen["pump_steady_iterations"]
        disk_output_J = regen["stored_energy_J"]
        field_out = regen["output_field"] / math.sqrt(seed_energy_J)
    else:
        def ideal_optics(local_temperature, encounter_opd):
            return _periodic_ideal_multipass(
                material, settings, grid, seed, seed_energy_J, pump, scale,
                pump_passes, repetition_rate_Hz, signal_traversals, time,
                pulse_shape, fluorescence_spectrum(material).mean_photon_energy_J,
                temperature_K_by_slice=local_temperature,
                encounter_opd_m=static_opd+(0 if encounter_opd is None else encounter_opd))
        if compute_thermal and thermal_optical_mode == "coupled_steady":
            ideal, coupled_assembly, coupled_mesh, coupled_history, coupled_cold_field = (
                _coupled_regenerative_steady(
                    material, settings, grid, seed, seed_energy_J, pump,
                    scale, pump_passes, repetition_rate_Hz, cavity,
                    fluorescence_spectrum(material).mean_photon_energy_J,
                    cooling_mode, cooling_target_C, cooling_h_max_W_m2K,
                    temperature_probes, optical_solver=ideal_optics))
        else:
            ideal = ideal_optics(None, None)
        heat_slices = ideal["heat_W_m3_by_slice"]
        pump_absorbed = ideal["pump_absorbed_W_m2_by_slice"]
        signal_gain = ideal["signal_gain_W_m2_by_slice"]
        fluorescence = ideal["escaping_fluorescence_W_m2_by_slice"]
        fluorescence_potential = ideal["fluorescence_potential_W_m2_by_slice"]
        excitation_storage = ideal["excitation_storage_change_W_m2_by_slice"]
        cycles, residual = ideal["cycles"], ideal["residual"]
        pre_pulse_beta = ideal["excited_fraction_before_pulse_by_slice"]
        mean_beta_before = ideal["mean_excited_fraction_before_pulse"]
        optical_temperature = ideal["local_temperature_K_by_slice"]
        pump_iterations = ideal["pump_steady_iterations"]
        disk_output_J = ideal["output_energy_J"]
        field_out = ideal["output_field"] / math.sqrt(seed_energy_J)
        signal = ideal["output_power_trace_W"]
    if perturb:
        heat_slices = heat_slices + background_absorbed[None]/settings.thickness_m
        pump_absorbed = pump_absorbed + background_absorbed[None]/settings.z_steps
    pixel_area = grid.dx * grid.dy
    heat_W = float(np.sum(heat_slices) * dz * pixel_area)
    absorbed_W = float(np.sum(pump_absorbed) * pixel_area)
    signal_gain_W = float(np.sum(signal_gain) * pixel_area)
    fluorescence_W = float(np.sum(fluorescence) * pixel_area)
    excitation_storage_W = float(np.sum(excitation_storage) * pixel_area)
    fluorescence_potential_W = float(np.sum(fluorescence_potential) * pixel_area)
    fluorescence_sensitivity = {
        "status": "fixed_population_effective_escape_screen",
        "rows": [{"effective_escape_yield": fraction,
                  "heat_W": absorbed_W-signal_gain_W-excitation_storage_W-
                  fraction*fluorescence_potential_W}
                 for fraction in (0.0, 0.5, 0.9, 1.0)],
        "scope": "Effective escaped fraction combines intrinsic radiative efficiency, escape and reabsorption; these are unmeasured separately here. Populations are held fixed.",
    }
    thermal = _empty_thermal("not_requested", "Thermal calculation disabled.")
    if not compute_thermal:
        thermal = _empty_thermal("not_requested", "Thermal calculation was shared from the Gaussian reference case.")
    timeline = None
    cooler_sensitivity = None
    if coupled_assembly is not None:
        thermal = _thermal_payload(
            coupled_assembly,
            "Periodic optical/pump/heat and disk/plate mechanics converged with "
            "local-temperature spectroscopy and per-encounter scalar bulk/surface phase. "
            "The selected property assumption is uncalibrated; photoelasticity omitted.")
        if operation_duration_s:
            configuration = thermal_configuration()
            mesh = DiskThermalMesh.disk(
                nr=settings.thermal_nr, nz=settings.thermal_nz,
                nphi=settings.thermal_nphi, radius_m=settings.disk_radius_m,
                thickness_m=settings.thickness_m)
            heat_polar = _conservative_heat_polar(heat_slices, grid, mesh)
            timeline = _pulsed_thermal_timeline(
                mesh, heat_polar, grid, configuration, operation_duration_s,
                cooling_mode, cooling_target_C, cooling_h_max_W_m2K,
                thermal_internal_max_step_s, temperature_probes, probe_seed)
    elif compute_thermal:
        configuration = thermal_configuration()
        mesh = DiskThermalMesh.disk(nr=settings.thermal_nr, nz=settings.thermal_nz,
                                    nphi=settings.thermal_nphi,
                                    radius_m=settings.disk_radius_m, thickness_m=settings.thickness_m)
        heat_polar = _conservative_heat_polar(heat_slices, grid, mesh)
        cooler_sensitivity = _fixed_heat_cooler_sensitivity(
            mesh, heat_polar, configuration)
        try:
            thermal = _thermal_or_design_reference(
                mesh, heat_polar, grid, configuration,
                ("Periodic pump and signal heat on generic C10100 copper; scalar optical path "
                 "and surface deformation; photoelasticity omitted. The regenerative pump "
                 "transport is recomputed during recovery."
                 if architecture == "regenerative" else
                 "Fixed pump-profile periodic heat on generic C10100 copper; scalar optical "
                 "path and surface deformation; photoelasticity omitted."))
        except ValueError as exc:
            thermal = _empty_thermal("out_of_scope", str(exc))
        if operation_duration_s and thermal_optical_mode == "lumped_phase":
            initial_fields=(None if "initial_disk_temperature_K" not in perturb else
                (perturb["initial_disk_temperature_K"],
                 perturb["initial_plate_temperature_K"]))
            timeline = _pulsed_thermal_timeline(mesh, heat_polar, grid,
                                                 configuration, operation_duration_s,
                                                 cooling_mode, cooling_target_C,
                                                 cooling_h_max_W_m2K,
                                                 thermal_internal_max_step_s,
                                                 temperature_probes, probe_seed,
                                                 initial_fields)
    thermal_feedback_applied = (coupled_assembly is not None or
                                (thermal_optical_mode == "lumped_phase" and
                                 timeline is not None and
                                 timeline["requested_material_range_valid"]))
    cold_field_out = (field_out.copy() if coupled_cold_field is None else
                      coupled_cold_field)
    if thermal_feedback_applied and coupled_assembly is None:
        # This is a lumped post-amplifier OPD approximation. A fully coupled
        # hot-cavity model must apply the screen on each disk encounter.
        effective_traversals = (2 * cavity.round_trips
                                if architecture == "regenerative" else signal_traversals)
        phase_screen = (np.pi * effective_traversals /
                        (material.signal_wavelength_nm * 1e-9) *
                        timeline["final_roundtrip_opd_m"])
        field_out *= np.exp(1j * phase_screen)
    extraction_fluence = abs(field_out)**2 * seed_energy_J
    def intensity_overlap(a, b):
        numerator = float(np.sum(np.sqrt(np.maximum(a, 0) * np.maximum(b, 0))))**2
        denominator = float(np.sum(a) * np.sum(b))
        return numerator / denominator if denominator > 0 else 0.0
    extraction_shape_retention = intensity_overlap(disk_input_fluence, extraction_fluence)
    if settings.post_disk_distance_m:
        field_out = angular_spectrum_propagate(field_out, grid,
                                               material.signal_wavelength_nm * 1e-9,
                                               settings.post_disk_distance_m)
        cold_field_out = angular_spectrum_propagate(
            cold_field_out, grid, material.signal_wavelength_nm * 1e-9,
            settings.post_disk_distance_m)
        target_field_out = angular_spectrum_propagate(
            seed, grid, material.signal_wavelength_nm * 1e-9,
            settings.post_disk_distance_m)
    else:
        target_field_out = seed
    observed_fluence = abs(field_out)**2 * seed_energy_J
    steady_observation = None
    if coupled_assembly is not None:
        configuration = thermal_configuration()
        cooler = yb_cooler_solver(coupled_mesh, configuration)
        probe_array = ProbeArray(
            default_five_probes(coupled_mesh.z_edges_m[-1],
                                coupled_mesh.r_edges_m[-1])
            if temperature_probes is None else temperature_probes,
            coupled_mesh, cooler.plate,
            initial_temperature_K=configuration["thermal"]["coolant_temperature_K"],
            seed=probe_seed)
        readings = probe_array.initial_samples(
            coupled_assembly.temperature.disk_temperature_K,
            coupled_assembly.temperature.plate_temperature_K)
        steady_observation = {
            "mode": "steady_state", "state_timestamp_s": 0.0,
            "probe_seed": probe_seed,
            "timestamp_convention": "settled-state reference time; not startup elapsed time",
            "sensor_response": "assumed fully settled before sample",
            "probe_specifications": [vars(p) for p in probe_array.probes],
            "probe_samples": readings,
            "beam_timestamp_s": 0.0, "slm_target_timestamp_s": 0.0,
        }
    return {
        "grid": grid, "phase_mask": phase,
        "slm_requested_phase_rad": np.mod(requested_phase, 2*np.pi),
        "slm_actual_phase_rad": (np.mod(applied,2*np.pi)
            if "slm_actual_phase_rad" in perturb else phase),
        "target_phase_mask": np.mod(target_phase, 2 * np.pi),
        "aberration_phase_mask": np.mod(aberration_phase, 2 * np.pi),
        "slm_correction_phase_rad": correction_phase,
        "static_cold_phase_rad": static_phase,
        "yb_density_m3": true_density,
        "dataset_physical_scope": (None if not perturb else
            "Yb and thickness maps alter effective optical column density; Yb concentration also sets local thermal conductivity from measured-fit concentration trends (a cross-family relative-slope proxy near 15 at.%). Background loss is a first-order pre-pump absorption and distributed heat term. The FEM retains nominal thickness. Yb:YAG hot gain remains unavailable."),
        "pre_pulse_excited_fraction_by_slice": pre_pulse_beta,
        "optical_temperature_K_by_slice": optical_temperature,
        "selected_beam": selected_beam,
        "input_fluence_J_m2": input_fluence,
        "disk_input_fluence_J_m2": disk_input_fluence,
        "output_fluence_J_m2": observed_fluence,
        "output_complex_field_sqrt_J_m": field_out*math.sqrt(seed_energy_J),
        "heat_W_m3_by_slice": heat_slices,
        "extraction_fluence_J_m2": extraction_fluence,
        "extraction_shape_retention": extraction_shape_retention,
        "observed_shape_retention": intensity_overlap(disk_input_fluence, observed_fluence),
        "field_fidelity": {
            "comparison_plane_m_after_extraction": settings.post_disk_distance_m,
            "target_definition": "phase-mask-generated cold disk input propagated to the same output plane; not an ideal pure LG/HG mode",
            "target_vs_cold_coherent": coherent_overlap(target_field_out, cold_field_out),
            "target_vs_cold_intensity": field_intensity_overlap(target_field_out, cold_field_out),
            "cold_vs_hot_coherent": (coherent_overlap(cold_field_out, field_out)
                                     if thermal_feedback_applied else None),
            "cold_vs_hot_intensity": (field_intensity_overlap(cold_field_out, field_out)
                                      if thermal_feedback_applied else None),
            "hot_comparison_validity": ("extrapolated_unvalidated" if thermal_feedback_applied else
                                        "not_calculated"),
        },
        "output_distance_m": settings.post_disk_distance_m,
        "input_phase": np.angle(source),
        "disk_input_phase": np.angle(seed),
        "output_phase": np.angle(field_out),
        "time_ps": (time * 1e12),
        "input_power_trace_W": np.sum(initial_signal, axis=(1, 2)) * grid.dx * grid.dy,
        "output_power_trace_W": signal,
        "output_trace_status": ("not_calculated_fluence_only_regenerative"
                                if signal is None else "time_sampled_intensity_transport"),
        "input_energy_J": seed_energy_J, "disk_output_energy_J": disk_output_J,
        "output_energy_J": optical_power(field_out, grid) * seed_energy_J,
        "architecture": architecture,
        "regenerative": ({key: value for key, value in regen.items()
                          if key not in ("output_field", "heat_W_m3_by_slice",
                                         "pump_absorbed_W_m2_by_slice", "signal_gain_W_m2_by_slice",
                                         "escaping_fluorescence_W_m2_by_slice",
                                         "fluorescence_potential_W_m2_by_slice",
                                         "excitation_storage_change_W_m2_by_slice",
                                         "excited_fraction_before_pulse_by_slice",
                                         "local_temperature_K_by_slice", "encounter_opd_m")}
                          if regen is not None else None),
        "ideal_multipass": ({key: ideal[key] for key in
                              ("ideal_relay", "ideal_relay_power_retention",
                               "ideal_relay_loss_J", "encounter_exit_energies_J",
                               "population_photon_balance_relative_L1",
                               "optical_energy_balance_residual_J",
                               "recovery_substeps", "recovery_scope")}
                             if architecture == "ideal_multipass" else None),
        "thermal_optical_mode": ("cold" if not compute_thermal else thermal_optical_mode),
        "coupled_steady_convergence": (None if coupled_history is None else {
            "status": "converged", "iterations": len(coupled_history),
            "history": coupled_history,
            "scope": ("One shared " + architecture + " disk state; local-temperature spectra and a scalar per-encounter thermal/surface screen. Generic assembly coefficients are not calibrated.")}),
        "cycles": cycles, "residual": residual,
        "pump_passes": pump_passes,
        "signal_traversals": signal_traversals,
        "effective_signal_traversals": (2*cavity.round_trips
                                          if architecture == "regenerative" else signal_traversals),
        "pump_steady_iterations": pump_iterations,
        "cycle_average_heat_W_upper_or_assumed": heat_W,
        "cycle_average_pump_absorbed_W": absorbed_W,
        "cycle_average_signal_gain_W": signal_gain_W,
        "cycle_average_escaping_fluorescence_W": fluorescence_W,
        "cycle_average_excitation_storage_change_W": excitation_storage_W,
        "fluorescence_effective_escape_sensitivity": fluorescence_sensitivity,
        "cooler_conductance_sensitivity": cooler_sensitivity,
        "fluorescence_escape_yield_assumed": settings.fluorescence_escape_yield,
        "thermal": thermal,
        "thermal_timeline": timeline,
        "steady_state_observation": steady_observation,
        "thermal_feedback_applied": thermal_feedback_applied,
        "thermal_feedback_time_s": (None if coupled_assembly is not None else
                                    operation_duration_s if thermal_feedback_applied else None),
        "steady_coupled_temperature_max_C": (
            float(np.max(coupled_assembly.temperature.disk_temperature_K)-273.15)
            if coupled_assembly is not None else None),
        "mean_excited_fraction_before_pulse": mean_beta_before,
        "scope": (("Coupled steady " + architecture + " mode: one shared Yb disk, periodic pump and "
                   "population, cycle-averaged heat, finite disk/plate, local-temperature "
                   "cross sections and scalar bulk/surface phase at each disk encounter. "
                   "The generic assembly and host thermo-optic/index proxies are not "
                   "experimentally calibrated. Photoelasticity, lifetime variation with "
                   "temperature, spectral saturation, ASE and finite switching are omitted. "
                   if coupled_assembly is not None else
                   "Regenerative mode: short-pulse Frantz–Nodvik saturation in one shared Yb disk, "
                   "two disk traversals per round trip, scalar diffraction over the air gap, "
                   "curved mirror, finite disk aperture, discrete Pockels-cell hold and extraction, "
                   "and pump recovery recomputed between rounds and seed pulses. The held loss, "
                   "switch efficiencies, cavity geometry and host index are engineering assumptions. "
                   "The thermal screen is applied after extraction, not fed back into each cavity "
                   "round; spectral effects, gain narrowing, Kerr phase, ASE, Pockels rise time, "
                   "and damage are not modeled. Output temporal shape is unavailable; "
                   "only pulse fluence is solved. "
                   if architecture == "regenerative" else
                   f"Gaussian TEM00 source shaped by a phase-only target mask plus optional "
                  "added phase, then propagated to the disk. LG, HG, Bessel and flat-top names "
                  "describe approximate targets, not guaranteed pure modes. The flat-top mask "
                  "uses a 48-step scalar alternating-projection design. "
                  f"Periodic two-manifold Yb:LuAG population with {pump_passes} alternating "
                  "CW pump traversals, short pulse gain depletion, and ideal image relays "
                  "between signal traversals. Retarded-time intensity transport omits "
                  "GVD, Kerr phase, walkoff, coherent diffraction within each pulse pass, "
                  "and gain feedback from thermal beam reshaping. The recorded thermal OPD "
                  "is applied as a lumped screen before output-plane diffraction "
                  "only when the requested operating time remains inside the 20–26.85 C "
                  "thermo-mechanical parameter range, which is not a crystal survival limit. "
                  "Cycle-averaged heat integrates pump recovery with changing inversion "
                  "and an assumed fluorescence escape yield; the generic copper assembly "
                  "runs only inside its material temperature range. The output spatial "
                  "phase retains the shaped disk-incident phase "
                  "and includes free-space propagation; high-extraction phase accuracy "
                  "requires a coupled space-time field solver."))
    }
