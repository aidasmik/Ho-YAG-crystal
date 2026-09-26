"""Physical Yb:YAG forward solves and calibrated diagnostic observations."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import json
import numpy as np

from hoyag.propagation import Grid2D, angular_spectrum_propagate
from hoyag.thermal import DiskThermalMesh
from ybluag.assembly import yb_cooler_solver
from ybluag.beam_shaping import gaussian_seed_and_target_mask
from ybluag.camera_dataset import CameraSettings
from ybluag.gallery import (
    YbGallerySettings,
    _assembly_configuration,
    simulate_pulsed_seed,
)
from ybyag.model import YbYAGMaterial
from ybyag_dataset.distortions.camera import capture, sample_camera_setup
from ybyag_dataset.distortions.material import sample_material
from ybyag_dataset.distortions.optical import sample_optical_phase
from ybyag_dataset.distortions.slm import apply_slm, sample_slm_setup
from ybyag_dataset.distortions.thermal import sample_contact
from ybyag_dataset.distortions.sensors import SensorSession
from .controller import Observation, PhaseControlGeometry, correction_basis
from .interferometry import (
    calibrated_interferometer, capture_interferograms, reference_amplitude,
)


@dataclass(frozen=True)
class EpisodeConfig:
    target: str = "Helical LG(0,+1)"
    mode: str = "snapshot"
    correction_enabled: bool = True
    seed: int = 91
    pump_W: float = 0.1
    yb_at_percent: float = 10.0
    thickness_um: float = 100.0
    pump_radius_mm: float = 1.0
    disk_radius_mm: float = 5.0
    waist_mm: float = 0.6
    seed_energy_nj: float = 10.0
    seed_fwhm_ps: float = 10.0
    repetition_rate_kHz: float = 10.0
    pump_passes: int = 10
    signal_traversals: int = 2
    grid_n: int = 48
    field_size_mm: float = 12.0
    thermal_nr: int = 4
    thermal_nphi: int = 8
    thermal_nz: int = 1
    operation_duration_s: float = 0.01
    thermal_timeline_mode: str = "requested_only"
    diagnostic_distance_m: float = 0.05
    diagnostic_astigmatism_waves: float = 0.25
    camera_width: int = 320
    camera_height: int = 176
    camera_fov_mm: float = 6.0
    slm_delay_s: float = 0.01
    slm_settle_s: float = 0.02
    control_period_s: float = 0.0
    slm_drift_fraction_per_sqrt_s: float = 0.0
    exposure_s: float = 0.01
    pulses_per_exposure: int = 100
    enable_material: bool = True
    enable_slm_error: bool = True
    enable_external_optics: bool = True
    enable_camera_noise: bool = True
    enable_thermal_variation: bool = False
    coolant_setpoint_C: float = 20.2
    coolant_jitter_K: float = 0.03
    coolant_walk_K_per_sqrt_s: float = 0.01
    pump_jitter_fraction: float = 0.005
    pump_walk_fraction_per_sqrt_s: float = 0.002
    pump_radius_jitter_fraction: float = 0.003
    pump_pointing_jitter_um: float = 5.0
    seed_energy_jitter_fraction: float = 0.005
    seed_pointing_jitter_um: float = 2.0
    camera_read_noise_e: float = 3.0
    camera_background_e: float = 2.0
    camera_gain_jitter_fraction: float = 0.005
    camera_gain_random_walk_per_sqrt_s: float = 0.001
    probe_noise_scale: float = 1.0
    photodiode_noise_fraction: float = 0.005
    architecture: str = "ideal_multipass"

    def __post_init__(self):
        if self.mode not in ("snapshot", "in_situ"):
            raise ValueError("mode must be snapshot or in_situ")
        if self.thermal_timeline_mode not in ("full", "requested_only"):
            raise ValueError("thermal timeline mode must be full or requested_only")
        if (
            not isinstance(self.grid_n, int)
            or not 32 <= self.grid_n <= 256
            or not isinstance(self.camera_width, int)
            or self.camera_width < 16
            or not isinstance(self.camera_height, int)
            or self.camera_height < 16
        ):
            raise ValueError("invalid optical/camera grid")
        if (
            self.camera_width % 8
            or self.camera_height % 8
            or self.camera_height > 1080
            or self.camera_width > 1920
        ):
            raise ValueError(
                "camera dimensions must be multiples of 8 within sensor limits"
            )
        positive = (
            "pump_W",
            "thickness_um",
            "pump_radius_mm",
            "disk_radius_mm",
            "waist_mm",
            "seed_energy_nj",
            "seed_fwhm_ps",
            "repetition_rate_kHz",
            "operation_duration_s",
            "exposure_s",
            "field_size_mm",
            "camera_fov_mm",
            "diagnostic_distance_m",
        )
        if any(
            not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0
            for name in positive
        ):
            raise ValueError(
                "physical dimensions, powers and times must be finite and positive"
            )
        if not math.isfinite(self.yb_at_percent) or not 0 < self.yb_at_percent <= 100:
            raise ValueError("Yb concentration must be within (0, 100] at.%")
        if (
            self.field_size_mm < 2 * self.disk_radius_mm
            or self.operation_duration_s > 120
            or not math.isfinite(self.diagnostic_astigmatism_waves)
        ):
            raise ValueError(
                "invalid optical field, thermal duration or diagnostic diversity"
            )
        if (
            not math.isfinite(self.slm_delay_s)
            or self.slm_delay_s < 0
            or not math.isfinite(self.slm_settle_s)
            or self.slm_settle_s < 0
        ):
            raise ValueError("SLM delay and settling time must be nonnegative")
        if (
            not isinstance(self.pump_passes, int)
            or not 1 <= self.pump_passes <= 48
            or not isinstance(self.signal_traversals, int)
            or not 1 <= self.signal_traversals <= 10
        ):
            raise ValueError("invalid pump-pass or signal-traversal count")
        if (
            not isinstance(self.pulses_per_exposure, int)
            or self.pulses_per_exposure < 1
            or self.pulses_per_exposure
            > self.exposure_s * self.repetition_rate_kHz * 1e3 + 1e-9
        ):
            raise ValueError("exposure cannot contain the requested number of pulses")
        if self.architecture != "ideal_multipass":
            raise ValueError(
                "controller fixture currently supports ideal_multipass only"
            )
        if self.enable_thermal_variation and self.mode != "in_situ":
            raise ValueError("changing thermal/pump conditions require in_situ mode")
        operating_values = (
            self.coolant_jitter_K,
            self.coolant_walk_K_per_sqrt_s,
            self.pump_jitter_fraction,
            self.pump_walk_fraction_per_sqrt_s,
            self.pump_radius_jitter_fraction,
            self.pump_pointing_jitter_um,
            self.seed_energy_jitter_fraction,
            self.seed_pointing_jitter_um,
        )
        if (
            not math.isfinite(self.coolant_setpoint_C)
            or not 20 <= self.coolant_setpoint_C <= 25
            or any(not math.isfinite(v) or v < 0 for v in operating_values)
            or self.coolant_jitter_K > 0.2
            or self.coolant_walk_K_per_sqrt_s > 0.1
            or self.pump_jitter_fraction > 0.05
            or self.pump_walk_fraction_per_sqrt_s > 0.05
            or self.pump_radius_jitter_fraction > 0.05
            or self.pump_pointing_jitter_um > 100
            or self.seed_energy_jitter_fraction > 0.05
            or self.seed_pointing_jitter_um > 100
        ):
            raise ValueError(
                "thermal/pump variation settings exceed supported small ranges"
            )
        if (
            not math.isfinite(self.control_period_s)
            or not 0 <= self.control_period_s <= 2
            or not math.isfinite(self.slm_drift_fraction_per_sqrt_s)
            or not 0 <= self.slm_drift_fraction_per_sqrt_s <= 0.02
        ):
            raise ValueError("invalid actuator timing or calibration drift")
        noise_values = (
            self.camera_read_noise_e,
            self.camera_background_e,
            self.camera_gain_jitter_fraction,
            self.camera_gain_random_walk_per_sqrt_s,
            self.probe_noise_scale,
            self.photodiode_noise_fraction,
        )
        if any(not math.isfinite(v) or v < 0 for v in noise_values):
            raise ValueError("noise settings must be finite and nonnegative")
        if (
            self.camera_gain_jitter_fraction > 0.2
            or self.camera_gain_random_walk_per_sqrt_s > 0.1
        ):
            raise ValueError("camera gain variation exceeds supported range")


def _default_ranges():
    path = Path(__file__).resolve().parents[2] / "config" / "ybyag_nn_dataset.json"
    return json.loads(path.read_text(encoding="utf-8"))["ranges"]


def diagnostic_fields(
    field, grid, wavelength_m, waist_m, distance_m, astigmatism_waves
):
    """Two sampled output arms; the second has a known asymmetric phase."""
    x, y = grid.mesh
    diversity = 2 * np.pi * astigmatism_waves * (x * x - y * y) / (2 * waist_m) ** 2
    return (
        field,
        angular_spectrum_propagate(
            field * np.exp(1j * diversity), grid, wavelength_m, distance_m
        ),
    )


class SimulationPlant:
    """The controller receives only observe(); validation may inspect truth()."""

    def __init__(self, episode: EpisodeConfig, mode_count=14, *, measurement_mode="intensity"):
        if measurement_mode not in ("intensity", "interferometric"):
            raise ValueError("unknown measurement mode")
        self.measurement_mode = measurement_mode
        self.episode = episode
        self.ranges = _default_ranges()
        self.material = YbYAGMaterial(
            yb_at_percent=episode.yb_at_percent,
            pump_wavelength_nm=969.0,
            signal_wavelength_nm=1030.0,
        )
        self.settings = YbGallerySettings(
            pump_power_W=episode.pump_W,
            pump_radius_m=episode.pump_radius_mm * 1e-3,
            thickness_m=episode.thickness_um * 1e-6,
            disk_radius_m=episode.disk_radius_mm * 1e-3,
            assembly_property_model="yag_rt_proxy",
            waist_m=episode.waist_mm * 1e-3,
            slm_to_disk_distance_m=0.25,
            post_disk_distance_m=0,
            grid_n=episode.grid_n,
            field_size_m=episode.field_size_mm * 1e-3,
            z_steps=1,
            thermal_nr=episode.thermal_nr,
            thermal_nphi=episode.thermal_nphi,
            thermal_nz=episode.thermal_nz,
            cluster_contrast=0.0,
        )
        self.grid = Grid2D.square(episode.grid_n, episode.field_size_mm * 1e-3)
        x, y = self.grid.mesh
        self.source_field, self.target_phase = gaussian_seed_and_target_mask(
            self.grid,
            self.settings.waist_m,
            1.0,
            episode.target,
            1030e-9,
            self.settings.slm_to_disk_distance_m,
        )
        self.basis = correction_basis(
            self.grid.x, self.grid.y, self.settings.waist_m, mode_count
        )
        self.material_maps = sample_material(
            self.grid.shape,
            self.ranges,
            episode.seed + 1,
            enabled=episode.enable_material,
        )
        self.contact = sample_contact(
            (episode.thermal_nr, episode.thermal_nphi),
            self.ranges,
            episode.seed + 2,
            enabled=episode.enable_material,
        )
        self.external_phase, _ = sample_optical_phase(
            x,
            y,
            2 * self.settings.waist_m,
            1030e-9,
            self.ranges,
            episode.seed + 3,
            enabled=episode.enable_external_optics,
        )
        self.slm_setup = sample_slm_setup(
            self.grid.shape,
            self.ranges,
            episode.seed + 4,
            enabled=episode.enable_slm_error,
        )
        base_camera = CameraSettings(
            width=episode.camera_width,
            height=episode.camera_height,
            object_fov_width_mm=episode.camera_fov_mm,
            pulses_per_exposure=episode.pulses_per_exposure,
            exposure_s=episode.exposure_s,
            read_noise_e=episode.camera_read_noise_e,
            background_e=episode.camera_background_e,
        )
        self.camera_setup = sample_camera_setup(
            base_camera,
            self.ranges,
            episode.seed + 5,
            enabled=episode.enable_camera_noise,
        )
        self.camera = base_camera
        self.noise_rng = np.random.default_rng(episode.seed + 7)
        self.operating_rng = np.random.default_rng(episode.seed + 8)
        self.slm_drift_rng = np.random.default_rng(episode.seed + 9)
        self.slm_drift_fraction = 0.0
        self.last_slm_time_s = 0.0
        self.operating_walk_K = 0.0
        self.operating_walk_pump_fraction = 0.0
        self.last_operating_time_s = 0.0
        self.latest_operating_point = None
        self.camera_gain_walk = 0.0
        self.last_camera_time_s = 0.0
        self.latest_noise_state = {}
        self.current_time_s = 0.0
        self.initial_thermal = None
        self.last_result = None
        self.last_observation = None
        self.command_log = []
        self.observations = 0
        self.full_solves = 0
        self.full_solve_limit = None
        self.previous_delivered = np.zeros(mode_count)
        self.phase_camera = None
        self.phase_setup = None
        self.local_oscillator = None
        mesh = DiskThermalMesh.disk(
            nr=episode.thermal_nr,
            nphi=episode.thermal_nphi,
            nz=episode.thermal_nz,
            radius_m=self.settings.disk_radius_m,
            thickness_m=self.settings.thickness_m,
        )
        assembly = _assembly_configuration(self.material, self.settings)
        plate = yb_cooler_solver(mesh, assembly).plate
        sensor_ranges = dict(self.ranges)
        sensor_ranges["probe_noise_K"] = [
            v * episode.probe_noise_scale for v in self.ranges["probe_noise_K"]
        ]
        self.sensors = SensorSession(
            mesh,
            plate,
            sensor_ranges,
            episode.seed + 6,
            enabled=episode.enable_camera_noise,
        )

    def _sample_operating_point(self, time_s):
        ep = self.episode
        dt = max(0.0, time_s - self.last_operating_time_s)
        self.last_operating_time_s = time_s
        self.operating_walk_K = float(
            np.clip(
                self.operating_walk_K
                + self.operating_rng.normal(
                    0.0, ep.coolant_walk_K_per_sqrt_s * np.sqrt(dt)
                ),
                -0.2,
                0.2,
            )
        )
        self.operating_walk_pump_fraction = float(
            np.clip(
                self.operating_walk_pump_fraction
                + self.operating_rng.normal(
                    0.0, ep.pump_walk_fraction_per_sqrt_s * np.sqrt(dt)
                ),
                -0.05,
                0.05,
            )
        )
        coolant = float(
            np.clip(
                273.15
                + ep.coolant_setpoint_C
                + self.operating_walk_K
                + self.operating_rng.normal(0.0, ep.coolant_jitter_K),
                293.15,
                298.15,
            )
        )
        power = float(
            ep.pump_W
            * (
                1
                + self.operating_walk_pump_fraction
                + self.operating_rng.normal(0.0, ep.pump_jitter_fraction)
            )
        )
        radius = float(
            ep.pump_radius_mm
            * (1 + self.operating_rng.normal(0.0, ep.pump_radius_jitter_fraction))
        )
        center = tuple(
            self.operating_rng.normal(0.0, ep.pump_pointing_jitter_um * 1e-6, 2)
        )
        seed_energy = float(
            ep.seed_energy_nj
            * (1 + self.operating_rng.normal(0.0, ep.seed_energy_jitter_fraction))
        )
        seed_center = tuple(
            self.operating_rng.normal(0.0, ep.seed_pointing_jitter_um * 1e-6, 2)
        )
        return dict(
            coolant_temperature_K=coolant,
            pump_power_W=power,
            pump_radius_m=radius * 1e-3,
            pump_center_m=center,
            seed_energy_nj=seed_energy,
            seed_center_m=seed_center,
        )

    def phase_geometry(self):
        """Nominal calibrated SLM-to-output path supplied to the controller."""
        return PhaseControlGeometry(
            self.grid, 1030e-9, self.settings.slm_to_disk_distance_m,
            self.source_field.copy(), self.target_phase.copy(),
        )

    def correction_phase(self, command):
        command = np.asarray(command, float)
        if command.shape == self.grid.shape:
            return command.copy()
        if command.shape == (len(self.basis),):
            return np.tensordot(command, self.basis, axes=(0, 0))
        raise ValueError("invalid SLM phase command geometry")

    def phase_compensation_truth(self, result=None):
        """First-order phase-only oracle and the SLM phase actually delivered.

        This is validation truth, never an input to the measured controller.
        The oracle uses the same thermal OPD, cold disk phase, and external
        optics as the forward solve. It cannot undo amplitude/gain errors or
        invert the finite-resolution SLM response.
        """
        result = self.last_result if result is None else result
        if result is None:
            raise ValueError("no solved optical state")
        wavelength_m = self.material.signal_wavelength_nm * 1e-9
        desired_disk = angular_spectrum_propagate(
            self.source_field * np.exp(1j * self.target_phase),
            self.grid, wavelength_m, self.settings.slm_to_disk_distance_m,
        )
        traversals = float(result["effective_signal_traversals"])
        timeline = result["thermal_timeline"]
        disk_phase = (
            np.pi * traversals / wavelength_m
            * np.asarray(timeline["final_roundtrip_opd_m"], float)
        )
        cold_phase = traversals * np.asarray(result["static_cold_phase_rad"], float)
        compensated_input = angular_spectrum_propagate(
            desired_disk * np.exp(-1j * (disk_phase + cold_phase)),
            self.grid, wavelength_m, -self.settings.slm_to_disk_distance_m,
        )
        ideal = np.angle(np.exp(1j * (
            np.angle(compensated_input) - self.target_phase - self.external_phase
        )))
        delivered = np.angle(np.exp(1j * (
            np.asarray(result["slm_actual_phase_rad"], float) - self.target_phase
        )))
        pupil = abs(self.source_field) ** 2 >= .001 * float(np.max(abs(self.source_field) ** 2))
        return np.where(pupil, ideal, np.nan), np.where(pupil, delivered, np.nan)

    def _solve(self, delivered, ideal=False, duration_s=None):
        if (
            self.full_solve_limit is not None
            and self.full_solves >= self.full_solve_limit
        ):
            raise StopIteration("full-simulator evaluation budget reached")
        # Both the ideal camera target and the perturbed plant use the same
        # nominal bath. Only the observed plant receives coolant drift.
        nominal_coolant_K = 273.15 + self.episode.coolant_setpoint_C
        if ideal:
            phase = self.target_phase
            physical = {
                "slm_actual_phase_rad": phase,
                "coolant_temperature_K": nominal_coolant_K,
            }
        else:
            requested = np.mod(
                self.target_phase + self.correction_phase(delivered),
                2 * np.pi,
            )
            applied, _ = apply_slm(
                requested, self.slm_setup, drift_fraction=self.slm_drift_fraction
            )
            physical = {
                **self.material_maps,
                "contact_scale_polar": self.contact,
                "external_phase_rad": self.external_phase,
                "slm_actual_phase_rad": applied,
                "coolant_temperature_K": nominal_coolant_K,
            }
            if self.episode.enable_thermal_variation:
                physical["coolant_temperature_K"] = self.latest_operating_point[
                    "coolant_temperature_K"
                ]
                physical["pump_center_m"] = self.latest_operating_point["pump_center_m"]
                physical["seed_center_m"] = self.latest_operating_point["seed_center_m"]
            if self.episode.mode == "in_situ" and self.initial_thermal is not None:
                physical["initial_disk_temperature_K"] = self.initial_thermal[0]
                physical["initial_plate_temperature_K"] = self.initial_thermal[1]
        settings = (
            replace(
                self.settings,
                pump_power_W=self.latest_operating_point["pump_power_W"],
                pump_radius_m=self.latest_operating_point["pump_radius_m"],
            )
            if not ideal and self.episode.enable_thermal_variation
            else self.settings
        )
        seed_energy_nj = (
            self.latest_operating_point["seed_energy_nj"]
            if not ideal and self.episode.enable_thermal_variation
            else self.episode.seed_energy_nj
        )
        result = simulate_pulsed_seed(
            self.material,
            settings,
            self.episode.target,
            seed_energy_nj * 1e-9,
            self.episode.seed_fwhm_ps * 1e-12,
            self.episode.repetition_rate_kHz * 1e3,
            self.episode.signal_traversals,
            pump_passes=self.episode.pump_passes,
            compute_thermal=True,
            operation_duration_s=(
                self.episode.operation_duration_s if duration_s is None else duration_s
            ),
            cooling_mode="fixed",
            thermal_optical_mode="lumped_phase",
            thermal_timeline_mode=self.episode.thermal_timeline_mode,
            dataset_physical=physical,
        )
        timeline = result.get("thermal_timeline")
        if (
            timeline is None
            or not timeline["requested_material_range_valid"]
            or (
                self.episode.thermal_timeline_mode == "requested_only"
                and not timeline.get("all_intermediate_material_states_valid", True)
            )
            or not result["thermal_feedback_applied"]
        ):
            raise ValueError(
                "Yb:YAG thermal state outside the supported 293–300 K model"
            )
        self.full_solves += 1
        return result

    def _carry_thermal(self, result):
        timeline = result["thermal_timeline"]
        self.initial_thermal = (
            np.asarray(timeline["requested_disk_temperature_K"]).copy(),
            np.asarray(timeline["requested_plate_temperature_K"]).copy(),
        )

    def _capture(self, result, coefficients, *, noisy, clock_s, seed):
        field = np.asarray(result["output_complex_field_sqrt_J_m"], complex)
        frames = []
        sat = []
        if noisy:
            elapsed = max(0.0, clock_s - self.last_camera_time_s)
            self.camera_gain_walk = np.clip(
                self.camera_gain_walk
                + self.noise_rng.normal(
                    0.0,
                    self.episode.camera_gain_random_walk_per_sqrt_s * np.sqrt(elapsed),
                ),
                -0.2,
                0.2,
            )
            self.last_camera_time_s = clock_s
        gains = []
        for arm, arm_field in enumerate(
            diagnostic_fields(
                field,
                self.grid,
                1030e-9,
                self.settings.waist_m,
                self.episode.diagnostic_distance_m,
                self.episode.diagnostic_astigmatism_waves,
            )
        ):
            gain = (
                (
                    1.0
                    + self.camera_gain_walk
                    + self.noise_rng.normal(
                        0.0, self.episode.camera_gain_jitter_fraction
                    )
                )
                if noisy
                else 1.0
            )
            gains.append(float(gain))
            camera = replace(
                self.camera,
                optical_throughput=min(
                    1.0, max(1e-12, self.camera.optical_throughput * gain)
                ),
            )
            frame, _, info = capture(
                abs(arm_field) ** 2,
                self.grid.x,
                self.grid.y,
                1030e-9,
                camera,
                self.camera_setup,
                seed + arm,
                enabled=noisy,
            )
            frames.append(frame)
            sat.append(info["saturated_fraction"])
        timeline = result["thermal_timeline"]
        # Even a noiseless diagnostic must read the same thermal field used
        # for deformation and OPD. SensorSession disables readout errors when
        # camera/probe noise is off, but still samples the physical probes.
        sensor = self.sensors.sample(
            timeline["requested_disk_temperature_K"],
            timeline["requested_plate_temperature_K"],
            clock_s,
        )
        photodiode = result["output_energy_J"]
        if noisy:
            photodiode *= 1 + np.random.default_rng(seed + 9).normal(
                0, self.episode.photodiode_noise_fraction
            )
        measured = np.asarray(sensor["measured_temperature_K"], float)
        self.latest_noise_state = dict(
            camera_gain_factors=gains,
            camera_gain_walk_fraction=float(self.camera_gain_walk),
            camera_read_noise_e=self.episode.camera_read_noise_e,
            camera_background_e=self.episode.camera_background_e,
            probe_noise_scale=self.episode.probe_noise_scale,
        )
        interferograms = None
        measured_phase_field = None
        if self.measurement_mode == "interferometric":
            if self.phase_camera is None or self.local_oscillator is None:
                raise ValueError("interferometer has not been calibrated")
            interferograms, measured_phase_field, phase_sat = capture_interferograms(
                field, self.local_oscillator, self.grid.x, self.grid.y, 1030e-9,
                self.phase_camera, self.phase_setup, seed + 100, noisy=noisy,
            )
            sat.append(phase_sat)
        return Observation(
            np.stack(frames),
            float(photodiode),
            measured,
            np.isfinite(measured),
            coefficients.copy(),
            clock_s,
            max(sat),
            valid=all(v <= 0.002 for v in sat),
            phase_field=measured_phase_field,
            interferograms_adu=interferograms,
        )

    def reference(self):
        # Solve the uniform pumped disk once to obtain its thermal state and
        # gain. The control target is its field *before* the lumped thermal
        # phase screen: the intended structured phase without aberration.
        hot_result = self._solve(np.zeros(len(self.basis)), ideal=True)
        cold_field = np.asarray(
            hot_result.get(
                "cold_output_complex_field_sqrt_J_m",
                hot_result["output_complex_field_sqrt_J_m"],
            ),
            complex,
        )
        result = {
            **hot_result,
            "output_complex_field_sqrt_J_m": cold_field,
            "output_fluence_J_m2": abs(cold_field) ** 2,
            "output_phase": np.angle(cold_field),
            "output_energy_J": float(np.sum(abs(cold_field) ** 2) * self.grid.dx * self.grid.dy),
        }
        # Calibrate against both diagnostic arms using the same pixel mapping,
        # PSF and fixed-pattern response as the actual detector. A single
        # extraction-plane peak can underpredict the defocused arm's charge.
        # Leave headroom for correction probes and slow operating drift.
        unattenuated = replace(self.camera, optical_throughput=1.0)
        field = np.asarray(result["output_complex_field_sqrt_J_m"], complex)
        peaks = []
        for arm, arm_field in enumerate(
            diagnostic_fields(
                field,
                self.grid,
                1030e-9,
                self.settings.waist_m,
                self.episode.diagnostic_distance_m,
                self.episode.diagnostic_astigmatism_waves,
            )
        ):
            _, _, info = capture(
                abs(arm_field) ** 2,
                self.grid.x,
                self.grid.y,
                1030e-9,
                unattenuated,
                self.camera_setup,
                self.episode.seed + 100 + arm,
                enabled=False,
            )
            peaks.append(info["expected_electron_peak"])
        if not np.all(np.isfinite(peaks)) or max(peaks) <= 0:
            raise ValueError("reference has no finite illuminated camera pixels")
        self.camera = replace(
            self.camera,
            optical_throughput=min(1.0, 0.15 * self.camera.full_well_e / max(peaks)),
        )
        if self.measurement_mode == "interferometric":
            self.local_oscillator = reference_amplitude(
                field, self.grid.x, self.grid.y, self.settings.waist_m,
            )
            self.phase_camera, self.phase_setup = calibrated_interferometer(
                field, self.local_oscillator, self.grid.x, self.grid.y, 1030e-9,
                self.camera, self.ranges, self.episode.seed + 501,
                noisy=self.episode.enable_camera_noise,
            )
        self.reference_result = result
        reference_observation = self._capture(
            result,
            np.zeros(len(self.basis)),
            noisy=False,
            clock_s=0.0,
            seed=self.episode.seed + 100,
        )
        # The requested operating duration establishes the initial hot state
        # once. Subsequent in-situ captures advance by the control period,
        # rather than repeating that entire duration on every camera frame.
        if self.episode.mode == "in_situ" and self.episode.operation_duration_s > 0:
            if self.episode.enable_thermal_variation:
                self.latest_operating_point = self._sample_operating_point(0.0)
            warm = self._solve(
                np.zeros(len(self.basis)), duration_s=self.episode.operation_duration_s
            )
            self._carry_thermal(warm)
            self.current_time_s = self.episode.operation_duration_s
            self.sensors.sample(
                warm["thermal_timeline"]["requested_disk_temperature_K"],
                warm["thermal_timeline"]["requested_plate_temperature_K"],
                self.current_time_s,
            )
        return reference_observation, result["output_energy_J"]

    def observe(self, coefficients_rad):
        a = np.asarray(coefficients_rad, float)
        allowed = ((len(self.basis),),)
        if self.measurement_mode == "interferometric":
            allowed = allowed + (self.grid.shape,)
        if a.shape not in allowed or not np.all(np.isfinite(a)):
            raise ValueError("invalid SLM phase command")
        # Reserve the whole observation before advancing the simulated clock
        # or thermal state. A budget stop must not occur after a held-command
        # solve but before the requested command is measured.
        held_solve = self.episode.mode == "in_situ" and self.episode.slm_delay_s > 0
        solves_needed = 1 + int(held_solve)
        if (
            self.full_solve_limit is not None
            and self.full_solves + solves_needed > self.full_solve_limit
        ):
            raise StopIteration("full-simulator evaluation budget reached")
        # An in-situ command must settle before its exposure. Probes consume
        # elapsed physical time; thermal state is carried from each solve.
        start = self.current_time_s
        if self.episode.mode == "in_situ" and self.episode.enable_slm_error:
            dt = max(0.0, start - self.last_slm_time_s)
            self.last_slm_time_s = start
            self.slm_drift_fraction = float(
                np.clip(
                    self.slm_drift_fraction
                    + self.slm_drift_rng.normal(
                        0.0, self.episode.slm_drift_fraction_per_sqrt_s * np.sqrt(dt)
                    ),
                    -0.02,
                    0.02,
                )
            )
        if self.episode.enable_thermal_variation:
            self.latest_operating_point = self._sample_operating_point(start)
        if held_solve:
            held = self._solve(
                self.previous_delivered, duration_s=self.episode.slm_delay_s
            )
            self._carry_thermal(held)
        delivered_time = start + self.episode.slm_delay_s
        exposure_duration = self.episode.exposure_s * (
            4 if self.measurement_mode == "interferometric" else 1
        )
        effective_settle = max(
            self.episode.slm_settle_s,
            (
                self.episode.operation_duration_s - exposure_duration
                if self.episode.mode == "snapshot"
                else 0.0
            ),
            (
                self.episode.control_period_s
                - self.episode.slm_delay_s
                - exposure_duration
                if self.episode.mode == "in_situ"
                else 0.0
            ),
        )
        exposure_start = delivered_time + effective_settle
        self.current_time_s = exposure_start + exposure_duration
        delivered = a.copy()
        result = self._solve(
            delivered,
            duration_s=(
                effective_settle + exposure_duration
                if self.episode.mode == "in_situ"
                else None
            ),
        )
        if self.episode.mode == "in_situ":
            self._carry_thermal(result)
        self.previous_delivered = delivered.copy()
        self.observations += 1
        obs = self._capture(
            result,
            delivered,
            noisy=self.episode.enable_camera_noise,
            clock_s=self.current_time_s,
            seed=self.episode.seed + 1000 * self.observations,
        )
        self.last_result = result
        self.last_observation = obs
        self.command_log.append(
            dict(
                requested_time_s=start,
                delivered_time_s=delivered_time,
                exposure_start_s=exposure_start,
                exposure_end_s=self.current_time_s,
                coefficients_rad=a.tolist(),
                operating_point=self.latest_operating_point,
                slm_calibration_drift_fraction=self.slm_drift_fraction,
                measured_energy_J=obs.measured_energy_J,
                valid=obs.valid,
            )
        )
        return obs

    def validation_truth(self, result=None):
        """Separate, truth-assisted report; never passed into controller."""
        result = result or self.last_result
        if result is None:
            return None
        field = np.asarray(result["output_complex_field_sqrt_J_m"], complex)
        target = np.asarray(
            self.reference_result["output_complex_field_sqrt_J_m"], complex
        )
        overlap = abs(np.vdot(field, target)) ** 2 / max(
            float(np.vdot(field, field).real * np.vdot(target, target).real), 1e-30
        )
        return dict(
            coherent_overlap=float(overlap),
            useful_target_mode_energy_J=float(overlap * result["output_energy_J"]),
            physical_output_energy_J=float(result["output_energy_J"]),
        )
