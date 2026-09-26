from dataclasses import replace

import numpy as np
import pytest

from hoyag.propagation import Grid2D
from hoyag.thermal import DiskThermalMesh
from ybluag.gallery import (
    YbGallerySettings,
    _assembly_configuration,
    _pulsed_thermal_timeline,
)
from ybyag.model import YbYAGMaterial
from ybyag_control.controller import (
    ControllerConfig,
    Observation,
    correction_basis,
    measurement_metrics,
    run_controller,
)
from ybyag_control.adapter import EpisodeConfig, SimulationPlant, diagnostic_fields
from ybyag_dataset.distortions.camera import capture
from ybyag_dataset.generator import absolute_oracle_command
from ybyag_dataset.distortions.sensors import SensorSession


class CameraOnlyPlant:
    """No simulator truth methods or attributes are exposed to the policy."""

    def __init__(self, *, dim_energy=False, saturation_once=False):
        self.calls = 0
        self.dim_energy = dim_energy
        self.saturation_once = saturation_once

    def observe(self, coefficients_rad):
        self.calls += 1
        value = float(coefficients_rad[0])
        axis = np.linspace(-1, 1, 16)
        x, y = np.meshgrid(axis, axis)
        image = np.exp(-8 * ((x - (value - 0.3)) ** 2 + y * y))
        frame = np.rint(32 + 2000 * image).astype(np.uint16)
        energy = 1e-8 * (0.4 if self.dim_energy and abs(value) > 0.05 else 1.0)
        return Observation(
            np.stack((frame, frame)),
            energy,
            np.full(5, 293.0),
            np.ones(5, bool),
            np.asarray(coefficients_rad).copy(),
            self.calls * 0.04,
            0.1 if self.saturation_once and self.calls == 1 else 0.0,
        )


def _reference():
    plant = CameraOnlyPlant()
    return plant.observe(np.array([0.3]))


def test_measurement_only_update_and_energy_floor():
    cfg = ControllerConfig(
        mode_count=1,
        perturbation_rad=0.08,
        max_update_rad=0.4,
        regularization=0.001,
        iterations=2,
        evaluation_limit=25,
        target_loss=0,
        improvement_tolerance=0,
        noise_acceptance_sigma=0,
    )
    good = run_controller(CameraOnlyPlant(), _reference(), 1e-8, cfg)
    assert good["final_coefficients_rad"][0] > 0
    assert good["history"][-1]["camera_loss"] < good["history"][0]["camera_loss"]
    dim = run_controller(CameraOnlyPlant(dim_energy=True), _reference(), 1e-8, cfg)
    assert abs(dim["final_coefficients_rad"][0]) < 0.05


def test_saturation_reacquisition_and_budget():
    cfg = ControllerConfig(
        mode_count=1, iterations=1, evaluation_limit=15, target_loss=1
    )
    result = run_controller(
        CameraOnlyPlant(saturation_once=True), _reference(), 1e-8, cfg
    )
    assert result["evaluations"] == 2


@pytest.mark.parametrize("method", ["spgd", "response_matrix"])
def test_persistent_probe_saturation_stops_with_recorded_status(method):
    class SaturatingProbe(CameraOnlyPlant):
        def observe(self, coefficients_rad):
            observation = super().observe(coefficients_rad)
            if self.calls >= 2:
                return replace(observation, saturated_fraction=0.1, valid=False)
            return observation

    config = ControllerConfig(
        method=method, mode_count=1, iterations=1,
        evaluation_limit=15, target_loss=0.0,
    )
    outcome = run_controller(SaturatingProbe(), _reference(), 1e-8, config)
    assert outcome["status"] == "camera_observation_invalid"
    assert outcome["evaluations"] == 4
    assert len(outcome["history"]) == 1
    invalid = replace(_reference(), saturated_fraction=0.1, valid=False)
    assert measurement_metrics(invalid, _reference(), 1e-8, config)["camera_loss"] is None


def test_absolute_command_preserves_vortex_and_basis_has_no_piston():
    axis = (np.arange(64) - 31.5) * 1e-4
    x, y = np.meshgrid(axis, axis)
    vortex = np.arctan2(y, x)
    correction = 0.2 * x / x.max()
    command = absolute_oracle_command(vortex, correction)
    np.testing.assert_allclose(np.exp(1j * command), np.exp(1j * (vortex + correction)))
    source = np.exp(-(x * x + y * y) / (0.6e-3) ** 2)
    np.testing.assert_allclose(
        abs(source * np.exp(1j * command)) ** 2, abs(source) ** 2
    )
    basis = correction_basis(axis, axis, 0.6e-3, 14)
    weight = np.exp(-2 * (x * x + y * y) / (0.6e-3) ** 2)
    for mode in basis:
        assert abs(np.average(mode, weights=weight)) < 1e-9


def test_asymmetric_diagnostic_increases_handedness_sensitivity():
    grid = Grid2D.square(128, 6e-3)
    x, y = grid.mesh
    amplitude = np.exp(-(x * x + y * y) / (0.6e-3) ** 2)
    theta = np.arctan2(y, x)
    plus = amplitude * np.exp(1j * theta)
    minus = amplitude * np.exp(-1j * theta)
    focus_p, plain_p = diagnostic_fields(plus, grid, 1030e-9, 0.6e-3, 0.1, 0)
    focus_m, plain_m = diagnostic_fields(minus, grid, 1030e-9, 0.6e-3, 0.1, 0)
    assert np.array_equal(abs(focus_p) ** 2, abs(focus_m) ** 2)
    _, diverse_p = diagnostic_fields(plus, grid, 1030e-9, 0.6e-3, 0.1, 0.25)
    _, diverse_m = diagnostic_fields(minus, grid, 1030e-9, 0.6e-3, 0.1, 0.25)
    plain = np.sum(abs(abs(plain_p) ** 2 - abs(plain_m) ** 2))
    diverse = np.sum(abs(abs(diverse_p) ** 2 - abs(diverse_m) ** 2))
    assert diverse > plain * 2


def test_in_situ_requires_feasible_timed_exposure():
    with pytest.raises(ValueError, match="exposure"):
        EpisodeConfig(pulses_per_exposure=200, exposure_s=0.01)
    with pytest.raises(ValueError, match="SLM delay"):
        EpisodeConfig(slm_delay_s=-0.01)
    with pytest.raises(ValueError, match="finite and positive"):
        EpisodeConfig(pump_W=float("nan"))


def test_noiseless_probes_read_the_solved_temperature_field():
    episode = EpisodeConfig(
        grid_n=32, camera_width=128, camera_height=72, enable_camera_noise=False
    )
    plant = SimulationPlant(episode, mode_count=1)
    disk = np.full(plant.sensors.disk_mesh.shape, 294.2)
    plate = np.full(plant.sensors.plate_mesh.shape, 293.7)
    result = {
        "output_complex_field_sqrt_J_m": np.ones(plant.grid.shape, complex),
        "output_energy_J": 1e-8,
        "thermal_timeline": {
            "requested_disk_temperature_K": disk,
            "requested_plate_temperature_K": plate,
        },
    }
    observation = plant._capture(result, np.zeros(1), noisy=False, clock_s=0.1, seed=7)
    np.testing.assert_allclose(observation.probe_temperature_K[:3], 294.2)
    np.testing.assert_allclose(observation.probe_temperature_K[3:], 293.7)


def test_probe_calibration_drift_is_stateful_and_time_ordered():
    episode = EpisodeConfig(grid_n=32, camera_width=128, camera_height=72)
    plant = SimulationPlant(episode, mode_count=1)
    ranges = dict(
        plant.ranges,
        probe_noise_K=[0.0, 0.0],
        probe_bias_K=0.0,
        probe_response_time_s=[0.0, 0.0],
        probe_dropout_probability=0.0,
        probe_drift_K_per_sqrt_s=0.2,
    )
    sensor = SensorSession(
        plant.sensors.disk_mesh, plant.sensors.plate_mesh, ranges, seed=18
    )
    disk = np.full(sensor.disk_mesh.shape, 294.0)
    plate = np.full(sensor.plate_mesh.shape, 294.0)
    first = sensor.sample(disk, plate, 0.0)
    later = sensor.sample(disk, plate, 1.0)
    repeat = sensor.sample(disk, plate, 1.0)
    np.testing.assert_allclose(first["measured_temperature_K"], 294.0)
    assert np.any(np.abs(later["drift_K"]) > 0)
    np.testing.assert_allclose(repeat["drift_K"], later["drift_K"])
    np.testing.assert_allclose(
        repeat["measured_temperature_K"], later["measured_temperature_K"]
    )
    with pytest.raises(ValueError, match="backwards"):
        sensor.sample(disk, plate, 0.5)


def test_in_situ_delay_holds_previous_command_and_never_rewinds(monkeypatch):
    episode = EpisodeConfig(
        mode="in_situ",
        grid_n=32,
        camera_width=128,
        camera_height=72,
        pump_W=0.01,
        operation_duration_s=0.01,
        enable_camera_noise=False,
    )
    plant = SimulationPlant(episode, mode_count=1)
    calls = []

    def solve(coefficients, ideal=False, duration_s=None):
        calls.append((float(coefficients[0]), duration_s))
        return {
            "thermal_timeline": {
                "requested_disk_temperature_K": np.full((1, 4, 4), 293.2),
                "requested_plate_temperature_K": np.full((1, 4, 4), 293.1),
            }
        }

    def capture(_result, coefficients, *, noisy, clock_s, seed):
        return Observation(
            np.full((2, 72, 128), 200, np.uint16),
            1e-8,
            np.full(5, 293.2),
            np.ones(5, bool),
            coefficients.copy(),
            clock_s,
            0.0,
        )

    monkeypatch.setattr(plant, "_solve", solve)
    monkeypatch.setattr(plant, "_capture", capture)
    plant.observe(np.array([0.2]))
    plant.observe(np.array([0.3]))
    assert [value for value, _ in calls] == [0, 0.2, 0.2, 0.3]
    assert all(duration == pytest.approx(0.01) for _, duration in calls[:1])
    assert plant.command_log[0]["delivered_time_s"] == pytest.approx(0.01)
    assert plant.command_log[0]["exposure_start_s"] == pytest.approx(0.03)
    assert plant.command_log[1]["exposure_end_s"] == pytest.approx(0.08)
    assert (
        plant.command_log[1]["requested_time_s"]
        >= plant.command_log[0]["exposure_end_s"]
    )


def test_in_situ_warmup_is_once_and_camera_calibrates_both_arms(monkeypatch):
    episode = EpisodeConfig(
        mode="in_situ",
        operation_duration_s=30.0,
        control_period_s=0.1,
        grid_n=32,
        camera_width=128,
        camera_height=72,
        enable_camera_noise=False,
    )
    plant = SimulationPlant(episode, mode_count=1)
    x, y = plant.grid.mesh
    field = np.sqrt(50.0) * np.exp(-(x*x + y*y) / (0.6e-3)**2)
    calls = []

    def solve(_coefficients, ideal=False, duration_s=None):
        calls.append((ideal, duration_s))
        return {
            "output_complex_field_sqrt_J_m": field.astype(complex),
            "output_energy_J": 1e-8,
            "thermal_timeline": {
                "requested_disk_temperature_K": np.full(plant.sensors.disk_mesh.shape, 293.2),
                "requested_plate_temperature_K": np.full(plant.sensors.plate_mesh.shape, 293.1),
            },
        }

    monkeypatch.setattr(plant, "_solve", solve)
    reference, _ = plant.reference()
    assert calls == [(True, None), (False, 30.0)]
    assert plant.current_time_s == pytest.approx(30.0)
    assert reference.valid
    assert plant.camera.optical_throughput < 1.0
    for arm, arm_field in enumerate(
        diagnostic_fields(field, plant.grid, 1030e-9, plant.settings.waist_m,
                          episode.diagnostic_distance_m,
                          episode.diagnostic_astigmatism_waves)
    ):
        _, _, info = capture(abs(arm_field)**2, plant.grid.x, plant.grid.y,
                             1030e-9, plant.camera, plant.camera_setup,
                             episode.seed + 100 + arm, enabled=False)
        assert info["expected_electron_peak"] <= 0.151 * plant.camera.full_well_e
    plant.observe(np.zeros(1))
    assert plant.current_time_s == pytest.approx(30.1)
    assert calls[-2:] == [(False, 0.01), (False, pytest.approx(0.09))]


def test_controller_reference_uses_cold_phase_from_same_pumped_solution(monkeypatch):
    episode = EpisodeConfig(
        grid_n=32, camera_width=128, camera_height=72,
        enable_camera_noise=False,
    )
    plant = SimulationPlant(episode, mode_count=1)
    x, y = plant.grid.mesh
    cold = np.exp(-(x*x + y*y) / (0.6e-3)**2).astype(complex)
    hot = cold * np.exp(1j * 0.4 * x / (0.6e-3))
    result = {
        "output_complex_field_sqrt_J_m": hot,
        "cold_output_complex_field_sqrt_J_m": cold,
        "output_energy_J": float(np.sum(abs(hot)**2) * plant.grid.dx * plant.grid.dy),
        "thermal_timeline": {
            "requested_disk_temperature_K": np.full(plant.sensors.disk_mesh.shape, 293.4),
            "requested_plate_temperature_K": np.full(plant.sensors.plate_mesh.shape, 293.2),
        },
    }
    monkeypatch.setattr(plant, "_solve", lambda *_args, **_kwargs: result)
    reference, _ = plant.reference()
    np.testing.assert_allclose(
        plant.reference_result["output_complex_field_sqrt_J_m"], cold
    )
    np.testing.assert_allclose(plant.reference_result["output_phase"], np.angle(cold))
    assert reference.valid


def test_in_situ_budget_stops_before_partial_thermal_advance():
    episode = EpisodeConfig(
        mode="in_situ", grid_n=32, camera_width=128, camera_height=72
    )
    plant = SimulationPlant(episode, mode_count=1)
    plant.full_solve_limit = 1  # A delayed observation needs two solves.
    with pytest.raises(StopIteration, match="evaluation budget"):
        plant.observe(np.zeros(1))
    assert plant.full_solves == 0
    assert plant.current_time_s == 0
    assert plant.initial_thermal is None
    assert plant.command_log == []


def test_camera_noise_changes_per_fit_measurement_but_replays_from_seed():
    episode = EpisodeConfig(
        grid_n=32,
        camera_width=128,
        camera_height=72,
        camera_gain_jitter_fraction=0.03,
        camera_gain_random_walk_per_sqrt_s=0.01,
        camera_read_noise_e=6.0,
    )

    def samples():
        plant = SimulationPlant(episode, mode_count=1)
        plant.sensors.sample = lambda *_: {"measured_temperature_K": np.full(5, 293.15)}
        x, y = plant.grid.mesh
        field = np.sqrt(10.0) * np.exp(-(x * x + y * y) / (0.6e-3) ** 2)
        result = {
            "output_complex_field_sqrt_J_m": field.astype(complex),
            "output_energy_J": 1e-8,
            "thermal_timeline": {
                "requested_disk_temperature_K": np.full((1, 4, 8), 293.15),
                "requested_plate_temperature_K": np.full((1, 4, 8), 293.15),
            },
        }
        observations = []
        for count in (1, 2):
            obs = plant._capture(
                result, np.zeros(1), noisy=True, clock_s=0.04 * count, seed=1000 * count
            )
            observations.append(
                (
                    obs.camera_adu.copy(),
                    tuple(plant.latest_noise_state["camera_gain_factors"]),
                )
            )
        return observations

    first = samples()
    second = samples()
    assert not np.array_equal(first[0][0], first[1][0])
    assert first[0][1] != first[1][1]
    for (frame_a, gain_a), (frame_b, gain_b) in zip(first, second):
        np.testing.assert_array_equal(frame_a, frame_b)
        assert gain_a == gain_b


def test_changing_coolant_and_pump_enter_each_physical_solve(monkeypatch):
    with pytest.raises(ValueError, match="in_situ"):
        EpisodeConfig(enable_thermal_variation=True)
    episode = EpisodeConfig(
        mode="in_situ",
        enable_thermal_variation=True,
        grid_n=32,
        camera_width=128,
        camera_height=72,
        thermal_nphi=4,
        enable_material=False,
        enable_slm_error=False,
    )
    plant = SimulationPlant(episode, mode_count=1)
    operating = plant._sample_operating_point(0.0)
    plant.latest_operating_point = operating
    received = []

    def forward(_material, settings, _target, *args, **kwargs):
        received.append((settings, kwargs["dataset_physical"]))
        return {
            "thermal_timeline": {"requested_material_range_valid": True},
            "thermal_feedback_applied": True,
        }

    monkeypatch.setattr("ybyag_control.adapter.simulate_pulsed_seed", forward)
    plant._solve(np.zeros(1))
    settings, physical = received[0]
    assert settings.pump_power_W == operating["pump_power_W"]
    assert settings.pump_radius_m == operating["pump_radius_m"]
    assert physical["coolant_temperature_K"] == operating["coolant_temperature_K"]
    assert physical["pump_center_m"] == operating["pump_center_m"]
    assert 293.15 <= physical["coolant_temperature_K"] <= 298.15
    later = plant._sample_operating_point(0.1)
    assert later != operating


def test_nominal_coolant_setting_reaches_snapshot_and_reference_solves(monkeypatch):
    episode = EpisodeConfig(
        mode="snapshot",
        coolant_setpoint_C=21.0,
        grid_n=32,
        camera_width=128,
        camera_height=72,
    )
    plant = SimulationPlant(episode, mode_count=1)
    received = []

    def forward(_material, _settings, _target, *args, **kwargs):
        received.append(kwargs["dataset_physical"]["coolant_temperature_K"])
        return {
            "thermal_timeline": {"requested_material_range_valid": True},
            "thermal_feedback_applied": True,
        }

    monkeypatch.setattr("ybyag_control.adapter.simulate_pulsed_seed", forward)
    plant._solve(np.zeros(1), ideal=True)
    plant._solve(np.zeros(1))
    assert received == pytest.approx([294.15, 294.15])


def test_requested_only_timeline_matches_full_at_observation_time():
    settings = YbGallerySettings(
        thickness_m=100e-6,
        thermal_nr=4,
        thermal_nphi=4,
        thermal_nz=1,
        grid_n=32,
        field_size_m=0.012,
        assembly_property_model="yag_rt_proxy",
    )
    mesh = DiskThermalMesh.disk(
        nr=4,
        nphi=4,
        nz=1,
        radius_m=settings.disk_radius_m,
        thickness_m=settings.thickness_m,
    )
    grid = Grid2D.square(32, 0.012)
    configuration = _assembly_configuration(YbYAGMaterial(yb_at_percent=10), settings)
    heat = np.full(mesh.shape, 1e7)
    args = (mesh, heat, grid, configuration, 0.1)
    full = _pulsed_thermal_timeline(*args, include_stabilization=True)
    short = _pulsed_thermal_timeline(*args, include_stabilization=False)
    assert full["stabilization_status"] == "evaluated"
    assert short["stabilization_status"] == "not_requested"
    assert short["steady_disk_max_C"] is None
    assert short["time_s"][-1] == pytest.approx(0.1)
    for key in (
        "requested_disk_temperature_K",
        "requested_plate_temperature_K",
        "final_roundtrip_opd_m",
        "final_front_displacement_nm",
        "final_rear_displacement_nm",
        "final_disk_displacement_m",
        "final_plate_displacement_m",
    ):
        np.testing.assert_allclose(short[key], full[key], rtol=0, atol=1e-12)


def test_spgd_uses_many_measured_updates_and_respects_energy_floor():
    config = ControllerConfig(
        method="spgd",
        mode_count=1,
        iterations=40,
        evaluation_limit=130,
        target_loss=0,
        spgd_gain=0.25,
        restore_best_at_end=False,
    )
    good = run_controller(CameraOnlyPlant(), _reference(), 1e-8, config)
    assert len(good["history"]) == 41
    assert good["evaluations"] == 121
    assert good["history"][-1]["camera_loss"] < good["history"][0]["camera_loss"]
    assert abs(good["final_coefficients_rad"][0] - 0.3) < 0.03
    dim = run_controller(
        CameraOnlyPlant(dim_energy=True),
        _reference(),
        1e-8,
        ControllerConfig(
            method="spgd",
            mode_count=1,
            iterations=5,
            evaluation_limit=25,
            target_loss=0,
            spgd_gain=1.0,
        ),
    )
    assert np.array_equal(dim["final_coefficients_rad"], np.zeros(1))
    assert all(row["update_status"] == "energy_rollback" for row in dim["history"][1:])


def test_spgd_rechecks_saved_command_with_new_observations():
    restored = run_controller(
        CameraOnlyPlant(),
        _reference(),
        1e-8,
        ControllerConfig(
            method="spgd",
            mode_count=1,
            iterations=5,
            evaluation_limit=25,
            target_loss=0,
            spgd_gain=0.1,
        ),
    )
    assert restored["best_recheck_status"] == "restored"
    assert restored["history"][-1]["update_status"] == "best_restore"
    assert restored["evaluations"] == 18  # initial, five updates, two rechecks
    assert (
        restored["history"][-1]["camera_loss"] < restored["history"][-2]["camera_loss"]
    )

    kept = run_controller(
        CameraOnlyPlant(),
        _reference(),
        1e-8,
        ControllerConfig(
            method="spgd",
            mode_count=1,
            iterations=40,
            evaluation_limit=130,
            target_loss=0,
            spgd_gain=0.25,
        ),
    )
    assert kept["best_recheck_status"] == "kept_current"
    assert kept["history"][-1]["update_status"] == "best_rejected"
    assert kept["evaluations"] == 124  # rejected trial needs a timed return


def test_in_situ_measurement_period_is_simulated_time(monkeypatch):
    episode = EpisodeConfig(
        mode="in_situ",
        control_period_s=0.1,
        grid_n=32,
        camera_width=128,
        camera_height=72,
        pump_W=0.01,
    )
    plant = SimulationPlant(episode, mode_count=1)
    durations = []

    def solve(_coefficients, ideal=False, duration_s=None):
        durations.append(duration_s)
        return {
            "thermal_timeline": {
                "requested_disk_temperature_K": np.full((1, 4, 8), 293.2),
                "requested_plate_temperature_K": np.full((1, 4, 8), 293.1),
            }
        }

    monkeypatch.setattr(plant, "_solve", solve)
    monkeypatch.setattr(
        plant,
        "_capture",
        lambda _result, coefficients, *, noisy, clock_s, seed: Observation(
            np.full((2, 72, 128), 200, np.uint16),
            1e-8,
            np.full(5, 293.2),
            np.ones(5, bool),
            coefficients.copy(),
            clock_s,
            0.0,
        ),
    )
    first = plant.observe(np.zeros(1))
    second = plant.observe(np.zeros(1))
    assert first.time_s == pytest.approx(0.1)
    assert second.time_s == pytest.approx(0.2)
    assert durations == pytest.approx([0.01, 0.09, 0.01, 0.09])
