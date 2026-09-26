"""Measurement-level checks for the proposal-style phase feedback path."""

import numpy as np

from hoyag.propagation import Grid2D, angular_spectrum_propagate
from ybluag.camera_dataset import CameraSettings
from ybluag.phase_diagnostics import compensation_residual, piston_removed_residual
from ybyag_control.controller import (
    ControllerConfig, Observation, PhaseControlGeometry, run_controller,
)
from ybyag_control.interferometry import (
    calibrated_interferometer, capture_interferograms, reference_amplitude,
)
from ybyag_control.adapter import EpisodeConfig, SimulationPlant


def test_four_phase_camera_recovers_a_known_field():
    grid = Grid2D.square(64, .012)
    x, y = grid.mesh
    field = np.exp(-(x*x+y*y)/(.0009**2)) * np.exp(
        1j * (.3*x/.001 + .2*(x*x-y*y)/1e-6)
    )
    lo = reference_amplitude(field, grid.x, grid.y, .0006)
    base = CameraSettings(width=64, height=64, object_fov_width_mm=12,
                          pulses_per_exposure=100, exposure_s=.01)
    camera, setup = calibrated_interferometer(
        field, lo, grid.x, grid.y, 1030e-9, base, {}, 123, noisy=False,
    )
    frames, recovered, saturation = capture_interferograms(
        field, lo, grid.x, grid.y, 1030e-9, camera, setup, 124, noisy=False,
    )
    assert frames.shape == (4, 64, 64)
    assert saturation == 0
    assert piston_removed_residual(recovered, field)[1] < .01
    bright = abs(field) > .1 * abs(field).max()
    relative_amplitude_error = np.linalg.norm(
        abs(recovered[bright]) - abs(field[bright])
    ) / np.linalg.norm(abs(field[bright]))
    assert relative_amplitude_error < .01


def test_interferometric_feedback_reduces_measured_phase_error():
    grid = Grid2D.square(64, .012)
    x, y = grid.mesh
    source = np.exp(-(x*x+y*y)/(.0006**2))
    aberration = .35*(x*x-y*y)/(.0006**2)
    distance = .25
    geometry = PhaseControlGeometry(grid, 1030e-9, distance,
                                    source, np.zeros(grid.shape))
    target_field = angular_spectrum_propagate(source, grid, 1030e-9, distance)

    def observation(command, field):
        intensity = abs(field)**2
        frame = np.rint(32 + 10000*intensity/intensity.max()).astype(np.uint16)
        return Observation(
            np.stack((frame, frame)), float(intensity.sum()),
            np.zeros(5), np.ones(5, bool), command.copy(), 0., 0.,
            phase_field=field,
        )

    reference = observation(np.zeros(grid.shape), target_field)

    class OpticalPlant:
        def observe(self, command):
            field = angular_spectrum_propagate(
                source*np.exp(1j*(aberration+command)), grid, 1030e-9, distance,
            )
            return observation(command, field)

    config = ControllerConfig(
        method="interferometric", iterations=10, evaluation_limit=20,
        phase_gain_rad=.12, target_confirmations=20,
    )
    result = run_controller(
        OpticalPlant(), reference, reference.measured_energy_J, config,
        phase_geometry=geometry,
    )
    assert result["history"][-1]["phase_rms_rad"] < .07
    assert result["history"][-1]["phase_rms_rad"] < result["history"][0]["phase_rms_rad"]


def test_camera_regression_is_rejected_even_if_phase_can_improve():
    grid = Grid2D.square(64, .012)
    x, y = grid.mesh
    source = np.exp(-(x*x+y*y)/(.0006**2))
    aberration = .35*(x*x-y*y)/(.0006**2)
    geometry = PhaseControlGeometry(grid, 1030e-9, .25,
                                    source, np.zeros(grid.shape))
    target = angular_spectrum_propagate(source, grid, 1030e-9, .25)
    rows, columns = np.indices(grid.shape)

    def observation(command, field):
        # In this plant a phase update improves the complex field but shifts
        # the diagnostic image, so the camera guard must reject it.
        shift = 50 * np.sqrt(np.mean(command**2))
        image = np.exp(-((columns-31.5-shift)**2 + (rows-31.5)**2)/50)
        frame = np.rint(32 + 10000*image).astype(np.uint16)
        return Observation(
            np.stack((frame, frame)), float(np.sum(abs(field)**2)),
            np.zeros(5), np.ones(5, bool), command.copy(), 0., 0.,
            phase_field=field,
        )

    class ConflictingPlant:
        def observe(self, command):
            field = angular_spectrum_propagate(
                source*np.exp(1j*(aberration+command)), grid, 1030e-9, .25,
            )
            return observation(command, field)

    reference = observation(np.zeros(grid.shape), target)
    config = ControllerConfig(
        method="interferometric", iterations=4, evaluation_limit=30,
        target_confirmations=20,
    )
    result = run_controller(
        ConflictingPlant(), reference, reference.measured_energy_J,
        config, phase_geometry=geometry,
    )
    assert result["status"] == "no_measured_improvement"
    assert all(row["camera_loss"] <= config.improvement_tolerance
               for row in result["history"])
    assert any(row["update_status"] == "rejected_restore"
               and row["rejected_trial_camera_loss"] > config.improvement_tolerance
               for row in result["history"])
    np.testing.assert_allclose(result["final_coefficients_rad"], 0)


def test_hybrid_recovers_camera_shape_after_phase_update_stalls():
    grid = Grid2D.square(64, .012)
    x, y = grid.mesh
    source = np.exp(-(x*x+y*y)/(.0006**2))
    tip = x / .0006
    weight = abs(source)**2
    geometry = PhaseControlGeometry(
        grid, 1030e-9, .25, source, np.zeros(grid.shape), tip[None],
    )
    target_field = angular_spectrum_propagate(source, grid, 1030e-9, .25)
    rows, columns = np.indices(grid.shape)

    def camera_frame(shift):
        image = np.exp(-((columns - 31.5 - shift)**2 + (rows - 31.5)**2) / 50)
        return np.rint(32 + 10000 * image).astype(np.uint16)

    reference_frame = camera_frame(0)
    reference = Observation(
        np.stack((reference_frame, reference_frame)), 1.0,
        np.zeros(5), np.ones(5, bool), np.zeros(grid.shape), 0., 0.,
        phase_field=target_field,
    )

    class ShapePlant:
        def observe(self, command):
            tip_coefficient = np.sum(weight * command * tip) / np.sum(weight * tip**2)
            frame = camera_frame(5 * (1 - tip_coefficient))
            # This separate fixed phase error makes the pure phase-gradient
            # path stall; only camera measurements reveal the tip correction.
            field = target_field * np.exp(1j * .4 * (x*x-y*y) / .0006**2)
            return Observation(
                np.stack((frame, frame)), 1.0,
                np.zeros(5), np.ones(5, bool), command.copy(), 0., 0.,
                phase_field=field,
            )

    config = ControllerConfig(
        method="hybrid", mode_count=1, iterations=8,
        evaluation_limit=80, target_confirmations=20,
    )
    result = run_controller(
        ShapePlant(), reference, 1.0, config, phase_geometry=geometry,
    )
    recovered = [row for row in result["history"]
                 if row["update_status"] == "camera_shape_recovery"]
    assert recovered
    assert recovered[0]["camera_loss"] < result["history"][0]["camera_loss"]


def test_truth_compensation_excludes_intentional_mask_and_tracks_delivered_phase():
    plant = SimulationPlant(EpisodeConfig(grid_n=64, enable_external_optics=False), 1)
    x, y = plant.grid.mesh
    plant.external_phase = .2 * x / .0006
    applied_correction = .1 * y / .0006
    solved = {
        "effective_signal_traversals": 2,
        "thermal_timeline": {"final_roundtrip_opd_m": np.zeros(plant.grid.shape)},
        "static_cold_phase_rad": np.zeros(plant.grid.shape),
        "slm_actual_phase_rad": plant.target_phase + applied_correction,
    }
    ideal, delivered = plant.phase_compensation_truth(solved)
    pupil = np.isfinite(ideal)
    assert pupil.any()
    np.testing.assert_allclose(ideal[pupil], -plant.external_phase[pupil], atol=1e-10)
    np.testing.assert_allclose(delivered[pupil], applied_correction[pupil], atol=1e-10)


def test_compensation_rms_weights_the_illuminated_seed_and_ignores_piston():
    ideal = np.zeros((3, 3))
    actual = np.full((3, 3), .4)
    actual[0, 0] += 1.0
    illumination = np.ones((3, 3))
    illumination[0, 0] = .0001
    residual, rms = compensation_residual(actual, ideal, illumination)
    assert np.isnan(residual[0, 0])
    assert rms < 1e-12
