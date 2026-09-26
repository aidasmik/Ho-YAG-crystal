"""Measured-field, response-matrix and sequential SPGD phase controllers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol
import numpy as np
from scipy.ndimage import gaussian_filter

from hoyag.propagation import Grid2D, angular_spectrum_propagate
from ybluag.phase_diagnostics import piston_removed_residual


@dataclass(frozen=True)
class Observation:
    camera_adu: np.ndarray  # planes, y, x; calibrated before scoring
    measured_energy_J: float
    probe_temperature_K: np.ndarray
    probe_valid: np.ndarray
    delivered_coefficients_rad: np.ndarray
    time_s: float
    saturated_fraction: float
    valid: bool = True
    phase_field: np.ndarray | None = None
    interferograms_adu: np.ndarray | None = None


class InvalidObservationError(ValueError):
    """A camera command remained unusable after repeat exposures."""


@dataclass(frozen=True)
class ControllerConfig:
    method: str = "response_matrix"
    mode_count: int = 14
    perturbation_rad: float = 0.08
    max_update_rad: float = 0.4
    regularization: float = 0.03
    iterations: int = 3
    evaluation_limit: int = 100
    improvement_tolerance: float = 0.002
    target_loss: float = 0.02
    energy_weight: float = 0.4
    minimum_energy_fraction: float = 0.65
    saturation_limit: float = 0.002
    noise_acceptance_sigma: float = 2.0
    refresh_every: int = 1
    spgd_gain: float = 0.25
    spgd_momentum: float = 0.5
    spgd_decay: float = 0.25
    spgd_seed: int = 413
    coefficient_limit_rad: float = 2.0
    target_confirmations: int = 5
    restore_best_at_end: bool = True
    phase_gain_rad: float = 0.12
    phase_smoothing_pixels: float = 0.7
    phase_target_rms_rad: float = 0.5

    def __post_init__(self):
        if (
            self.method not in ("response_matrix", "spgd", "interferometric")
            or not 1 <= self.mode_count <= 14
            or not 1 <= self.iterations <= 200
            or not 1 <= self.evaluation_limit <= 500
            or not 0 < self.perturbation_rad <= 1
            or not 0 < self.max_update_rad <= 2
            or not 0 <= self.regularization <= 10
            or not 0 <= self.improvement_tolerance <= 1
            or not 0 <= self.target_loss <= 1
            or not 0 <= self.energy_weight <= 10
            or not 0 < self.minimum_energy_fraction <= 1
            or not 0 <= self.saturation_limit <= 1
            or not 0 <= self.noise_acceptance_sigma <= 10
            or not 1 <= self.refresh_every <= 10
            or not np.isfinite(self.spgd_gain)
            or not 0 < self.spgd_gain <= 5
            or not np.isfinite(self.spgd_momentum)
            or not 0 <= self.spgd_momentum < 1
            or not np.isfinite(self.spgd_decay)
            or not 0 <= self.spgd_decay <= 2
            or not isinstance(self.spgd_seed, int)
            or self.spgd_seed < 0
            or not np.isfinite(self.coefficient_limit_rad)
            or not 0 < self.coefficient_limit_rad <= 4
            or not 1 <= self.target_confirmations <= 20
            or not isinstance(self.restore_best_at_end, bool)
            or not np.isfinite(self.phase_gain_rad)
            or not 0 < self.phase_gain_rad <= 1
            or not np.isfinite(self.phase_smoothing_pixels)
            or not 0 <= self.phase_smoothing_pixels <= 4
            or not np.isfinite(self.phase_target_rms_rad)
            or not 0 < self.phase_target_rms_rad <= np.pi
        ):
            raise ValueError("invalid controller configuration")


class MeasurementPlant(Protocol):
    def observe(self, coefficients_rad: np.ndarray) -> Observation: ...


class ControllerPolicy(Protocol):
    """Future NN policies use the same observation/action contract."""

    def propose(
        self, observation: Observation, history: tuple[dict, ...]
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class PhaseControlGeometry:
    """Nominal free-space SLM-to-disk path; contains no hidden crystal truth.

    The current control fixture observes the extraction plane immediately at
    the disk, so this is also the passive part of the SLM-to-camera adjoint.
    It does not differentiate gain, thermal lensing or disk deformation.
    """

    grid: Grid2D
    wavelength_m: float
    slm_to_disk_m: float
    source_field: np.ndarray
    desired_slm_phase_rad: np.ndarray

    def __post_init__(self):
        if (
            np.shape(self.source_field) != self.grid.shape
            or np.shape(self.desired_slm_phase_rad) != self.grid.shape
            or not np.all(np.isfinite(self.source_field))
            or not np.all(np.isfinite(self.desired_slm_phase_rad))
            or not np.isfinite(self.wavelength_m)
            or self.wavelength_m <= 0
            or not np.isfinite(self.slm_to_disk_m)
            or self.slm_to_disk_m <= 0
        ):
            raise ValueError("invalid public phase-control geometry")


def correction_basis(x_m, y_m, waist_m, mode_count=14):
    """Smooth, seed-weighted, piston-free phase basis in radians per coefficient."""
    x, y = np.meshgrid(np.asarray(x_m), np.asarray(y_m))
    u, v = x / (2 * waist_m), y / (2 * waist_m)
    r2 = u * u + v * v
    z = u + 1j * v
    raw = [
        u,
        v,
        2 * r2 - 1,
        u * u - v * v,
        2 * u * v,
        (3 * r2 - 2) * u,
        (3 * r2 - 2) * v,
        np.real(z**3),
        np.imag(z**3),
        6 * r2 * r2 - 6 * r2 + 1,
        (4 * r2 - 3) * (u * u - v * v),
        (4 * r2 - 3) * 2 * u * v,
        np.real(z**4),
        np.imag(z**4),
    ]
    weight = np.exp(-2 * (x * x + y * y) / waist_m**2)
    weight /= weight.sum()
    basis = []
    for term in raw[:mode_count]:
        centered = term - np.sum(weight * term)
        rms = np.sqrt(np.sum(weight * centered**2))
        basis.append(centered / max(rms, 1e-12))
    return np.stack(basis)


def _calibrated_vector(obs: Observation, black_level_adu: int, reference_shape):
    frame = np.asarray(obs.camera_adu, float)
    if (
        frame.shape != reference_shape
        or frame.ndim != 3
        or not np.all(np.isfinite(frame))
    ):
        raise ValueError("invalid diagnostic frame geometry")
    signal = np.maximum(frame - black_level_adu, 0)
    # Fixed 8x8 binning reduces noisy CCD pixels; it does not create optical
    # resolution or shift/recenter the observation.
    h, w = frame.shape[-2:]
    bh, bw = max(1, h // 8), max(1, w // 8)
    signal = (
        signal[:, : bh * 8, : bw * 8]
        .reshape(frame.shape[0], bh, 8, bw, 8)
        .sum(axis=(2, 4))
    )
    totals = signal.sum(axis=(1, 2))
    if np.any(totals <= 0):
        raise ValueError("dark diagnostic frame")
    return np.sqrt(signal / np.maximum(totals[:, None, None], 1e-30)).ravel()


def _score(obs, reference, target_energy_J, config, black_level):
    if (
        not obs.valid
        or not np.isfinite(obs.measured_energy_J)
        or obs.measured_energy_J <= 0
        or obs.saturated_fraction > config.saturation_limit
    ):
        raise ValueError("invalid or saturated observation")
    vector = _calibrated_vector(obs, black_level, reference.shape)
    energy_fraction = obs.measured_energy_J / target_energy_J
    shape = vector - reference.vector
    # A smaller beam may improve normalized shape. Keep a separate measured
    # energy penalty and a hard floor for useful target-mode energy.
    energy_penalty = config.energy_weight * max(0, 1 - energy_fraction)
    loss = float(np.mean(shape**2) * len(shape) / reference.planes + energy_penalty**2)
    if energy_fraction < config.minimum_energy_fraction:
        loss += (config.minimum_energy_fraction - energy_fraction) ** 2 * 10
    return vector, float(np.sqrt(loss)), energy_fraction


@dataclass(frozen=True)
class _Reference:
    vector: np.ndarray
    shape: tuple
    planes: int


def run_controller(
    plant: MeasurementPlant,
    reference_observation: Observation,
    target_energy_J: float,
    config=ControllerConfig(),
    *,
    black_level_adu=32,
    on_step: Callable | None = None,
    on_observation: Callable | None = None,
    cancelled: Callable[[], bool] | None = None,
    phase_geometry: PhaseControlGeometry | None = None,
):
    """Run the selected measurement-only controller."""
    if config.method == "interferometric":
        if phase_geometry is None:
            raise ValueError("interferometric control requires calibrated geometry")
        return run_interferometric_controller(
            plant, reference_observation, target_energy_J, config, phase_geometry,
            black_level_adu=black_level_adu, on_step=on_step,
            on_observation=on_observation, cancelled=cancelled,
        )
    if config.method == "spgd":
        return run_spgd_controller(
            plant,
            reference_observation,
            target_energy_J,
            config,
            black_level_adu=black_level_adu,
            on_step=on_step,
            on_observation=on_observation,
            cancelled=cancelled,
        )
    return run_response_matrix_controller(
        plant, reference_observation, target_energy_J, config,
        black_level_adu=black_level_adu, on_step=on_step,
        on_observation=on_observation, cancelled=cancelled,
    )


def run_response_matrix_controller(
    plant: MeasurementPlant,
    reference_observation: Observation,
    target_energy_J: float,
    config: ControllerConfig,
    *,
    black_level_adu=32,
    on_step: Callable | None = None,
    on_observation: Callable | None = None,
    cancelled: Callable[[], bool] | None = None,
):
    """Fit a small modal response matrix from measured plus/minus probes."""
    if not np.isfinite(target_energy_J) or target_energy_J <= 0:
        raise ValueError("positive calibrated reference energy required")
    if (
        not reference_observation.valid
        or reference_observation.saturated_fraction > config.saturation_limit
    ):
        raise ValueError("reference diagnostic observation is invalid or saturated")
    ref_array = np.asarray(reference_observation.camera_adu)
    reference = _Reference(
        _calibrated_vector(reference_observation, black_level_adu, ref_array.shape),
        ref_array.shape,
        ref_array.shape[0],
    )
    coeff = np.zeros(config.mode_count)
    count = 0
    history = []

    def acquire(a):
        nonlocal count
        for _ in range(3):
            if cancelled is not None and cancelled():
                raise InterruptedError("controller cancelled")
            if count >= config.evaluation_limit:
                raise StopIteration("evaluation budget reached")
            obs = plant.observe(np.asarray(a, float).copy())
            count += 1
            if not np.allclose(obs.delivered_coefficients_rad, a, rtol=0, atol=1e-8):
                raise ValueError("observation was exposed under another SLM command")
            if on_observation is not None:
                on_observation(obs, count)
            if obs.valid and obs.saturated_fraction <= config.saturation_limit:
                return obs
        raise InvalidObservationError(
            "three invalid or saturated observations under one command"
        )

    current = acquire(coeff)
    _, loss, energy = _score(
        current, reference, target_energy_J, config, black_level_adu
    )
    row = dict(
        iteration=0,
        evaluation=count,
        camera_loss=loss,
        measured_energy_J=current.measured_energy_J,
        energy_fraction=energy,
        time_s=current.time_s,
        coefficients_rad=coeff.copy(),
        accepted=True,
    )
    history.append(row)
    if on_step:
        on_step(row, current)
    status = "iteration_limit"
    response_matrix = None
    for iteration in range(1, config.iterations + 1):
        if loss <= config.target_loss:
            status = "target_loss_reached"
            break
        try:
            baseline, _, baseline_energy = _score(
                current, reference, target_energy_J, config, black_level_adu
            )
            if response_matrix is None or (iteration - 1) % config.refresh_every == 0:
                columns = []
                for j in range(config.mode_count):
                    shift = np.zeros_like(coeff)
                    shift[j] = config.perturbation_rad
                    plus = acquire(coeff + shift)
                    minus = acquire(coeff - shift)
                    vp, _, ep = _score(
                        plus, reference, target_energy_J, config, black_level_adu
                    )
                    vm, _, em = _score(
                        minus, reference, target_energy_J, config, black_level_adu
                    )
                    fp = np.r_[vp, config.energy_weight * max(0, 1 - ep)]
                    fm = np.r_[vm, config.energy_weight * max(0, 1 - em)]
                    columns.append((fp - fm) / (2 * config.perturbation_rad))
                response_matrix = np.column_stack(columns)
            # The response probes themselves advance time in in-situ mode.
            # Reacquire the baseline under its actual returned command so
            # drift cannot masquerade as a correction response.
            baseline_observation = acquire(coeff)
            baseline_now, baseline_loss, baseline_energy = _score(
                baseline_observation,
                reference,
                target_energy_J,
                config,
                black_level_adu,
            )
            if np.linalg.norm(baseline_now - baseline) > max(
                0.08, config.noise_acceptance_sigma * 0.01
            ):
                status = "response_drift"
                break
            baseline = baseline_now
            loss = baseline_loss
            g = response_matrix
            residual = np.r_[
                baseline - reference.vector,
                config.energy_weight * max(0, 1 - baseline_energy),
            ]
            delta = -np.linalg.solve(
                g.T @ g + config.regularization * np.eye(len(coeff)), g.T @ residual
            )
            delta = np.clip(delta, -config.max_update_rad, config.max_update_rad)
            if not np.all(np.isfinite(delta)):
                raise ValueError("nonfinite controller update")
            accepted = False
            for factor in (1.0, 0.5, 0.25):
                candidate = coeff + factor * delta
                trial = acquire(candidate)
                _, trial_loss, trial_energy = _score(
                    trial, reference, target_energy_J, config, black_level_adu
                )
                # A validation capture at the same command estimates whether
                # a small apparent improvement exceeds detector noise.
                verify = acquire(candidate)
                _, verify_loss, verify_energy = _score(
                    verify, reference, target_energy_J, config, black_level_adu
                )
                new_loss = 0.5 * (trial_loss + verify_loss)
                uncertainty = (
                    config.noise_acceptance_sigma * abs(trial_loss - verify_loss) / 2
                )
                if (
                    new_loss + uncertainty < loss - config.improvement_tolerance
                    and min(trial_energy, verify_energy)
                    >= config.minimum_energy_fraction
                ):
                    coeff = candidate
                    current = verify
                    loss = new_loss
                    energy = 0.5 * (trial_energy + verify_energy)
                    accepted = True
                    break
            if not accepted:
                # Rejected in-situ commands already affected the real state.
                # Restoring the previous command is another timed action.
                current = acquire(coeff)
                status = "no_measured_improvement"
                break
            row = dict(
                iteration=iteration,
                evaluation=count,
                camera_loss=loss,
                measured_energy_J=current.measured_energy_J,
                energy_fraction=energy,
                time_s=current.time_s,
                coefficients_rad=coeff.copy(),
                accepted=True,
            )
            history.append(row)
            if on_step:
                on_step(row, current)
        except StopIteration:
            status = "evaluation_budget"
            break
        except InvalidObservationError:
            status = "camera_observation_invalid"
            break
    return dict(
        status=status, history=history, evaluations=count, final_coefficients_rad=coeff
    )


def run_interferometric_controller(
    plant: MeasurementPlant,
    reference_observation: Observation,
    target_energy_J: float,
    config: ControllerConfig,
    geometry: PhaseControlGeometry,
    *,
    black_level_adu=32,
    on_step: Callable | None = None,
    on_observation: Callable | None = None,
    cancelled: Callable[[], bool] | None = None,
):
    """Correct a pixelwise SLM map from four measured interferograms per update.

    The adjoint is the calibrated free-space SLM-to-output path. Disk gain and
    thermal feedback are *not* differentiated; the following exposure measures
    their actual response. No solver truth enters this controller.
    """
    if not np.isfinite(target_energy_J) or target_energy_J <= 0:
        raise ValueError("positive calibrated reference energy required")
    target_field = reference_observation.phase_field
    if target_field is None or np.shape(target_field) != geometry.grid.shape:
        raise ValueError("a calibrated interferometric reference is required")
    target_field = np.asarray(target_field, complex)
    if not np.all(np.isfinite(target_field)) or np.max(abs(target_field)) <= 0:
        raise ValueError("invalid interferometric reference field")
    frame = np.asarray(reference_observation.camera_adu)
    reference = _Reference(
        _calibrated_vector(reference_observation, black_level_adu, frame.shape),
        frame.shape, frame.shape[0],
    )
    command = np.zeros(geometry.grid.shape, float)
    pupil = abs(geometry.source_field) > .01 * np.max(abs(geometry.source_field))
    weight = abs(geometry.source_field) ** 2
    weight = np.where(pupil, weight, 0.0)
    weight /= max(float(weight.sum()), 1e-30)
    count = 0
    history = []

    def acquire(requested):
        nonlocal count
        for _ in range(3):
            if cancelled is not None and cancelled():
                raise InterruptedError("controller cancelled")
            if count >= config.evaluation_limit:
                raise StopIteration("evaluation budget reached")
            obs = plant.observe(requested.copy())
            count += 1
            if not np.allclose(obs.delivered_coefficients_rad, requested, rtol=0, atol=1e-8):
                raise ValueError("observation was exposed under another SLM command")
            if on_observation is not None:
                on_observation(obs, count)
            if (obs.valid and obs.saturated_fraction <= config.saturation_limit
                    and obs.phase_field is not None
                    and np.shape(obs.phase_field) == geometry.grid.shape
                    and np.all(np.isfinite(obs.phase_field))):
                return obs
        raise InvalidObservationError("three invalid interferometric observations")

    def metrics(obs):
        _, camera_loss, energy = _score(
            obs, reference, target_energy_J, config, black_level_adu
        )
        _, phase_rms = piston_removed_residual(obs.phase_field, target_field)
        return camera_loss, energy, phase_rms

    def append(iteration, obs, status, step_rms, *, rejected_trial_loss=None):
        camera_loss, energy, phase_rms = metrics(obs)
        row = dict(
            iteration=iteration, evaluation=count, camera_loss=camera_loss,
            measured_energy_J=obs.measured_energy_J, energy_fraction=energy,
            phase_rms_rad=phase_rms, time_s=obs.time_s,
            coefficients_rad=command.copy(), accepted=status != "rejected_restore",
            improved=status in ("initial", "improved"), update_status=status,
            attempted_step_rms_rad=step_rms,
            rejected_trial_camera_loss=rejected_trial_loss,
        )
        history.append(row)
        if on_step is not None:
            on_step(row, obs)
        return camera_loss, energy, phase_rms

    current = acquire(command)
    camera_loss, energy, phase_rms = append(0, current, "initial", 0.0)
    best_command = command.copy()
    best_score = camera_loss + phase_rms / np.pi
    best_camera_loss = camera_loss
    confirmations = 0
    consecutive_rejections = 0
    status = "iteration_limit"
    gain = config.phase_gain_rad
    for iteration in range(1, config.iterations + 1):
        if confirmations >= config.target_confirmations:
            status = "measured_phase_confirmed"
            break
        try:
            observed = np.asarray(current.phase_field, complex)
            scale = np.vdot(target_field, observed)
            if abs(scale) <= 1e-30:
                raise InvalidObservationError("interferometric fringe visibility vanished")
            aligned = observed * np.exp(-1j * np.angle(scale))
            aligned *= np.linalg.norm(target_field) / max(np.linalg.norm(aligned), 1e-30)
            error = aligned - target_field
            adjoint = angular_spectrum_propagate(
                error, geometry.grid, geometry.wavelength_m,
                -geometry.slm_to_disk_m,
            )
            at_slm = geometry.source_field * np.exp(
                1j * (geometry.desired_slm_phase_rad + command)
            )
            gradient = np.imag(np.conj(adjoint) * at_slm)
            gradient = np.where(pupil, gradient, 0.0)
            if config.phase_smoothing_pixels:
                gradient = gaussian_filter(gradient, config.phase_smoothing_pixels)
            gradient -= np.sum(weight * gradient)
            rms = np.sqrt(np.sum(weight * gradient**2))
            if not np.isfinite(rms) or rms < 1e-12:
                status = "no_observable_phase_gradient"
                break
            update = np.clip(gain * gradient / rms,
                             -config.max_update_rad, config.max_update_rad)
            previous = command.copy()
            baseline_score = camera_loss + phase_rms / np.pi
            accepted = False
            rejected_trial_loss = None
            for factor in (1.0, 0.5, 0.25):
                candidate = np.clip(
                    previous + factor * update,
                    -config.coefficient_limit_rad, config.coefficient_limit_rad,
                )
                candidate = np.where(pupil, candidate, 0.0)
                trial = acquire(candidate)
                trial_camera, trial_energy, trial_phase = metrics(trial)
                new_score = trial_camera + trial_phase / np.pi
                # Phase fidelity may improve while the actual camera image
                # worsens. The camera floor is therefore a separate guard,
                # not a term that phase error can compensate for.
                if (trial_energy >= config.minimum_energy_fraction
                        and trial_camera <= best_camera_loss + config.improvement_tolerance
                        and new_score < baseline_score - config.improvement_tolerance):
                    accepted = True
                    break
                rejected_trial_loss = trial_camera
            if not accepted:
                # The unsuccessful probes advanced a real in-situ plant. A
                # rollback is another timed exposure, never a silent rewind.
                current = acquire(previous)
                command = previous
                camera_loss, energy, phase_rms = append(
                    iteration, current, "rejected_restore",
                    float(np.sqrt(np.sum(weight * update**2))),
                    rejected_trial_loss=rejected_trial_loss,
                )
                gain = max(.01, gain * .5)
                confirmations = 0
                consecutive_rejections += 1
                if consecutive_rejections >= 3:
                    status = "no_measured_improvement"
                    break
                continue
            consecutive_rejections = 0
            command = candidate
            current = trial
            camera_loss, energy, phase_rms = append(
                iteration, current, "improved",
                float(np.sqrt(np.sum(weight * (factor * update)**2))),
            )
            best_camera_loss = min(best_camera_loss, camera_loss)
            gain = min(config.phase_gain_rad, gain * 1.02)
            if new_score < best_score and trial_energy >= config.minimum_energy_fraction:
                best_score = new_score
                best_command = command.copy()
            confirmations = (confirmations + 1 if
                phase_rms <= config.phase_target_rms_rad
                and camera_loss <= config.target_loss
                and energy >= config.minimum_energy_fraction else 0)
        except StopIteration:
            status = "evaluation_budget"
            break
        except InvalidObservationError:
            status = "camera_observation_invalid"
            break
    best_recheck_status = "not_needed"
    if config.restore_best_at_end and not np.array_equal(command, best_command):
        try:
            present = acquire(command)
            present_camera, present_energy, present_phase = metrics(present)
            saved = acquire(best_command)
            saved_camera, saved_energy, saved_phase = metrics(saved)
            present_score = present_camera + present_phase / np.pi
            saved_score = saved_camera + saved_phase / np.pi
            if (
                saved_energy >= config.minimum_energy_fraction
                and saved_camera <= present_camera + config.improvement_tolerance
                and saved_score < present_score - config.improvement_tolerance
            ):
                command = best_command.copy()
                current = saved
                best_recheck_status = "restored"
            else:
                # Measuring the saved command changed the physical SLM and
                # advanced the in-situ thermal clock. Restore the current
                # command through another measured observation.
                current = acquire(command)
                best_recheck_status = "kept_current"
            append(
                len(history), current,
                "best_restore" if best_recheck_status == "restored" else "best_rejected",
                0.0,
            )
        except StopIteration:
            best_recheck_status = "evaluation_budget"
        except InvalidObservationError:
            best_recheck_status = "camera_observation_invalid"
    return dict(status=status, history=history, evaluations=count,
                final_coefficients_rad=command, best_recheck_status=best_recheck_status)


def run_spgd_controller(
    plant: MeasurementPlant,
    reference_observation: Observation,
    target_energy_J: float,
    config: ControllerConfig,
    *,
    black_level_adu=32,
    on_step: Callable | None = None,
    on_observation: Callable | None = None,
    cancelled: Callable[[], bool] | None = None,
):
    """Sequential stochastic parallel-gradient descent from camera/energy data.

    A physical plus/minus pair estimates one random simultaneous phase-mode
    direction. The resulting command is applied and measured once. Deteriorating
    updates remain in the chronological history; only the hard energy floor
    triggers a new rollback command. No hidden field or phase is consulted.
    """
    if not np.isfinite(target_energy_J) or target_energy_J <= 0:
        raise ValueError("positive calibrated reference energy required")
    if (
        not reference_observation.valid
        or reference_observation.saturated_fraction > config.saturation_limit
    ):
        raise ValueError("reference diagnostic observation is invalid or saturated")
    frame = np.asarray(reference_observation.camera_adu)
    reference = _Reference(
        _calibrated_vector(reference_observation, black_level_adu, frame.shape),
        frame.shape,
        frame.shape[0],
    )
    rng = np.random.default_rng(config.spgd_seed)
    coeff = np.zeros(config.mode_count)
    velocity = np.zeros_like(coeff)
    count = 0
    history = []

    def acquire(command):
        nonlocal count
        for _ in range(3):
            if cancelled is not None and cancelled():
                raise InterruptedError("controller cancelled")
            if count >= config.evaluation_limit:
                raise StopIteration("evaluation budget reached")
            obs = plant.observe(np.asarray(command, float).copy())
            count += 1
            if not np.allclose(
                obs.delivered_coefficients_rad, command, rtol=0, atol=1e-8
            ):
                raise ValueError("observation was exposed under another SLM command")
            if on_observation is not None:
                on_observation(obs, count)
            if obs.valid and obs.saturated_fraction <= config.saturation_limit:
                return obs
        raise InvalidObservationError(
            "three invalid or saturated observations under one command"
        )

    current = acquire(coeff)
    _, loss, energy = _score(
        current, reference, target_energy_J, config, black_level_adu
    )
    row = dict(
        iteration=0,
        evaluation=count,
        camera_loss=loss,
        measured_energy_J=current.measured_energy_J,
        energy_fraction=energy,
        time_s=current.time_s,
        coefficients_rad=coeff.copy(),
        accepted=True,
        improved=True,
        update_status="initial",
        attempted_step_rms_rad=0.0,
    )
    history.append(row)
    if on_step:
        on_step(row, current)
    best_coeff = coeff.copy()
    best_loss = loss
    status = "iteration_limit"
    confirmations = int(loss <= config.target_loss)
    for iteration in range(1, config.iterations + 1):
        if confirmations >= config.target_confirmations:
            status = "target_loss_confirmed"
            break
        try:
            signs = rng.choice((-1.0, 1.0), size=config.mode_count)
            dither = config.perturbation_rad / (1 + iteration / 50) ** 0.1
            order = (1.0, -1.0) if rng.integers(0, 2) else (-1.0, 1.0)
            scores = {}
            for sign in order:
                probe = acquire(coeff + sign * dither * signs)
                _, probe_loss, _ = _score(
                    probe, reference, target_energy_J, config, black_level_adu
                )
                scores[sign] = probe_loss
            gradient = (scores[1.0] - scores[-1.0]) / (2 * dither) * signs
            velocity = (
                config.spgd_momentum * velocity + (1 - config.spgd_momentum) * gradient
            )
            gain = config.spgd_gain / (1 + iteration / 20) ** config.spgd_decay
            step = np.clip(
                -gain * velocity, -config.max_update_rad, config.max_update_rad
            )
            candidate = np.clip(
                coeff + step,
                -config.coefficient_limit_rad,
                config.coefficient_limit_rad,
            )
            trial = acquire(candidate)
            _, trial_loss, trial_energy = _score(
                trial, reference, target_energy_J, config, black_level_adu
            )
            if trial_energy < config.minimum_energy_fraction:
                # The trial already happened. Restore the previous command
                # through a new timed physical observation.
                current = acquire(coeff)
                _, loss, energy = _score(
                    current, reference, target_energy_J, config, black_level_adu
                )
                update_status = "energy_rollback"
                applied = False
                velocity *= 0.5
            else:
                previous_loss = loss
                coeff = candidate
                current = trial
                loss = trial_loss
                energy = trial_energy
                applied = True
                update_status = "improved" if loss < previous_loss else "regressed"
                if loss < best_loss - config.improvement_tolerance:
                    best_loss = loss
                    best_coeff = coeff.copy()
            confirmations = (
                confirmations + 1 if loss <= config.target_loss and applied else 0
            )
            row = dict(
                iteration=iteration,
                evaluation=count,
                camera_loss=loss,
                measured_energy_J=current.measured_energy_J,
                energy_fraction=energy,
                time_s=current.time_s,
                coefficients_rad=coeff.copy(),
                accepted=applied,
                improved=update_status == "improved",
                update_status=update_status,
                attempted_step_rms_rad=float(np.sqrt(np.mean(step**2))),
                probe_loss_plus=scores[1.0],
                probe_loss_minus=scores[-1.0],
            )
            history.append(row)
            if on_step:
                on_step(row, current)
        except StopIteration:
            status = "evaluation_budget"
            break
        except InvalidObservationError:
            status = "camera_observation_invalid"
            break
    best_recheck_status = "not_needed"
    if (
        status != "camera_observation_invalid"
        and config.restore_best_at_end
        and not np.allclose(best_coeff, coeff, rtol=0, atol=1e-8)
    ):
        prior_coeff = coeff.copy()
        try:
            present = acquire(prior_coeff)
            _, present_loss, present_energy = _score(
                present, reference, target_energy_J, config, black_level_adu
            )
            saved = acquire(best_coeff)
            _, saved_loss, saved_energy = _score(
                saved, reference, target_energy_J, config, black_level_adu
            )
            if (
                saved_energy >= config.minimum_energy_fraction
                and saved_loss + config.improvement_tolerance < present_loss
            ):
                coeff = best_coeff.copy()
                current = saved
                loss = saved_loss
                energy = saved_energy
                best_recheck_status = "restored"
            else:
                current = acquire(prior_coeff)
                _, loss, energy = _score(
                    current, reference, target_energy_J, config, black_level_adu
                )
                best_recheck_status = "kept_current"
            row = dict(
                iteration=history[-1]["iteration"],
                evaluation=count,
                camera_loss=loss,
                measured_energy_J=current.measured_energy_J,
                energy_fraction=energy,
                time_s=current.time_s,
                coefficients_rad=coeff.copy(),
                accepted=best_recheck_status == "restored",
                improved=loss < present_loss,
                update_status=(
                    "best_restore"
                    if best_recheck_status == "restored"
                    else "best_rejected"
                ),
                attempted_step_rms_rad=float(
                    np.sqrt(np.mean((best_coeff - prior_coeff) ** 2))
                ),
            )
            history.append(row)
            if on_step:
                on_step(row, current)
        except StopIteration:
            best_recheck_status = "evaluation_budget"
        except InvalidObservationError:
            best_recheck_status = "camera_observation_invalid"
            status = "camera_observation_invalid"
    return dict(
        status=status,
        history=history,
        evaluations=count,
        final_coefficients_rad=coeff,
        best_recheck_status=best_recheck_status,
    )


def measurement_metrics(
    observation: Observation,
    reference_observation: Observation,
    target_energy_J: float,
    config=ControllerConfig(),
    *,
    black_level_adu=32,
):
    """Score an independent camera/photodiode reading without simulator truth."""
    if not observation.valid or observation.saturated_fraction > config.saturation_limit:
        return dict(
            camera_loss=None,
            energy_fraction=float(observation.measured_energy_J / target_energy_J),
            valid=False,
            saturated_fraction=float(observation.saturated_fraction),
        )
    reference_frame = np.asarray(reference_observation.camera_adu)
    reference = _Reference(
        _calibrated_vector(
            reference_observation, black_level_adu, reference_frame.shape
        ),
        reference_frame.shape,
        reference_frame.shape[0],
    )
    _, loss, energy_fraction = _score(
        observation, reference, target_energy_J, config, black_level_adu
    )
    return dict(
        camera_loss=loss,
        energy_fraction=energy_fraction,
        valid=True,
        saturated_fraction=float(observation.saturated_fraction),
    )
