"""Bounded modal correction teacher with a frozen optical proposal model.

The proposal model reuses the solved input field and physical phase screens.
Only the reference callback can qualify a label. It must replay each command
from the same pre-action material state and must preserve populations within
each amplifier pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np
from scipy.optimize import minimize

from hoyag.propagation import angular_spectrum_propagate
from ybluag.multipass_geometry import uniform as uniform_multipass
from .distortions.slm import apply_slm
from .field_metrics import FieldMetrics, field_metrics, phase_support


def _zernike(n: int, m: int, radius: np.ndarray, angle: np.ndarray) -> np.ndarray:
    order = abs(m)
    radial = np.zeros_like(radius)
    for s in range((n - order) // 2 + 1):
        coefficient = ((-1) ** s * factorial(n - s) /
                       (factorial(s) * factorial((n + order) // 2 - s) *
                        factorial((n - order) // 2 - s)))
        radial += coefficient * radius ** (n - 2 * s)
    if m < 0:
        return radial * np.sin(order * angle)
    if m > 0:
        return radial * np.cos(order * angle)
    return radial


def modal_basis(x_m: np.ndarray, y_m: np.ndarray, input_field: np.ndarray,
                waist_m: float, *, radial_order: int = 4,
                residual_grid: int = 0) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Weighted-orthonormal, piston-free modes on the illuminated input pupil."""
    x_m, y_m = np.asarray(x_m, float), np.asarray(y_m, float)
    power = np.abs(np.asarray(input_field, complex)) ** 2
    if (x_m.shape != y_m.shape or power.shape != x_m.shape or x_m.ndim != 2 or
            waist_m <= 0 or radial_order < 1 or residual_grid < 0 or
            power.max() <= 0):
        raise ValueError("invalid modal pupil")
    pupil = power >= 1e-3 * power.max()
    weight = np.where(pupil, power, 0.0)
    weight /= weight.sum()
    cx = float(np.sum(weight * x_m))
    cy = float(np.sum(weight * y_m))
    radius = np.hypot(x_m - cx, y_m - cy) / (2 * waist_m)
    angle = np.arctan2(y_m - cy, x_m - cx)
    raw = []
    names = []
    for n in range(1, radial_order + 1):
        for m in range(-n, n + 1, 2):
            raw.append(_zernike(n, m, radius, angle))
            names.append(f"Z_{n}_{m:+d}")
    if residual_grid:
        positions = np.linspace(-1.1, 1.1, residual_grid)
        for row, py in enumerate(positions):
            for column, px in enumerate(positions):
                raw.append(np.exp(-((x_m - cx - px * waist_m) ** 2 +
                                    (y_m - cy - py * waist_m) ** 2) /
                                  (0.65 * waist_m) ** 2))
                names.append(f"residual_{row}_{column}")
    raw = np.stack(raw, axis=-1)
    raw -= np.sum(weight[..., None] * raw, axis=(0, 1))
    weighted = raw[pupil] * np.sqrt(weight[pupil, None])
    _, triangular = np.linalg.qr(weighted, mode="reduced")
    singular = np.linalg.svd(triangular, compute_uv=False)
    if singular[-1] <= singular[0] * 1e-9:
        raise ValueError("modal pupil is rank deficient")
    basis = (raw @ np.linalg.inv(triangular)).transpose(2, 0, 1)
    # Away from the useful pupil, commanding large high-order values only
    # generates numerical edge diffraction. This boundary carries <0.1% of
    # the nominal input energy by construction.
    basis *= pupil[None]
    return names, basis, weight


@dataclass
class ModalTeacherResult:
    step_rad: np.ndarray | None
    command_rad: np.ndarray | None
    field: np.ndarray | None
    metrics: FieldMetrics | None
    status: str
    checks: list[dict]
    evaluations: int
    selected_candidate: str | None
    baseline_metrics: FieldMetrics
    passive_fidelity_limit: float
    baseline_model_overlap: float
    best_improved_command_rad: np.ndarray | None = None
    best_improved_metrics: FieldMetrics | None = None
    selected_coefficients_rad: np.ndarray | None = None
    best_improved_coefficients_rad: np.ndarray | None = None


class FrozenOpticalModel:
    """Calibrated passive proposal; never treated as the reference amplifier."""

    def __init__(self, *, grid, wavelength_m, slm_to_disk_m, output_distance_m,
                 input_field, current_command, actual_slm_phase, external_phase,
                 screen_phase, baseline_field, slm_setup, drift_fraction=0.,
                 encounters=1, inter_pass_distance_m=0., inter_pass_focal_m=0.,
                 geometry=None):
        """``screen_phase`` is the total disk phase over all encounters.

        With a multipass ``geometry`` (or a nonzero uniform
        ``inter_pass_distance_m``) it is split equally over the encounters,
        each seen at its own angle of incidence, with the solver's relay paths
        between them; otherwise it is one screen.
        """
        self.grid = grid
        self.wavelength_m = float(wavelength_m)
        self.slm_to_disk_m = float(slm_to_disk_m)
        self.output_distance_m = float(output_distance_m)
        self.input_field = np.asarray(input_field, complex)
        self.current_command = np.asarray(current_command, float)
        self.actual_slm_phase = np.asarray(actual_slm_phase, float)
        self.external_phase = np.asarray(external_phase, float)
        if geometry is None and float(inter_pass_distance_m) > 0:
            geometry = uniform_multipass(int(encounters), float(inter_pass_distance_m),
                                         float(inter_pass_focal_m))
        self.geometry = geometry
        self.encounters = 1 if geometry is None else geometry.encounters
        per = np.asarray(screen_phase, float) / self.encounters
        if geometry is None:
            self.screens = [np.exp(1j * per)]
            self.mirrors = []
        else:
            self.screens = [np.exp(1j * geometry.encounter_map(k, per, grid))
                            for k in range(self.encounters)]
            self.mirrors = geometry.mirror_phases(grid, self.wavelength_m)
        self.screen = self.screens[0]
        self.slm_setup = slm_setup
        self.drift_fraction = float(drift_fraction)
        shape = grid.shape
        for value in (self.input_field, self.current_command,
                      self.actual_slm_phase, self.external_phase, *self.screens):
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError("passive model arrays must be finite and aligned")
        self.source = self.input_field * np.exp(
            -1j * (self.actual_slm_phase + self.external_phase))
        unscaled = self._operator(self.input_field)
        baseline_field = np.asarray(baseline_field, complex)
        if baseline_field.shape != shape or np.vdot(unscaled, unscaled).real <= 0:
            raise ValueError("invalid baseline for passive calibration")
        self.scale = np.vdot(unscaled, baseline_field) / np.vdot(unscaled, unscaled)
        self.baseline_overlap = self._overlap(unscaled, baseline_field)

    @staticmethod
    def _overlap(a, b):
        return float(np.abs(np.vdot(a, b)) ** 2 /
                     (np.vdot(a, a).real * np.vdot(b, b).real))

    def _operator(self, input_field):
        output = angular_spectrum_propagate(input_field, self.grid,
            self.wavelength_m, self.slm_to_disk_m) * self.screens[0]
        for k in range(1, self.encounters):
            output = self.geometry.relay(k-1, output, self.grid, self.wavelength_m,
                                         self.mirrors[k-1]) * self.screens[k]
        if self.output_distance_m:
            output = angular_spectrum_propagate(output, self.grid,
                self.wavelength_m, self.output_distance_m)
        return output

    def _adjoint(self, output_field):
        field = np.asarray(output_field, complex)
        if self.output_distance_m:
            field = angular_spectrum_propagate(field, self.grid,
                self.wavelength_m, -self.output_distance_m)
        for k in range(self.encounters-1, 0, -1):
            field = field * np.conj(self.screens[k])
            field = self.geometry.relay_adjoint(k-1, field, self.grid, self.wavelength_m,
                                                self.mirrors[k-1])
        field = field * np.conj(self.screens[0])
        return angular_spectrum_propagate(field, self.grid,
            self.wavelength_m, -self.slm_to_disk_m)

    def forward_delta(self, step):
        input_field = self.input_field * np.exp(1j * step)
        return self.scale * self._operator(input_field), input_field

    def forward_command(self, command):
        applied, _ = apply_slm(command, self.slm_setup,
                              drift_fraction=self.drift_fraction)
        input_field = self.source * np.exp(1j * (applied + self.external_phase))
        return self.scale * self._operator(input_field)

    def adjoint_scaled(self, output_field):
        return np.conj(self.scale) * self._adjoint(output_field)

    def fidelity_limit(self, desired):
        """Exact phase-only full-field bound for this passive unitary operator."""
        backward = self._adjoint(desired)
        amplitude = np.abs(self.source)
        return float(np.sum(amplitude * np.abs(backward)) ** 2 /
                     (np.sum(amplitude ** 2) * np.sum(np.abs(backward) ** 2)))


def _loss_gradient(coefficients, basis, model, desired, shape_requirement,
                   shape_penalty, regularization):
    step = np.einsum("i,ijk->jk", coefficients, basis, optimize=True)
    field, input_field = model.forward_delta(step)
    norm = float(np.vdot(field, field).real)
    target_norm = float(np.vdot(desired, desired).real)
    inner = np.vdot(desired, field)
    fidelity = float(np.abs(inner) ** 2 / (norm * target_norm))
    g_fidelity = inner * desired / (norm * target_norm) - fidelity * field / norm
    target_amplitude = np.abs(desired)
    amplitude = np.abs(field)
    amplitude_inner = float(np.sum(amplitude * target_amplitude))
    shape = float(amplitude_inner ** 2 / (norm * target_norm))
    g_shape = (amplitude_inner * target_amplitude * field /
               np.maximum(amplitude, 1e-30) / (norm * target_norm) -
               shape * field / norm)
    violation = max(0., shape_requirement - shape)
    loss = 1. - fidelity + shape_penalty * violation ** 2 + \
           regularization * float(np.dot(coefficients, coefficients))
    field_gradient = -g_fidelity - 2 * shape_penalty * violation * g_shape
    input_gradient = model.adjoint_scaled(field_gradient)
    phase_gradient = -2 * np.imag(np.conj(input_gradient) * input_field)
    coefficient_gradient = np.einsum("jk,ijk->i", phase_gradient, basis,
                                     optimize=True) + 2 * regularization * coefficients
    return float(loss), coefficient_gradient, fidelity, shape


def _backprop_modal_start(model, desired, basis):
    """Lift a smooth physical inverse before fitting; reject vortex seams."""
    backward = model._adjoint(desired)
    relative = np.angle(backward * np.conj(model.input_field))
    lifted = np.unwrap(np.unwrap(relative, axis=0), axis=1)
    amplitude = np.abs(model.input_field) ** 2
    support = amplitude >= 1e-3 * amplitude.max()
    weights = amplitude[support]
    weights /= weights.sum()
    matrix = basis[:, support].T
    target = lifted[support] - np.sum(weights * lifted[support])
    coefficients = np.linalg.lstsq(matrix * np.sqrt(weights[:, None]),
                                   target * np.sqrt(weights), rcond=None)[0]
    coefficients = np.clip(coefficients, -16., 16.)
    fit = np.einsum("i,ijk->jk", coefficients, basis, optimize=True)
    error_phase = fit[support] - relative[support]
    piston = np.angle(np.sum(weights * np.exp(1j * error_phase)))
    circular_error = float(np.sqrt(np.sum(weights * np.abs(
        np.exp(1j * (error_phase - piston)) - 1) ** 2)))
    return coefficients, circular_error


def modal_correction_teacher(baseline, desired, current_command, model,
                             verify, basis, *, max_reference_evaluations=4,
                             min_coherent_gain=0.01,
                             target_fidelity=0.9,
                             min_shape_overlap=0.95,
                             max_shape_drop=0.012,
                             min_energy_fraction=0.8,
                             max_fast_iterations=60,
                             shape_penalty=30.):
    """Search continuous modes; qualify exact commands with independent replay."""
    baseline = np.asarray(baseline, complex)
    desired = np.asarray(desired, complex)
    current_command = np.asarray(current_command, float)
    basis = np.asarray(basis, float)
    if (baseline.shape != desired.shape or baseline.shape != current_command.shape or
            basis.ndim != 3 or basis.shape[1:] != baseline.shape or
            max_reference_evaluations < 1 or target_fidelity <= 0 or
            target_fidelity > 1 or not np.all(np.isfinite(basis))):
        raise ValueError("invalid modal teacher inputs")
    weights = np.abs(baseline) ** 2
    support = phase_support(baseline, desired)
    before = field_metrics(baseline, desired, weights, support)
    shape_requirement = max(min_shape_overlap,
                            before.shape_overlap - max_shape_drop)
    bound = model.fidelity_limit(desired)
    checks = [{"kind": "reachability", "passive_fidelity_limit": bound,
               "baseline_model_overlap": model.baseline_overlap,
               "assumption": "frozen passive full-field phase-only operator"}]
    verification_budget = (min(max_reference_evaluations, 1)
                           if bound < target_fidelity else max_reference_evaluations)
    checks.append({"kind": "reference_budget",
                   "limit": verification_budget,
                   "reason": ("passive target bound below required fidelity"
                              if bound < target_fidelity else "configured limit")})
    result = ModalTeacherResult(None, None, None, None, "search_incomplete",
                                checks, 0, None, before, bound,
                                model.baseline_overlap)
    if (before.coherent_fidelity >= target_fidelity and
            before.shape_overlap >= shape_requirement):
        # Settled operation needs a verified hold label so the controller can
        # learn that no further correction is required.
        try:
            held_command = np.mod(current_command, 2 * np.pi)
            result.evaluations = 1
            held_field, valid = verify(held_command)
            if not valid:
                raise ValueError("reference solver rejected hold command")
            held = field_metrics(held_field, desired, weights, support)
            if (held.coherent_fidelity < target_fidelity or
                    held.shape_overlap < shape_requirement or
                    held.output_norm < min_energy_fraction * before.output_norm):
                raise ValueError("fresh hold replay did not meet task criteria")
            result.step_rad = np.zeros_like(current_command)
            result.command_rad = held_command
            result.field = np.asarray(held_field, complex).copy()
            result.metrics = held
            result.status = "task_success"
            result.selected_candidate = "hold"
            result.selected_coefficients_rad = np.zeros(len(basis))
            checks.append({"kind": "reference_verification", "candidate": "hold",
                           "coherent_fidelity": held.coherent_fidelity,
                           "shape_overlap": held.shape_overlap,
                           "passed": True})
        except Exception as exc:
            result.status = "numerically_unresolved"
            checks.append({"kind": "reference_verification", "candidate": "hold",
                           "passed": False,
                           "error": f"{type(exc).__name__}: {exc}"})
        return result
    if model.baseline_overlap < 0.99:
        result.status = "numerically_unresolved"
        checks.append({"kind": "proposal_model", "reason": "baseline mismatch"})
        return result
    regularization = 1e-5
    best_fast = []

    def objective(coefficients):
        loss, gradient, fidelity, shape = _loss_gradient(
            coefficients, basis, model, desired, shape_requirement,
            shape_penalty, regularization)
        if shape >= shape_requirement - 1e-8:
            best_fast.append((fidelity, coefficients.copy()))
        return loss, gradient

    starts = [np.zeros(len(basis))]
    inverse_start, inverse_error = _backprop_modal_start(model, desired, basis)
    checks.append({"kind": "backprop_initialization",
                   "circular_fit_error": inverse_error,
                   "used": inverse_error < 0.45})
    if inverse_error < 0.45:
        starts.append(inverse_start)
    # A small deterministic tilt/defocus disturbance can escape a near-zero
    # initial complex overlap without using a wrapped oracle direction.
    if len(basis) >= 3 and before.coherent_fidelity < 1e-3:
        alternative = np.zeros(len(basis))
        alternative[:3] = (0.3, -0.2, 0.15)
        starts.append(alternative)
    for start in starts:
        fit = minimize(objective, start, jac=True, method="L-BFGS-B",
                       bounds=[(-16., 16.)] * len(basis),
                       options={"maxiter": max_fast_iterations,
                                "maxfun": max_fast_iterations * 3,
                                "ftol": 1e-9})
        loss, _, fidelity, shape = _loss_gradient(fit.x, basis, model, desired,
            shape_requirement, shape_penalty, regularization)
        checks.append({"kind": "modal_fit", "iterations": int(fit.nit),
                       "fast_fidelity": fidelity, "fast_shape": shape,
                       "loss": loss, "converged": bool(fit.success)})
        best_fast.append((fidelity, fit.x.copy()))
    ranked = sorted(best_fast, key=lambda row: row[0], reverse=True)
    proposals = []
    # The shape guard can cut between coarse fractional steps. Evaluate the
    # quantized SLM response on a fine one-dimensional line before spending
    # reference solves; the first admissible step may be near the boundary.
    gains = (*np.linspace(1., 0.25, 151), 0.2, 0.15, 0.1)
    for _, coefficients in ranked:
        for gain in gains:
            step = np.einsum("i,ijk->jk", gain * coefficients, basis, optimize=True)
            command = np.mod(current_command + step, 2 * np.pi)
            if any(np.sqrt(np.mean(np.angle(np.exp(1j * (command - old))) ** 2))
                   < 0.03 for _, old, _, _ in proposals):
                continue
            preview = field_metrics(model.forward_command(command), desired,
                                    weights, support)
            if preview.shape_overlap >= shape_requirement - 1e-8:
                proposals.append((preview.coherent_fidelity, command,
                                  f"modal_{gain:g}", gain * coefficients.copy()))
            if len(proposals) >= 20:
                break
        if len(proposals) >= 20:
            break
    proposals.sort(key=lambda row: row[0], reverse=True)
    for predicted_fidelity, command, name, coefficients in proposals[:verification_budget]:
        try:
            field, physically_valid = verify(command)
            if not physically_valid:
                raise ValueError("reference solver rejected physical state")
            metrics = field_metrics(field, desired, weights, support)
            energy_fraction = metrics.output_norm / before.output_norm
            shape_ok = metrics.shape_overlap >= shape_requirement
            improvement = metrics.coherent_fidelity - before.coherent_fidelity
            passed = (shape_ok and energy_fraction >= min_energy_fraction and
                      improvement >= min_coherent_gain)
            checks.append({"kind": "reference_verification", "candidate": name,
                           "predicted_fidelity": predicted_fidelity,
                           "coherent_fidelity": metrics.coherent_fidelity,
                           "shape_overlap": metrics.shape_overlap,
                           "phase_rms_rad": metrics.phase_rms_rad,
                           "energy_fraction": energy_fraction,
                           "passed": bool(passed)})
            if passed and (result.best_improved_metrics is None or
                           metrics.coherent_fidelity >
                           result.best_improved_metrics.coherent_fidelity):
                result.best_improved_command_rad = command.copy()
                result.best_improved_metrics = metrics
                result.best_improved_coefficients_rad = coefficients.copy()
                if metrics.coherent_fidelity >= target_fidelity:
                    result.step_rad = np.angle(np.exp(1j *
                        (command - current_command)))
                    result.command_rad = command.copy()
                    result.field = np.asarray(field, complex).copy()
                    result.metrics = metrics
                    result.status = "task_success"
                    result.selected_candidate = name
                    result.selected_coefficients_rad = coefficients.copy()
                else:
                    result.status = "improved_only"
        except Exception as exc:
            checks.append({"kind": "reference_verification", "candidate": name,
                           "passed": False,
                           "error": f"{type(exc).__name__}: {exc}"})
        result.evaluations += 1
    if result.status == "search_incomplete" and bound < target_fidelity:
        result.status = "control_limited_passive"
    return result
