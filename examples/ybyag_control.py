"""One bounded closed-loop Yb:YAG episode; launch through local_supervisor.

The controller receives detector measurements, photodiode energy, probe readings,
delivered phase commands and timestamps. Truth is recorded separately for audit.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ybyag_control import ControllerConfig, run_controller
from ybyag_control.controller import measurement_metrics
from ybyag_control.adapter import EpisodeConfig, SimulationPlant
from hoyag.local_supervisor import atomic_json
from ybyag.material_data import ApproximationWarning


# The thermal solver changes warning filters inside a context for matrix-rank
# diagnostics. That resets Python's once registry, so emit the material caveat
# explicitly once and suppress only this repeated interpolation warning.
warnings.filterwarnings(
    "ignore",
    message="Interpolating thermal resistivity across different dopings.",
    category=ApproximationWarning,
)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def run(config, *, progress_path=None):
    episode = EpisodeConfig(**config["episode"])
    controller = ControllerConfig(**config["controller"])
    if episode.yb_at_percent != 15.0:
        print(
            "Yb:YAG material approximation: interpolating thermal resistivity "
            "across measured dopings.",
            file=sys.stderr,
        )
    plant = SimulationPlant(
        episode, controller.mode_count,
        measurement_mode=("interferometric" if controller.method in ("interferometric", "hybrid")
                          else "intensity"),
    )
    start = time.perf_counter()
    # The reference method counts its ideal and physical warm-up solves.
    # Reserve one final snapshot solve or a held-plus-exposed in-situ pair.
    reserve = 2 if episode.mode == "in_situ" else 1
    minimum_solves = 6 if episode.mode == "in_situ" else 3
    if controller.evaluation_limit < minimum_solves:
        raise ValueError(
            "evaluation limit cannot cover reference, baseline and verification"
        )
    plant.full_solve_limit = controller.evaluation_limit - reserve
    target, target_energy = plant.reference()

    def camera_profiles(frames):
        signal = np.maximum(
            np.asarray(frames, float) - plant.camera.black_level_adu, 0.0
        )
        return dict(x_mean_adu=signal.mean(axis=1), y_mean_adu=signal.mean(axis=2))

    reference_profiles = camera_profiles(target.camera_adu)
    def phase_payload(observation):
        if observation.phase_field is None:
            return {}
        field = np.asarray(observation.phase_field, complex)
        return dict(
            measured_phase_rad=np.angle(field),
            measured_phase_amplitude=np.abs(field),
            interferograms_adu=observation.interferograms_adu,
        )
    steps = []
    observation_trace = []
    latest_observation = None
    progress_lock_reported = False

    def write_progress(path, value):
        nonlocal progress_lock_reported
        try:
            atomic_json(path, value)
        except PermissionError as exc:
            # The Tk reader (or a file scanner) may hold progress.json while
            # Windows refuses replacement. Progress is a disposable snapshot;
            # the next update can replace it, and the physical run must continue.
            if not progress_lock_reported:
                print(f"Live progress temporarily unavailable: {exc}", file=sys.stderr)
                progress_lock_reported = True

    def publish():
        if progress_path is not None:
            write_progress(
                Path(progress_path),
                _jsonable(
                    dict(
                        status="running",
                        episode=asdict(episode),
                        controller=asdict(controller),
                        steps=steps,
                        latest_observation=latest_observation,
                        observation_trace=observation_trace,
                        reference_camera_adu=target.camera_adu[:, ::4, ::4],
                        reference_camera_profiles=reference_profiles,
                        reference_fluence_J_m2=plant.reference_result[
                            "output_fluence_J_m2"
                        ],
                        reference_phase_rad=plant.reference_result["output_phase"],
                        reference_slm_illumination=abs(plant.source_field) ** 2,
                        reference_measured_phase_rad=(
                            np.angle(target.phase_field) if target.phase_field is not None else None
                        ),
                        reference_measured_phase_amplitude=(
                            np.abs(target.phase_field) if target.phase_field is not None else None
                        ),
                    )
                ),
            )

    def measured(observation, count):
        nonlocal latest_observation
        timeline = plant.last_result["thermal_timeline"]
        ideal_compensation, delivered_compensation = plant.phase_compensation_truth()
        ledger = plant.last_result["ideal_multipass"]
        requested_time = float(timeline["requested_time_s"])
        opd_pv = float(
            np.interp(
                requested_time, timeline["time_s"], timeline["roundtrip_opd_pv_nm"]
            )
        )
        operating = plant.latest_operating_point
        latest_observation = dict(
            evaluation=count,
            camera_adu=observation.camera_adu[:, ::4, ::4],
            camera_profiles=camera_profiles(observation.camera_adu),
            output_fluence_J_m2=plant.last_result["output_fluence_J_m2"],
            output_phase_rad=plant.last_result["output_phase"],
            measured_energy_J=observation.measured_energy_J,
            measured_probe_temperature_K=observation.probe_temperature_K,
            time_s=observation.time_s,
            valid=observation.valid,
            saturated_fraction=observation.saturated_fraction,
            noise_state=plant.latest_noise_state,
            ideal_compensation_rad=ideal_compensation,
            delivered_slm_compensation_rad=delivered_compensation,
            requested_compensation_rad=plant.correction_phase(
                observation.delivered_coefficients_rad
            ),
            **phase_payload(observation),
        )
        observation_trace.append(
            dict(
                evaluation=count,
                time_s=observation.time_s,
                saturated_fraction=observation.saturated_fraction,
                measured_energy_J=observation.measured_energy_J,
                measured_probe_temperature_K=observation.probe_temperature_K,
                pump_power_W=(
                    operating["pump_power_W"] if operating else episode.pump_W
                ),
                seed_energy_nj=(
                    operating["seed_energy_nj"] if operating else episode.seed_energy_nj
                ),
                coolant_temperature_K=(
                    operating["coolant_temperature_K"]
                    if operating
                    else 273.15 + episode.coolant_setpoint_C
                ),
                slm_calibration_drift_fraction=plant.slm_drift_fraction,
                disk_peak_temperature_C=float(timeline["requested_disk_max_C"]),
                roundtrip_opd_pv_nm=opd_pv,
                thermal_energy_balance_relative_max=float(
                    timeline["energy_balance_relative_max"]
                ),
                population_photon_balance_relative_L1=float(
                    ledger["population_photon_balance_relative_L1"]
                ),
                optical_energy_balance_residual_J=float(
                    ledger["optical_energy_balance_residual_J"]
                ),
                valid=observation.valid,
                camera_mean_adu=float(np.mean(observation.camera_adu)),
                camera_gain_factors=plant.latest_noise_state["camera_gain_factors"],
            )
        )
        publish()

    def record(row, observation):
        result = plant.last_result
        correction = plant.correction_phase(row["coefficients_rad"])
        ideal_compensation, delivered_compensation = plant.phase_compensation_truth(result)
        steps.append(
            dict(
                **row,
                full_solves=plant.full_solves,
                camera_adu=observation.camera_adu[:, ::4, ::4],
                camera_profiles=camera_profiles(observation.camera_adu),
                output_fluence_J_m2=result["output_fluence_J_m2"],
                output_phase_rad=result["output_phase"],
                correction_phase_rad=correction,
                ideal_compensation_rad=ideal_compensation,
                delivered_slm_compensation_rad=delivered_compensation,
                **phase_payload(observation),
                validation_truth=plant.validation_truth(result),
            )
        )
        publish()

    if episode.correction_enabled:
        outcome = run_controller(
            plant,
            target,
            target_energy,
            controller,
            black_level_adu=plant.camera.black_level_adu,
            on_step=record,
            on_observation=measured,
            phase_geometry=(plant.phase_geometry()
                            if controller.method in ("interferometric", "hybrid") else None),
        )
    else:
        zero = (np.zeros(plant.grid.shape) if controller.method in ("interferometric", "hybrid")
                else np.zeros(controller.mode_count))
        uncorrected = plant.observe(zero)
        record(
            dict(
                iteration=0,
                evaluation=1,
                camera_loss=None,
                measured_energy_J=uncorrected.measured_energy_J,
                energy_fraction=uncorrected.measured_energy_J / target_energy,
                time_s=uncorrected.time_s,
                coefficients_rad=zero,
                accepted=True,
            ),
            uncorrected,
        )
        outcome = dict(status="correction_disabled", final_coefficients_rad=zero)
    # Independent held-command measurement for final noisy verification.
    plant.full_solve_limit = None
    final = plant.observe(outcome["final_coefficients_rad"])
    final_truth = plant.validation_truth()
    final_metrics = measurement_metrics(
        final,
        target,
        target_energy,
        controller,
        black_level_adu=plant.camera.black_level_adu,
    )
    pixels_per_waist = episode.waist_mm / (episode.field_size_mm / episode.grid_n)
    return _jsonable(
        dict(
            status=outcome["status"],
            episode=asdict(episode),
            controller=asdict(controller),
            best_recheck_status=outcome.get("best_recheck_status"),
            evaluations=plant.full_solves,
            observations=plant.observations,
            runtime_s=time.perf_counter() - start,
            reference_camera_adu=target.camera_adu[:, ::4, ::4],
            reference_camera_profiles=reference_profiles,
            reference_fluence_J_m2=plant.reference_result["output_fluence_J_m2"],
            reference_phase_rad=plant.reference_result["output_phase"],
            reference_slm_illumination=abs(plant.source_field) ** 2,
            reference_measured_phase_rad=(
                np.angle(target.phase_field) if target.phase_field is not None else None
            ),
            reference_measured_phase_amplitude=(
                np.abs(target.phase_field) if target.phase_field is not None else None
            ),
            reference_energy_J=target_energy,
            steps=steps,
            command_history=plant.command_log,
            observation_trace=observation_trace,
            final_verification_camera_adu=final.camera_adu[:, ::4, ::4],
            final_verification_energy_J=final.measured_energy_J,
            final_verification_measured=final_metrics,
            final_verification_phase=phase_payload(final),
            final_validation_truth=final_truth,
            optical_pixels_per_waist=pixels_per_waist,
            optical_sampling_status=(
                "coarse; refine grid before interpreting field fidelity"
                if pixels_per_waist < 4
                else "not_convergence_certified; compare independent grids"
            ),
            pump_transport_backend=plant.last_result["pump_transport_backend"],
            scope=(
                "Snapshot mode reruns every candidate from the same initial thermal state. "
                "In-situ mode carries disk/plate temperature and five sensor states forward; "
                "SPGD makes sequential noisy physical probes and records regressions as well as improvements. "
                "Interferometric control uses four phase-stepped camera exposures and a nominal "
                "free-space adjoint for pixelwise SLM updates; the disk response is measured, not differentiated. "
                "The amplifier API recomputes periodic population equilibrium for each command. "
                "It does not expose pulse-by-pulse population continuity. Yb:YAG hot gain is unavailable."
            ),
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/ybyag_control.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run(config, progress_path=args.progress)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    print(
        f"{result['status']}: {result['steps'][-1]['iteration']} command updates, "
        f"{result['evaluations']} full-model evaluations, {result['runtime_s']:.1f} s"
    )


if __name__ == "__main__":
    main()
