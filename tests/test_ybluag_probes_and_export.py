import copy
from dataclasses import replace

import numpy as np
import pytest

from hoyag.thermal import DiskThermalMesh
from ybluag.sensors import TemperatureProbe, ProbeArray, default_five_probes
from ybluag.research_export import (REQUIRED_REFINEMENTS, REQUIRED_SHAPES,
                                    REQUIRED_METRICS,
                                    check_convergence_evidence,
                                    export_research_sample, source_fingerprint)
from ybluag.gallery import (YbGallerySettings, _assembly_configuration,
                            _pulsed_thermal_timeline)
from ybluag.model import YbLuAGMaterial
from hoyag.propagation import Grid2D


def test_five_probes_use_physical_fields_response_and_missing_status():
    disk = DiskThermalMesh.disk(nr=4, nphi=4, nz=2, radius_m=.005,
                                thickness_m=.0001)
    plate = DiskThermalMesh.disk(nr=4, nphi=4, nz=2, radius_m=.01,
                                 thickness_m=.002)
    probes = list(default_five_probes(.0001))
    assert all(p.position_m[2] == 0 for p in probes if p.region == "disk")
    probes[0] = TemperatureProbe("disk_center", "disk", (0, 0, .00005),
                                  response_time_s=1, sampling_rate_Hz=2,
                                  latency_s=.2, bias_K=.3)
    probes[4] = TemperatureProbe("plate_x", "plate", (.003, 0, .0006),
                                  sampling_rate_Hz=2, missing=True)
    adapter = ProbeArray(probes, disk, plate, initial_temperature_K=293.15,
                         seed=123)
    initial = adapter.initial_samples(np.full(disk.shape, 293.15),
                                      np.full(plate.shape, 293.15))
    assert len(initial) == 5
    assert next(x for x in initial if x["name"] == "disk_center")["available_time_s"] == .2
    events = adapter.advance(1, np.full(disk.shape, 303.15),
                             np.full(plate.shape, 298.15))
    center = [x for x in events if x["name"] == "disk_center"]
    assert len(center) == 2
    assert center[-1]["local_average_K"] == pytest.approx(303.15)
    assert center[-1]["noiseless_response_K"] == pytest.approx(
        293.15 + 10*np.exp(-1))
    assert center[0]["local_average_K"] == pytest.approx(298.15)
    assert center[-1]["measured_K"] == pytest.approx(
        center[-1]["noiseless_response_K"] + .3)
    assert next(x for x in events if x["name"] == "plate_x")["measured_K"] is None
    assert all(x["valid"] is False for x in events if x["name"] == "plate_x")


def test_sensor_controller_cannot_use_hidden_disk_maximum():
    settings = YbGallerySettings(thickness_m=100e-6, thermal_nr=4,
                                 thermal_nphi=4, thermal_nz=1,
                                 grid_n=32, field_size_m=.012)
    mesh = DiskThermalMesh.disk(nr=4, nphi=4, nz=1, radius_m=.005,
                                thickness_m=100e-6)
    grid = Grid2D.square(32, .012)
    config = _assembly_configuration(YbLuAGMaterial(), settings)
    heat = np.full(mesh.shape, 1e9)
    probes = tuple(replace(p, missing=True) if p.region == "disk" else p
                   for p in default_five_probes(100e-6))
    sensor = _pulsed_thermal_timeline(
        mesh, heat, grid, config, .1, cooling_mode="sensor_feedback",
        cooling_target_C=20, internal_max_step_s=1, temperature_probes=probes)
    fixed = _pulsed_thermal_timeline(
        mesh, heat, grid, config, .1, cooling_mode="fixed",
        cooling_target_C=20, internal_max_step_s=1, temperature_probes=probes)
    np.testing.assert_allclose(sensor["coolant_conductance_W_m2K"],
                               fixed["coolant_conductance_W_m2K"])
    assert "only delivered" in sensor["temperature_probes"]["scope"]
    biased = tuple(replace(p, bias_K=5) if p.region == "disk" else p
                   for p in default_five_probes(100e-6))
    sensor_bias = _pulsed_thermal_timeline(
        mesh, np.zeros(mesh.shape), grid, config, .1,
        cooling_mode="sensor_feedback", cooling_target_C=20,
        internal_max_step_s=1, temperature_probes=biased)
    assert np.max(sensor_bias["coolant_conductance_W_m2K"]) > config[
        "thermal"]["coolant_conductance_W_m2K"]


def _evidence():
    return {axis: {"status": "passed", "relative_error": .001,
                   "configuration_pair": ["coarse", "fine"],
                   "shape_metric_errors": {
                       shape: {metric: .001 for metric in REQUIRED_METRICS}
                       for shape in REQUIRED_SHAPES}}
            for axis in REQUIRED_REFINEMENTS}


def _matching_evidence(configuration):
    import hashlib
    import json
    result = _evidence()
    result["configuration_sha256"] = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, allow_nan=False).encode()).hexdigest()
    result["source_sha256"] = source_fingerprint()
    return result


def test_export_gate_rejects_unresolved_and_unavailable_physics(tmp_path):
    thresholds = {axis: .01 for axis in REQUIRED_REFINEMENTS}
    evidence = _matching_evidence({})
    check_convergence_evidence(evidence, thresholds)
    missing = copy.deepcopy(evidence)
    del missing["optical_pitch"]
    with pytest.raises(ValueError, match="optical_pitch"):
        check_convergence_evidence(missing, thresholds)
    with pytest.raises(ValueError, match="configuration and source"):
        export_research_sample(tmp_path/"stale", result={"thermal_optical_mode": "cold"},
                               configuration={"changed": True}, evidence=evidence,
                               thresholds=thresholds, slm_command=np.zeros((2, 2)),
                               intended_target_field=np.ones((2, 2)),
                               random_seeds={}, material_version="test")
    result = {"thermal_optical_mode": "cold"}
    with pytest.raises(ValueError, match="coupled"):
        export_research_sample(tmp_path/"sample", result=result,
                               configuration={}, evidence=evidence,
                               thresholds=thresholds,
                               slm_command=np.zeros((2, 2)),
                               intended_target_field=np.ones((2, 2)),
                               random_seeds={}, material_version="test")
    assert not (tmp_path/"sample.npz").exists()


def test_export_rejects_transient_pairing_and_unvalidated_correction(tmp_path):
    thresholds = {axis: .01 for axis in REQUIRED_REFINEMENTS}
    result = {"thermal_optical_mode": "coupled_steady",
              "thermal_feedback_applied": True,
              "coupled_steady_convergence": {"status": "converged"},
              "thermal": {"validity": "extrapolated_unvalidated",
                          "scalar_roundtrip_opd_nm": [[0]],
                          "front_displacement_nm": [[0]],
                          "rear_displacement_nm": [[0]],
                          "disk_temperature_K": [[293.15]],
                          "plate_temperature_K": [[293.15]]},
              "spectral_gain_screen": {"status": "not_calculated"},
              "thermal_timeline": {"disk_max_C": [20, 21],
                                   "temperature_probes": {"samples": []}},
              "scope": "fixture"}
    for key in ("input_fluence_J_m2", "disk_input_fluence_J_m2",
                "output_fluence_J_m2", "output_phase", "yb_density_m3",
                "pre_pulse_excited_fraction_by_slice", "optical_temperature_K_by_slice"):
        result[key] = np.ones((2, 2))
    with pytest.raises(ValueError, match="transient dataset export unavailable"):
        export_research_sample(
            tmp_path/"transient", result=result, configuration={},
            evidence=_matching_evidence({}), thresholds=thresholds,
            slm_command=np.zeros((2, 2)),
            intended_target_field=np.ones((2, 2)), random_seeds={},
            material_version="fixture", solver_commit="test", dataset_mode="transient")
    with pytest.raises(ValueError, match="settled same-state"):
        export_research_sample(
            tmp_path/"sample", result=result, configuration={},
            evidence=_matching_evidence({}), thresholds=thresholds,
            slm_command=np.zeros((2, 2)),
            intended_target_field=np.ones((2, 2)), random_seeds={},
            material_version="fixture", solver_commit="test")


def test_export_separates_measured_inputs_and_hidden_truth(tmp_path):
    base = np.array([[1, 1], [1, -1]], complex)
    target = np.ones((2, 2), complex)
    result = {
        "architecture": "ideal_multipass", "selected_beam": "Gaussian TEM00",
        "input_energy_J": 1e-9, "pump_passes": 2,
        "thermal_optical_mode": "coupled_steady", "thermal_feedback_applied": True,
        "coupled_steady_convergence": {"status": "converged"},
        "thermal": {"validity": "extrapolated_unvalidated",
                    "scalar_roundtrip_opd_nm": np.zeros((2, 2)),
                    "front_displacement_nm": np.zeros((2, 2)),
                    "rear_displacement_nm": np.zeros((2, 2)),
                    "disk_temperature_K": np.full((1, 2, 2), 294.),
                    "plate_temperature_K": np.full((1, 2, 2), 293.5)},
        "steady_state_observation": {
            "mode": "steady_state", "state_timestamp_s": 0., "probe_seed": 77,
            "beam_timestamp_s": 0., "slm_target_timestamp_s": 0.,
            "probe_specifications": [{"name": str(i)} for i in range(5)],
            "probe_samples": [{"name": str(i), "sample_time_s": 0.,
                               "measured_K": 294., "valid": True} for i in range(5)]},
        "output_complex_field_sqrt_J_m": base,
        "output_fluence_J_m2": abs(base)**2,
    }
    for key in ("input_fluence_J_m2", "disk_input_fluence_J_m2",
                "yb_density_m3", "pre_pulse_excited_fraction_by_slice",
                "optical_temperature_K_by_slice", "heat_W_m3_by_slice",
                "static_cold_phase_rad"):
        result[key] = np.ones((2, 2))
    corrected = dict(result, output_complex_field_sqrt_J_m=target)
    args = dict(result=result, configuration={"pump_power_W": 1.},
                evidence=_matching_evidence({"pump_power_W": 1.}),
                thresholds={axis: .01 for axis in REQUIRED_REFINEMENTS},
                slm_command=np.zeros((2, 2)), intended_target_field=target,
                random_seeds={"camera": 123, "probe": 77}, material_version="fixture",
                solver_commit="test", correction_phase_rad=np.ones((2, 2)),
                forward_model=lambda _: corrected)
    paths = export_research_sample(tmp_path/"sample", **args)
    with np.load(paths[0]) as arrays:
        assert "observable__measured_output_fluence_J_m2" in arrays
        assert "truth__output_complex_field_sqrt_J_m" in arrays
        assert "observable__optical_temperature_K_by_slice" not in arrays
    import json
    metadata = json.loads(paths[1].read_text())
    assert metadata["physical_state_timestamp_s"] == 0
    assert metadata["correction_validation"]["corrected_coherent_overlap"] > .99
    bad = copy.deepcopy(result)
    bad["steady_state_observation"]["probe_samples"][0]["sample_time_s"] = 1
    with pytest.raises(ValueError, match="same physical state"):
        export_research_sample(tmp_path/"bad", **dict(args, result=bad))
