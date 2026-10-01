"""Replay one NN SLM candidate from a saved pre-action Yb:YAG simulator state."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hoyag.propagation import Grid2D, angular_spectrum_propagate
from ybluag.beam_shaping import gaussian_seed_and_target_mask
from ybluag.gallery import simulate_pulsed_seed
from ybluag.regenerative import RegenerativeCavity
from ybyag.model import YbYAGMaterial
from ybyag_dataset.generator import _settings, setup_geometry
from ybyag_dataset.distortions.slm import apply_slm
from ybyag_dataset.field_metrics import field_metrics, phase_support


def replay(trial: Path, config_file: Path, candidate_file: Path) -> dict:
    metadata = json.loads(trial.read_text())
    config = json.loads(config_file.read_text())
    nominal, ranges = config['nominal'], config['ranges']
    pump = metadata['physical_parameters']['pump']
    beam = metadata['physical_parameters']['beam']
    settings = _settings(nominal, pump, beam)
    material = YbYAGMaterial(yb_at_percent=metadata['physical_parameters']['yb_at_percent'])
    grid = Grid2D.square(settings.grid_n, settings.field_size_m)
    x, y = grid.mesh
    wavelength_m = 1030e-9
    with np.load(trial.parent / 'setup.npz') as setup:
        crystal = {key: np.asarray(setup[key], float) for key in (
            'yb_concentration_scale', 'thickness_scale',
            'background_absorption_m1', 'surface_figure_m')}
        contact = np.asarray(setup['contact_scale'], float)
        external = np.asarray(setup['external_optics_phase_rad'], float)
        slm_setup = {'spatial_gain': np.asarray(setup['slm_spatial_gain'], float),
                     'pixel_gain': np.asarray(setup['slm_pixel_gain'], float),
                     'global_gain': float(setup['slm_global_gain']),
                     'bits': int(setup['slm_bits']),
                     'crosstalk_sigma_pixels': float(setup['slm_crosstalk_sigma_pixels']),
                     'phase_offset': np.zeros(grid.shape)}
        lut = np.asarray(setup['slm_phase_lut_rad'], float)
        if lut.size:
            slm_setup['phase_lut_rad'] = lut
        tilts = (np.asarray(setup['multipass_mirror_tilt_rad'], float)
                 if 'multipass_mirror_tilt_rad' in setup.files else None)
        focus = (np.asarray(setup['multipass_relay_defocus_m'], float)
                 if 'multipass_relay_defocus_m' in setup.files else None)
    # The same multipass geometry, pointing and focus errors as the setup.
    settings = _settings(nominal, pump, beam,
                         setup_geometry(nominal, tilts, beam['waist_mm']*1e-3, focus))
    with np.load(trial.parent / metadata['truth_file']) as truth:
        saved_field = np.asarray(truth['truth__complex_field_sqrt_J_m'], complex)
        baseline_slm = np.asarray(truth['truth__slm_actual_phase_rad'], float)
    with np.load(candidate_file) as candidate:
        proposed_command = np.asarray(candidate['proposed_slm_command_rad'], float)
    if proposed_command.shape != grid.shape or not np.all(np.isfinite(proposed_command)):
        raise ValueError('candidate command must be finite on the optical grid')
    elapsed = float(metadata['sequence_time_s']) - float(metadata['thermal_interval_s'])
    proposed_slm, _ = apply_slm(proposed_command, slm_setup,
        drift_fraction=float(ranges['slm_drift_fraction_per_s']) * elapsed)
    group_seed = metadata['setup_seed']
    aperture_rng = np.random.default_rng(int(group_seed[8]) + 1)
    aperture_radius_waists = aperture_rng.uniform(*ranges['seed_aperture_radius_waists'])
    aperture_decenter = aperture_rng.normal(0,
        ranges['seed_aperture_decenter_waist_fraction'] * nominal['waist_mm'] * 1e-3, 2)
    physical = {**crystal, 'contact_scale_polar': contact,
        'coolant_temperature_K': pump['coolant_temperature_K'],
        'pump_center_m': pump['pump_center_m'],
        'seed_center_m': beam['seed_center_m'],
        'seed_angle_rad': beam['seed_angle_rad'],
        'seed_ellipticity': beam['seed_ellipticity'],
        'external_phase_rad': external}
    if config['enabled']['beam']:
        physical['seed_aperture'] = {'radius_m': aperture_radius_waists * settings.waist_m,
                                     'center_m': tuple(aperture_decenter)}
    point = int(metadata['point_index'])
    if point:
        prior = trial.parent / f'point_{point-1:03d}_truth.npz'
        with np.load(prior) as previous:
            physical['initial_disk_temperature_K'] = np.asarray(
                previous['truth__temperature_K'], float)
            physical['initial_plate_temperature_K'] = np.asarray(
                previous['truth__plate_temperature_K'], float)
    architecture = nominal['architecture']
    cavity = (RegenerativeCavity(
        round_trips=int(nominal['regen_round_trips']),
        air_gap_m=nominal['cavity_length_m'],
        mirror_radius_m=nominal['mirror_radius_m'],
        disk_hr_reflectivity=nominal['disk_hr_reflectivity'],
        held_roundtrip_retention=nominal['held_retention'],
        injection_efficiency=nominal['injection_efficiency'],
        extraction_efficiency=nominal['extraction_efficiency'],
        disk_diameter_m=2*settings.disk_radius_m)
        if architecture == 'regenerative' else None)
    def solve(actual_slm):
        state = {key: (value.copy() if isinstance(value, np.ndarray) else value)
                 for key, value in physical.items()}
        state['slm_actual_phase_rad'] = actual_slm.copy()
        return simulate_pulsed_seed(material, settings, metadata['selected_target'],
            beam['seed_energy_nj']*1e-9, nominal['seed_fwhm_ps']*1e-12,
            nominal['repetition_rate_kHz']*1e3, int(nominal['signal_traversals']),
            pump_passes=int(nominal['pump_passes']), compute_thermal=True,
            operation_duration_s=float(metadata['thermal_interval_s']),
            cooling_mode='fixed', thermal_optical_mode='lumped_phase',
            architecture=architecture, regenerative_cavity=cavity,
            dataset_physical=state)
    baseline = solve(baseline_slm)
    baseline_field = np.asarray(baseline['output_complex_field_sqrt_J_m'], complex)
    reproduction_error = float(np.linalg.norm(baseline_field-saved_field) /
                               max(np.linalg.norm(saved_field), 1e-30))
    if reproduction_error > 1e-4:
        raise RuntimeError(f'baseline replay disagrees with saved field: {reproduction_error}')
    proposed = solve(proposed_slm)
    proposed_field = np.asarray(proposed['output_complex_field_sqrt_J_m'], complex)
    if not (proposed['thermal_feedback_applied'] and
            proposed['thermal_timeline']['requested_material_range_valid']):
        raise RuntimeError('candidate thermal state left the supported range')
    _, target_phase = gaussian_seed_and_target_mask(grid, settings.waist_m, 1.,
        metadata['selected_target'], wavelength_m, settings.slm_to_disk_distance_m)
    source = np.exp(-(x*x+y*y)/settings.waist_m**2)
    source /= np.sqrt(np.sum(abs(source)**2)*grid.dx*grid.dy)
    desired = angular_spectrum_propagate(source*np.exp(1j*target_phase),
        grid, wavelength_m, settings.slm_to_disk_distance_m)
    if settings.post_disk_distance_m:
        desired = angular_spectrum_propagate(desired, grid, wavelength_m,
                                              settings.post_disk_distance_m)
    weights = abs(baseline_field)**2
    support = phase_support(baseline_field, desired)
    before = field_metrics(baseline_field, desired, weights, support)
    after = field_metrics(proposed_field, desired, weights, support)
    if (abs(before.coherent_fidelity-metadata['coherent_fidelity']['uncorrected']) > 1e-5 or
            abs(before.shape_overlap-metadata['shape_overlap']['uncorrected']) > 1e-5):
        raise RuntimeError('replayed baseline metrics disagree with saved metadata')
    energy_fraction = after.output_norm / before.output_norm
    thresholds = metadata['correction_label_thresholds']
    shape_limit = max(thresholds['min_shape_overlap'],
                      before.shape_overlap-thresholds['max_shape_drop'])
    return {'trial': str(trial),
        'trial_sha256': hashlib.sha256(trial.read_bytes()).hexdigest(),
        'candidate_sha256': hashlib.sha256(candidate_file.read_bytes()).hexdigest(),
        'target': metadata['selected_target'],
        'yb_at_percent': metadata['physical_parameters']['yb_at_percent'],
        'point_index': point, 'teacher_label_valid': metadata['correction_label_valid'],
        'teacher_method': metadata['correction_label_method'],
        'teacher_corrected_fidelity': metadata['coherent_fidelity']['corrected'],
        'baseline_reproduction_relative_l2': reproduction_error,
        'baseline': {'fidelity': before.coherent_fidelity,
                     'shape': before.shape_overlap, 'phase_rms_rad': before.phase_rms_rad,
                     'output_energy_J': baseline['output_energy_J']},
        'candidate': {'fidelity': after.coherent_fidelity,
                      'shape': after.shape_overlap, 'phase_rms_rad': after.phase_rms_rad,
                      'output_energy_J': proposed['output_energy_J'],
                      'energy_fraction': energy_fraction},
        'fidelity_gain': after.coherent_fidelity-before.coherent_fidelity,
        'shape_change': after.shape_overlap-before.shape_overlap,
        'task_success': bool(after.coherent_fidelity >= thresholds['target_fidelity'] and
                             after.shape_overlap >= shape_limit and
                             energy_fraction >= thresholds['min_energy_fraction']),
        'safe_shape_energy': bool(after.shape_overlap >= shape_limit and
                                  energy_fraction >= thresholds['min_energy_fraction']),
        'source_config_sha256': hashlib.sha256(config_file.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial_json', type=Path)
    parser.add_argument('config', type=Path)
    parser.add_argument('candidate_npz', type=Path)
    parser.add_argument('output_json', type=Path)
    args = parser.parse_args()
    result = replay(args.trial_json, args.config, args.candidate_npz)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
