import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hoyag.propagation import Grid2D
from ybluag.gallery import simulate_pulsed_seed
from ybluag.multipass_geometry import (dataset_defaults, from_nominal, image_relay, mirror_array,
                                       sample_mirror_tilts, sample_relay_defocus, uniform)
from ybyag.model import YbYAGMaterial
from ybyag_dataset.distortions.material import sample_material
from ybyag_dataset.generator import _settings, setup_geometry
from ybyag_dataset.modal_teacher import FrozenOpticalModel

CONFIG = json.loads((Path(__file__).resolve().parents[1]/'config/ybyag_nn_dataset.json').read_text())
NOMINAL = CONFIG['nominal']
PASSES = int(NOMINAL['signal_traversals'])
MIRROR_ARRAY = dict(NOMINAL, signal_traversals=10, multipass=dict(
    layout='mirror_array', array_distance_m=.5, array_half_height_m=.08,
    array_mirror_incidence_deg=1.5, mirror_focal_m=None))


def test_doping_map_respects_hard_peak_bound():
    ranges = dict(CONFIG['ranges'], yb_distribution='random')
    for seed in range(5):
        for stress in (1., 1.5):
            yb = sample_material((96, 96), ranges, seed, stress=stress)['yb_concentration_scale']
            assert np.max(abs(yb-1)) == pytest.approx(ranges['yb_concentration_max_fraction'])
    legacy = {k: v for k, v in ranges.items() if k != 'yb_concentration_max_fraction'}
    yb = sample_material((96, 96), legacy, 0)['yb_concentration_scale']
    assert np.std(yb) == pytest.approx(legacy['yb_concentration_rms_fraction'], rel=1e-6)


def test_default_is_a_24_pass_image_relay():
    assert PASSES == 24
    geometry = from_nominal(NOMINAL, PASSES, waist_m=.6e-3)
    assert geometry.layout == 'image_relay' and geometry.encounters == 24
    # 24 traversals = 12 disk bounces: HR reflection inside a bounce, relay between.
    kinds = [p.kind for p in geometry.paths]
    assert kinds == ['reflection', 'image']*11+['reflection']
    assert geometry.summary()['disk_bounces'] == 12
    angles = np.degrees(geometry.incidence_rad)
    assert angles[0] == angles[1] == pytest.approx(NOMINAL['multipass']['max_incidence_deg'])
    assert len(set(np.round(angles, 9))) == 12          # every bounce its own angle
    grid = Grid2D.square(64, 6e-3)
    rng = np.random.default_rng(0)
    field = rng.normal(size=grid.shape)+1j*rng.normal(size=grid.shape)
    other = rng.normal(size=grid.shape)+1j*rng.normal(size=grid.shape)
    # The HR reflection inside a bounce is the identity; an ideal 4f relay
    # point-inverts the image and two relays restore it.
    assert np.allclose(geometry.relay(0, field, grid, 1030e-9), field)
    once = geometry.relay(1, field, grid, 1030e-9)
    assert np.allclose(once, field[::-1, ::-1])
    assert np.allclose(geometry.relay(3, once, grid, 1030e-9), field)
    errors = image_relay(4, relay_focal_m=.5, tilt_rad=np.full((3, 2), 5e-6),
                         defocus_m=sample_relay_defocus(4, .5, 1))
    assert errors.image_shift_m(0) == (0., 0.)                 # inside a bounce
    assert errors.image_shift_m(1) == pytest.approx((5e-6, 5e-6))
    out = errors.relay(1, field, grid, 1030e-9)
    assert np.linalg.norm(out) == pytest.approx(np.linalg.norm(field))
    assert np.vdot(other, out) == pytest.approx(np.vdot(errors.relay_adjoint(1, other, grid, 1030e-9), field))


def test_mirror_array_gives_every_pass_its_own_path():
    geometry = from_nominal(MIRROR_ARRAY, 10, waist_m=.6e-3)
    lengths = [p.length_m for p in geometry.paths if p.kind == 'free_space']
    assert geometry.layout == 'mirror_array' and geometry.encounters == 10
    assert len(lengths) == 4                                      # 5 bounces, 4 returns
    assert len(set(np.round(geometry.incidence_rad, 9))) == 3      # symmetric fan
    assert max(lengths)-min(lengths) > 5e-3 and min(lengths) >= 1.
    grid = Grid2D.square(128, 12e-3)
    x, _ = grid.mesh
    seen = geometry.encounter_map(0, x, grid)
    theta = geometry.incidence_rad[0]
    inner = abs(x) < 4e-3
    scale = 1/math.cos(math.asin(math.sin(theta)/geometry.index))
    assert np.allclose(seen[inner], x[inner]/math.cos(theta)*scale, atol=1e-9)
    tilts = sample_mirror_tilts(10, 5., 1)
    assert tilts.shape == (9, 2) and 1e-6 < np.std(tilts) < 1e-5


def _solve(nominal, geometry=None, distance=0., focal=0., grid_n=96):
    nominal = dict(nominal, grid_n=grid_n)
    pump = dict(pump_W=nominal['pump_W'], radius_mm=nominal['radius_mm'],
                coolant_temperature_K=nominal['coolant_temperature_C']+273.15,
                pump_center_m=(0., 0.))
    beam = dict(seed_energy_nj=nominal['seed_energy_nj'], waist_mm=nominal['waist_mm'])
    settings = replace(_settings(nominal, pump, beam, geometry), inter_pass_distance_m=distance,
                       inter_pass_focal_m=focal)
    grid = Grid2D.square(grid_n, settings.field_size_m)
    crystal = sample_material(grid.shape, CONFIG['ranges'], 3, grid=grid, yb_at_percent=10.)
    physical = {**crystal, 'slm_actual_phase_rad': np.zeros(grid.shape),
                'external_phase_rad': np.zeros(grid.shape)}
    result = simulate_pulsed_seed(YbYAGMaterial(yb_at_percent=10.), settings, 'Gaussian TEM00',
        beam['seed_energy_nj']*1e-9, nominal['seed_fwhm_ps']*1e-12,
        nominal['repetition_rate_kHz']*1e3, int(nominal['signal_traversals']), pump_passes=10,
        compute_thermal=True, operation_duration_s=1., cooling_mode='fixed',
        thermal_optical_mode='lumped_phase', thermal_timeline_mode='requested_only',
        dataset_physical=physical)
    return grid, settings, result


def _passive(grid, settings, result, geometry):
    wavelength = 1030e-9
    n = result['effective_signal_traversals']
    screen = (n*np.asarray(result['static_cold_phase_rad'], float) +
              np.pi*n/wavelength*np.asarray(result['thermal_timeline']['final_roundtrip_opd_m'], float))
    zero = np.zeros(grid.shape)
    return FrozenOpticalModel(grid=grid, wavelength_m=wavelength,
        slm_to_disk_m=settings.slm_to_disk_distance_m, output_distance_m=settings.post_disk_distance_m,
        input_field=result['input_complex_field_sqrt_J_m'], current_command=zero,
        actual_slm_phase=zero, external_phase=zero, screen_phase=screen,
        baseline_field=result['output_complex_field_sqrt_J_m'],
        slm_setup=dict(global_gain=1., spatial_gain=np.ones(grid.shape), pixel_gain=np.ones(grid.shape),
                       phase_offset=zero, bits=8, crosstalk_sigma_pixels=.35), geometry=geometry)


@pytest.mark.parametrize('nominal,layout', [(NOMINAL, 'image_relay'), (MIRROR_ARRAY, 'mirror_array')])
def test_layout_with_errors_conserves_energy_and_matches_teacher(nominal, layout):
    passes = int(nominal['signal_traversals'])
    tilts = sample_mirror_tilts(passes, CONFIG['ranges']['mirror_tilt_error_urad'], 5)
    focus = sample_relay_defocus(passes, CONFIG['ranges']['relay_defocus_error_mm'], 6)
    geometry = setup_geometry(nominal, tilts, nominal['waist_mm']*1e-3, focus)
    assert geometry.layout == layout
    grid, settings, result = _solve(nominal, geometry)
    ideal = result['ideal_multipass']
    assert ideal['ideal_relay'] == 'free_space_between_encounters'
    assert ideal['thermal_phase_per_encounter']
    assert ideal['multipass_geometry']['layout'] == layout
    assert len(ideal['encounter_exit_energies_J']) == passes
    scale = result['output_energy_J']
    assert abs(ideal['optical_energy_balance_residual_J']) < 1e-9*scale
    assert abs(ideal['ideal_relay_loss_J']) < 1e-6*scale
    assert _passive(grid, settings, result, geometry).baseline_overlap > .9999
    # Ignoring the per-pass geometry no longer describes the solver output.
    assert _passive(grid, settings, result, None).baseline_overlap < _passive(
        grid, settings, result, geometry).baseline_overlap


def test_uniform_path_and_ideal_relay_remain_available():
    short = dict(NOMINAL, signal_traversals=10)
    short.pop('multipass')
    geometry = uniform(10, .25, 4.89)
    grid, settings, result = _solve(short, distance=.25, focal=4.89)
    assert result['ideal_multipass']['multipass_geometry']['layout'] == 'uniform'
    assert _passive(grid, settings, result, geometry).baseline_overlap > .9999
    grid, settings, result = _solve(short)
    ideal = result['ideal_multipass']
    assert ideal['ideal_relay'] == 'unit_magnification_phase_preserving'
    assert not ideal['thermal_phase_per_encounter']
    assert _passive(grid, settings, result, None).baseline_overlap > .9999


@pytest.mark.parametrize('layout', ['image_relay', 'mirror_array'])
def test_v3_optics_matches_teacher_operator(layout):
    torch = pytest.importorskip('torch')
    from ybyag_control import nn_v3 as v3
    nominal = NOMINAL if layout == 'image_relay' else MIRROR_ARRAY
    passes = int(nominal['signal_traversals'])
    spec = v3.Spec(slm_n=96, field_m=12e-3, factor=1, crop=48, encounters=passes,
                   multipass=nominal['multipass'])
    optics = v3.Optics(spec, 'cpu', 'double')
    grid = spec.opt_grid()
    x, y = grid.mesh
    amp = np.exp(-(x*x+y*y)/spec.waist_m**2)
    screen = .8*np.exp(-(x*x+y*y)/(1e-3)**2)+.1*x/1e-3
    tilts = sample_mirror_tilts(passes, 5., 2)
    geometry = from_nominal(nominal, passes, waist_m=spec.waist_m, tilt_rad=tilts)
    zero = np.zeros(grid.shape)
    teacher = FrozenOpticalModel(grid=grid, wavelength_m=v3.WAVELENGTH_M, slm_to_disk_m=.25,
        output_distance_m=0., input_field=amp.astype(complex), current_command=zero,
        actual_slm_phase=zero, external_phase=zero, screen_phase=screen, baseline_field=amp,
        slm_setup={}, geometry=geometry)
    reference = teacher._operator(amp.astype(complex))
    t = optics.tensor
    out = optics.forward(t(amp)[None], torch.ones((1, *grid.shape), dtype=torch.complex128),
                         disk_phase=t(screen)[None], planes=False,
                         tilts=t(tilts)[None])['output'][0].numpy()
    assert np.max(abs(out-reference)) < 1e-10*np.max(abs(reference))


def test_controller_plant_uses_dataset_layout_by_default():
    from ybyag_control.adapter import EpisodeConfig, SimulationPlant
    defaults = dataset_defaults()
    # Two bounces (four traversals) so that one relay with errors exists.
    plant = SimulationPlant(EpisodeConfig(grid_n=32, camera_width=128, camera_height=72,
                                          signal_traversals=4), mode_count=1)
    geometry = plant.settings.multipass_geometry
    assert geometry.layout == defaults['multipass_layout'] == 'image_relay'
    assert geometry.encounters == plant.episode.signal_traversals
    tilts = np.array([p.tilt_rad for p in geometry.paths])
    focus = np.array([p.defocus_m for p in geometry.paths])
    assert np.any(tilts != 0) and np.any(focus != 0)
    # The ideal reference has the same relays without errors.
    ideal = plant.ideal_settings.multipass_geometry
    assert ideal.incidence_rad == geometry.incidence_rad
    assert all(p.tilt_rad == (0., 0.) and p.defocus_m == 0 for p in ideal.paths)
    relay = SimulationPlant(EpisodeConfig(grid_n=32, camera_width=128, camera_height=72,
                                          multipass_layout='ideal_relay'), mode_count=1)
    assert relay.settings.multipass_geometry is None
    array = SimulationPlant(EpisodeConfig(grid_n=32, camera_width=128, camera_height=72,
                                          multipass_layout='mirror_array'), mode_count=1)
    assert array.settings.multipass_geometry.layout == 'mirror_array'


def test_desktop_pulsed_yag_uses_dataset_layout_by_default():
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]/'examples'))
    from ybluag_app import calculate_pulsed
    request = dict(material='Yb:YAG', selected_beam='Gaussian TEM00', phase_mask='none',
                   assembly_property_model='yag_rt_proxy', yb_at_percent=10, pump_W=.1,
                   radius_mm=1., waist_mm=.6, thickness_um=100, grid_n=64, field_size_mm=12,
                   signal_traversals=24, operation_duration_s=0, thermal_optical_mode='cold',
                   cooling_mode='fixed', cluster_contrast=0, distance_m=0)
    result = calculate_pulsed(dict(request), compute_thermal=False)
    summary = result['ideal_multipass']['multipass_geometry']
    assert summary['layout'] == 'image_relay' and len(summary['incidence_deg']) == 24
    relay = calculate_pulsed(dict(request, multipass_layout='ideal_relay'), compute_thermal=False)
    assert relay['ideal_multipass']['multipass_geometry'] is None


def test_v3_per_sample_relay_focus_errors_match_teacher():
    torch = pytest.importorskip('torch')
    from ybyag_control import nn_v3 as v3
    passes = 6
    nominal = dict(NOMINAL, signal_traversals=passes)
    spec = v3.Spec(slm_n=96, field_m=12e-3, factor=1, crop=48, encounters=passes,
                   multipass=nominal['multipass'])
    optics = v3.Optics(spec, 'cpu', 'double')
    grid = spec.opt_grid()
    x, y = grid.mesh
    amp = np.exp(-(x*x+y*y)/spec.waist_m**2)
    screen = .8*np.exp(-(x*x+y*y)/(1e-3)**2)+.1*x/1e-3
    tilts = sample_mirror_tilts(passes, 5., 3)
    focus = sample_relay_defocus(passes, 2., 4)            # exaggerated, 2 mm RMS
    geometry = from_nominal(nominal, passes, waist_m=spec.waist_m, tilt_rad=tilts, defocus_m=focus)
    zero = np.zeros(grid.shape)
    teacher = FrozenOpticalModel(grid=grid, wavelength_m=v3.WAVELENGTH_M, slm_to_disk_m=.25,
        output_distance_m=0., input_field=amp.astype(complex), current_command=zero,
        actual_slm_phase=zero, external_phase=zero, screen_phase=screen, baseline_field=amp,
        slm_setup={}, geometry=geometry)
    reference = teacher._operator(amp.astype(complex))
    t = optics.tensor
    out = optics.forward(t(amp)[None], torch.ones((1, *grid.shape), dtype=torch.complex128),
                         disk_phase=t(screen)[None], planes=False, tilts=t(tilts)[None],
                         defocus=t(focus)[None])['output'][0].numpy()
    assert np.max(abs(out-reference)) < 1e-9*np.max(abs(reference))
