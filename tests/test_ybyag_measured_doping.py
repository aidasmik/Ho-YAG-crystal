import json
from pathlib import Path

import numpy as np
import pytest

from hoyag.propagation import Grid2D
from ybyag_dataset.distortions.material import sample_material
from ybyag_dataset.distortions.measured_doping import provenance, relative_map, sample_for

CONFIG = json.loads((Path(__file__).resolve().parents[1]/'config/ybyag_nn_dataset.json').read_text())
GRID = Grid2D.square(128, 12e-3)


def test_measured_maps_are_the_dataset_default_and_traceable():
    assert CONFIG['ranges']['yb_distribution'] == 'measured'
    info = provenance()
    assert set(info['samples']) == {'5', '10', '15'}
    assert 'not a calibrated' in info['status'].lower()


@pytest.mark.parametrize('at', [5, 10, 15])
def test_measured_map_is_relative_smooth_and_fades_to_one(at):
    m = relative_map(GRID, at)
    x, y = GRID.mesh
    r = np.hypot(x, y)
    assert np.all(np.isfinite(m))
    assert abs(np.median(m[r < 1.5e-3])-1) < .03              # near the sample median
    assert np.max(abs(m[r > 5.8e-3]-1)) < 1e-3                  # fades outside the map
    assert .8 < m.min() and m.max() < 1.2                       # boundary ring trimmed
    assert np.std(m[r < 1.2e-3]) > 1e-3                         # real structure under the beam


def test_placement_is_seeded_and_scale_is_explicit():
    ranges = CONFIG['ranges']
    a = sample_material(GRID.shape, ranges, 1, grid=GRID, yb_at_percent=5)['yb_concentration_scale']
    b = sample_material(GRID.shape, ranges, 1, grid=GRID, yb_at_percent=5)['yb_concentration_scale']
    c = sample_material(GRID.shape, ranges, 2, grid=GRID, yb_at_percent=5)['yb_concentration_scale']
    assert np.array_equal(a, b) and not np.allclose(a, c)
    doubled = relative_map(GRID, 5, deviation_scale=2.)
    single = relative_map(GRID, 5, deviation_scale=1.)
    # Default sign: higher 969/1030 ratio means less Yb (969 nm reabsorption).
    assert np.allclose(relative_map(GRID, 5)-1, -1.7*(single-1))
    assert np.allclose(doubled-1, 2*(single-1))
    with pytest.raises(ValueError):
        sample_material(GRID.shape, ranges, 1)                   # needs grid and concentration
    with pytest.raises(ValueError):
        sample_for(20.)                                         # no 20 at.% measurement
    assert sample_for(20., nearest=True) == 15.


def test_controller_and_desktop_use_measured_map():
    from ybyag_control.adapter import EpisodeConfig, SimulationPlant
    plant = SimulationPlant(EpisodeConfig(grid_n=32, camera_width=128, camera_height=72,
                                          yb_at_percent=10.), mode_count=1)
    expected = relative_map(plant.grid, 10., offset_m=plant_offset(plant), rotation_rad=plant_rotation(plant))
    assert np.allclose(plant.material_maps['yb_concentration_scale'], expected)
    random = SimulationPlant(EpisodeConfig(grid_n=32, camera_width=128, camera_height=72,
                                           yb_distribution='random'), mode_count=1)
    assert not np.allclose(random.material_maps['yb_concentration_scale'], expected)
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]/'examples'))
    from ybluag_app import calculate_pulsed
    request = dict(material='Yb:YAG', selected_beam='Gaussian TEM00', phase_mask='none',
                   assembly_property_model='yag_rt_proxy', yb_at_percent=20, pump_W=.1,
                   radius_mm=1., waist_mm=.6, thickness_um=100, grid_n=48, field_size_mm=12,
                   signal_traversals=4, operation_duration_s=0, thermal_optical_mode='cold',
                   cooling_mode='fixed', distance_m=0, multipass_layout='ideal_relay')
    measured = calculate_pulsed(dict(request), compute_thermal=False)
    assert measured['yb_distribution'].startswith('measured') and '15 at.%' in measured['yb_distribution']
    clusters = calculate_pulsed(dict(request, yb_distribution='random_clusters'), compute_thermal=False)
    assert clusters['yb_distribution'] == 'seeded random clusters'
    assert measured['output_energy_J'] != clusters['output_energy_J']


def plant_offset(plant):
    return _placement(plant)[0]


def plant_rotation(plant):
    return _placement(plant)[1]


def _placement(plant):
    from ybyag_dataset.distortions.common import child_seeds
    from ybyag_dataset.distortions.measured_doping import sampled_placement
    seed = child_seeds(plant.episode.seed+1, 4)[0]
    return sampled_placement(plant.ranges, np.random.default_rng(seed))
