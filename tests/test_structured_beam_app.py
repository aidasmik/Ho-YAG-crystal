from pathlib import Path

import pytest

from examples.structured_beam_app import ROOT, build_gallery_command, validate_request


def test_app_validates_custom_calculation_parameters(tmp_path):
    values=validate_request({
        'phase_mask':'vortex+1','phase_strength_rad':2.5,
        'density_seed':23,'cluster_count':30,'cluster_contrast':.2,
        'cluster_min_radius_mm':.1,'cluster_max_radius_mm':1.4,
        'post_disk_distance_m':.4})
    command=build_gallery_command(values,ROOT/'results'/'structured_beams'/'test-run')
    assert '--phase-mask' in command
    assert 'vortex+1' in command
    assert '--plots-only' in command
    assert '--density-seed' in command and '23' in command


@pytest.mark.parametrize('payload', [
    {'phase_mask':'unknown'},
    {'phase_mask':'none','cluster_count':1},
    {'phase_mask':'none','cluster_count':2.5},
    {'phase_mask':'none','cluster_min_radius_mm':2,'cluster_max_radius_mm':1},
    {'phase_mask':'none','cluster_contrast':2},
    {'phase_mask':'none','cluster_contrast':'nan'},
])
def test_app_rejects_invalid_parameters(payload):
    with pytest.raises(ValueError):
        validate_request(payload)
