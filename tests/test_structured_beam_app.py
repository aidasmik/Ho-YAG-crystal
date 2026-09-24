import json
from pathlib import Path

import pytest

import examples.structured_beam_app as app
from examples.structured_beam_app import ROOT, build_gallery_command, validate_request


def test_app_validates_custom_calculation_parameters(tmp_path):
    values=validate_request({
        'solver_mode':'full_seeded_modal',
        'phase_mask':'vortex+1','phase_strength_rad':2.5,
        'density_seed':23,'cluster_count':30,'cluster_contrast':.2,
        'cluster_min_radius_mm':.1,'cluster_max_radius_mm':1.4,
        'post_disk_distance_m':.4})
    command=build_gallery_command(values,ROOT/'results'/'structured_beams'/'test-run')
    assert '--phase-mask' in command
    assert 'vortex+1' in command
    assert '--plots-only' in command
    assert '--density-seed' in command and '23' in command
    assert '--solver-mode' in command and 'full_seeded_modal' in command


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


def test_explicit_budget_renewal_archives_the_exhausted_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(app, 'ROOT', tmp_path)
    ledger_path=tmp_path/'.local_runtime'/'budget.json'
    ledger_path.parent.mkdir()
    old={'schema':1,'total_seconds':7200,'max_attempts':4,'active':None,
         'attempts':[{'category':'coupled','elapsed_s':10.} for _ in range(4)]}
    ledger_path.write_text(json.dumps(old))
    assert app.budget_status()['coupled_exhausted']
    reset=app.start_new_budget()
    assert json.loads(ledger_path.read_text())['attempts']==[]
    assert json.loads(Path(reset['archive']).read_text())==old
    assert not app.budget_status()['coupled_exhausted']
    with pytest.raises(RuntimeError, match='still available'):
        app.start_new_budget()
