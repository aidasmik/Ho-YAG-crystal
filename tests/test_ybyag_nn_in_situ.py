import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT/'examples'))


def small_model(tmp_path):
    """An untrained V3 model whose training config is a small, fast grid."""
    from ybyag_control import nn_v3 as v3
    config = json.loads((ROOT/'config/ybyag_nn_dataset_10at.json').read_text())
    config['nominal'].update(grid_n=128, signal_traversals=4, thermal_nr=4, thermal_nphi=4,
                             thermal_nz=1)
    spec = v3.Spec.from_config(config, crop=32)
    folder = tmp_path/'model'
    (folder/'member_0').mkdir(parents=True)
    torch.manual_seed(0)
    torch.save(v3.OneShotNet(spec.crop).state_dict(), folder/'member_0'/'weights.pt')
    (folder/'model.json').write_text(json.dumps(dict(schema=v3.SCHEMA, spec=spec.to_json(),
        config=config, members=1, training_complete=True)))
    return folder


@pytest.mark.parametrize('allow', [False, True])
def test_in_situ_nn_episode(tmp_path, allow):
    import importlib.util
    spec = importlib.util.spec_from_file_location('episode_runner', ROOT/'examples'/'ybyag_control.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    config = json.loads((ROOT/'config/ybyag_control_nn_v3.json').read_text())
    config['episode'].update(operation_duration_s=1.0, target='Gaussian TEM00')
    config['controller'].update(nn_model=str(small_model(tmp_path)), nn_device='cpu',
                                nn_allow_uncalibrated=allow, iterations=2, evaluation_limit=12,
                                target_confirmations=1)
    result = runner.run(config)
    assert result['episode']['grid_n'] == 128 and result['episode']['signal_traversals'] == 4
    statuses = [s['update_status'] for s in result['steps']]
    assert statuses[0] == 'baseline'
    assert result['observations'] >= 2
    # Dataset-camera frames: two 1080p planes, exactly like a dataset trial.
    assert np.asarray(result['reference_camera_adu']).shape[0] == 2
    if allow:
        assert any(s in ('nn_applied', 'nn_rejected', 'nn_hold') for s in statuses[1:])
    else:
        assert statuses[1] == 'nn_hold' and result['status'] == 'nn_certified_hold'
        assert all(s.get('nn_status') == 'hold_uncalibrated' for s in result['steps'][1:])
