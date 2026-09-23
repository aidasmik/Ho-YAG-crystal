import json
import numpy as np
import pytest

from hoyag.snapshots import (ScientificSnapshot, DisplayQueue, snapshot_from_case,
                             save_snapshot,load_snapshot,newest_snapshot)
from hoyag.replay_viewer import scientific_maps
from hoyag.validation_backend import sha256_file


def _metadata():
    return dict(schema_version=1,run_id='fixture',configuration_hash='a',source_hash='b',
        state_id='one',application_mode='oscillator_reference',fidelity_mode='replay',
        solver_status='converged',time_kind='steady_state',optical_state_kind='incoherent_cavity_modes',
        coordinate_frames={'optical':'x/y'},units={},approximation_flags=['replay'],
        modal_powers_W=[2.,3.],t_published_wall=0.)


def test_immutable_incoherent_snapshot_and_display_queue():
    fields=np.ones((2,2,2,2),complex)
    snap=ScientificSnapshot(_metadata(),{'output_coupler_fields':fields,
        'x_m':np.array([0.,1.]),'y_m':np.array([0.,1.])})
    fields[:]=0
    assert np.all(snap.array('output_coupler_fields')==1)
    with pytest.raises(ValueError):snap.array('output_coupler_fields')[0,0,0,0]=0
    maps=scientific_maps(snap)
    assert np.isclose(maps['output_average_irradiance_W_m2'][0].sum(),5.)
    assert 'total_phase' not in maps
    with pytest.raises(ValueError,match='unique total phase'):
        ScientificSnapshot(_metadata(),{'total_phase':np.zeros((2,2))})
    queue=DisplayQueue(2)
    for _ in range(3):queue.publish(snap)
    assert len(queue)==2 and queue.newest() is snap


def test_atomic_snapshot_record_and_checksum(tmp_path):
    metadata=_metadata();metadata['outer_iteration']=1
    snap=ScientificSnapshot(metadata,{'disk_temperature_K':np.ones((1,2,2))*300})
    path=save_snapshot(snap,tmp_path)
    restored=load_snapshot(path)
    assert newest_snapshot(tmp_path).metadata['state_id']=='one'
    np.testing.assert_array_equal(restored.array('disk_temperature_K'),snap.array('disk_temperature_K'))
    (tmp_path/'state_0001.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        load_snapshot(path)


def test_replay_rejects_corrupt_or_unfinished_state(tmp_path):
    np.savez_compressed(tmp_path/'state.npz',fields_used=np.ones((2,2,2,2)))
    summary={'status':'completed','state_sha256':sha256_file(tmp_path/'state.npz'),
             'execution_key':'x','spec_hash':'s','source_hash':'h',
             'field_semantics':'incoherent_cavity_modes'}
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    snap=snapshot_from_case(tmp_path)
    assert snap.metadata['time_kind']=='steady_state'
    assert snap.array('disk_temperature_K') is None
    (tmp_path/'state.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        snapshot_from_case(tmp_path)
