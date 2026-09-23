from copy import deepcopy
import numpy as np
import pytest

from hoyag.checkpoints import save_checkpoint,load_checkpoint


def _case():
    return {'spec_hash':'configuration-one','numerics':{'mechanical':{'nr':2,'ntheta':8}}}


def _manifest():
    return {'source_hash':'source-one','git_revision':'revision-one',
            'python':'3.12','numpy':'1.26','scipy':'1.17','platform':'linux'}


def _arrays():
    return {'x_m':np.array([0.,1.]),'y_m':np.array([0.,1.]),
            'r_edges_m':np.array([0.,.5,1.]),'z_edges_m':np.array([0.,1.]),
            'fields_next':np.ones((2,2,2,2),complex),
            'initial_log_photons':None}


def test_checkpoint_exact_resume_and_incompatibility(tmp_path):
    info={'iteration':1,'streak':0,'history':[{'iteration':1}], 'branch_ids':[0,1]}
    path=save_checkpoint(tmp_path,_case(),_manifest(),info,_arrays())
    loaded=load_checkpoint(tmp_path,_case(),_manifest())
    assert loaded['iteration']==1 and loaded['branch_ids']==[0,1]
    assert loaded['initial_log_photons'] is None
    np.testing.assert_array_equal(loaded['fields_next'],_arrays()['fields_next'])
    changed=deepcopy(_case());changed['spec_hash']='changed'
    with pytest.raises(ValueError,match='incompatible checkpoint'):
        load_checkpoint(tmp_path,changed,_manifest())
    changed=deepcopy(_manifest());changed['source_hash']='changed'
    with pytest.raises(ValueError,match='incompatible checkpoint'):
        load_checkpoint(tmp_path,_case(),changed)
    (tmp_path/'checkpoint_0001.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        load_checkpoint(tmp_path,_case(),_manifest())
