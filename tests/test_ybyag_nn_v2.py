import numpy as np
import pytest

from ybyag_control.nn_v2 import (choose, features, command, build_models,
                                verify_measured_outcome, MODES)


def sample():
    n=32
    y,x=np.mgrid[-1:1:complex(n),-1:1:complex(n)]
    m=dict(input__camera_adu=np.full((2,48,80),150,dtype=np.uint16),
        input__incoming_beam_shape=np.exp(-8*(x*x+y*y)),
        input__yb_relative_map=np.ones((n,n)),input__slm_command_rad=np.zeros((n,n)),
        input__target_phase_mask_rad=np.zeros((n,n)),input__seed_waist_m=.001,
        input__room_temperature_K=293.,input__disk_temperature_probes_K=np.array([294.,np.nan,295.]),
        input__disk_temperature_probe_valid=np.array([True,False,True]),
        input__pump_power_W=.1,input__seed_energy_J=1e-8,input__pump_radius_m=.001,
        input__disk_radius_m=.005,input__disk_thickness_m=1e-4,input__slm_to_disk_m=.25,
        input__output_distance_m=0.,input__yb_nominal_at_percent=10.)
    meta=dict(selected_target='Gaussian TEM00',camera_settings=dict(
        object_fov_width_mm=10.,black_level_adu=32,adc_bits=12,exposure_s=.01))
    config=dict(nominal=dict(field_size_mm=12.))
    return m,meta,config


def test_rectangular_camera_is_padded_not_stretched_and_missing_probe_masked():
    m,meta,config=sample()
    f=features(m,meta,config,overview_size=32,detail_size=16)
    assert f['overview'].shape==(32,32,7)
    assert f['detail'].shape==(16,16,3)
    assert f['overview'][0,:,2].sum()==0
    assert f['overview'][16,:,2].sum()>20
    assert np.all(np.isfinite(f['context']))
    assert f['context'][1]==0 and f['context'][4]==0
    # Truth keys are ignored, including poisoned values.
    m['truth__phase_rad']=np.full((32,32),np.nan)
    other=features(m,meta,config,overview_size=32,detail_size=16)
    for k in f:np.testing.assert_array_equal(f[k],other[k])


def test_registered_pixel_translation_changes_beam_position():
    m,meta,c=sample()
    m['input__camera_adu'][:]=32
    m['input__camera_adu'][:,22:26,38:42]=2000
    a=features(m,meta,c,overview_size=32,detail_size=16)['overview'][...,0]
    pixel=.01/80
    meta['camera_settings']['object_to_pixel_affine']=[[1/pixel,0,49.5],[0,1/pixel,23.5]]
    b=features(m,meta,c,overview_size=32,detail_size=16)['overview'][...,0]
    assert np.unravel_index(np.argmax(b),b.shape)[1] < np.unravel_index(np.argmax(a),a.shape)[1]


def test_exact_hold_and_phase_command():
    m,_,c=sample()
    m['input__slm_command_rad'][:]=6.2
    held,step=command(m,c,np.zeros(MODES))
    np.testing.assert_allclose(held,m['input__slm_command_rad'])
    assert not step.any()
    a=np.zeros(MODES);a[3]=.5
    changed,step=command(m,c,a)
    np.testing.assert_allclose(np.exp(1j*changed),np.exp(1j*(m['input__slm_command_rad']+step)))
    assert changed.min()>=0 and changed.max()<2*np.pi


def test_gate_holds_without_calibration_and_rejects_shape_and_energy_regressions():
    actions=np.zeros((4,MODES));actions[1:,0]=[.1,.2,.3]
    deltas=np.array([[0,0,0],[.1,.02,0],[.2,-.02,0],[.3,.03,-1.]])
    assert choose(actions,deltas,[.8,.97],np.zeros(5),calibrated=False)[0]==0
    assert choose(actions,deltas,[.8,.97],np.zeros(5),calibrated=True)[0]==1
    assert choose(actions,deltas,[.8,.97],np.zeros(5),calibrated=True,
                  supported=[True,False,False,False])[0]==0
    assert choose(actions,deltas,[.8,.97],np.array([0,0,.2,0,0]),calibrated=True)[0]==0
    deltas[1,0]=np.nan
    with pytest.raises(ValueError):choose(actions,deltas,[.8,.97],np.zeros(5),calibrated=True)


def test_measurement_rollback():
    old=dict(valid=True,shape=.98,energy_J=1.)
    assert verify_measured_outcome(old,dict(valid=True,shape=.99,energy_J=.99))[0]
    assert not verify_measured_outcome(old,dict(valid=True,shape=.94,energy_J=1.))[0]
    assert not verify_measured_outcome(old,dict(valid=True,shape=.99,energy_J=.1))[0]


def test_network_has_finite_gradients_exact_zero_action_and_serializable_weights(tmp_path):
    tf=pytest.importorskip('tensorflow')
    p,c=build_models(25,overview_size=32,detail_size=32)
    x=[np.ones((2,32,32,7),np.float32),np.ones((2,32,32,3),np.float32),
       np.ones((2,32,32,3),np.float32),np.ones((2,25),np.float32),np.eye(4,dtype=np.float32)[:2]]
    with tf.GradientTape() as tape:
        a,b=p(x,training=True)
        loss=tf.reduce_mean((a-.1)**2)+tf.reduce_mean((b-.9)**2)
    grads=tape.gradient(loss,p.trainable_variables)
    assert all(g is not None and np.isfinite(g.numpy()).all() for g in grads)
    tf.keras.optimizers.Adam(1e-3).apply_gradients(zip(grads,p.trainable_variables))
    zero=c([*x,np.zeros((2,MODES),np.float32)],training=False).numpy()
    np.testing.assert_array_equal(zero,0)
    p.compile(optimizer='adam',loss=['huber','mse'])
    loss=p.train_on_batch(x,[np.zeros((2,MODES),np.float32),np.ones((2,2),np.float32)],
                         sample_weight=[np.array([1.,0.]),np.ones(2)])
    assert np.isfinite(loss).all()
    c.compile(optimizer='adam',loss='mse')
    loss=c.train_on_batch([*x,np.full((2,MODES),.1,np.float32)],np.zeros((2,3),np.float32))
    assert np.isfinite(loss).all()
    np.testing.assert_array_equal(c([*x,np.zeros((2,MODES),np.float32)]).numpy(),0)
    p.save_weights(tmp_path/'p.weights.h5')
    loaded,_=build_models(25,overview_size=32,detail_size=32)
    loaded.load_weights(tmp_path/'p.weights.h5')
    np.testing.assert_allclose(p(x)[0].numpy(),loaded(x)[0].numpy())


def test_pack_rejects_setup_leakage_before_loading_arrays(tmp_path):
    import importlib.util
    import json
    from pathlib import Path
    script=Path(__file__).resolve().parents[1]/'examples/train_ybyag_controller_v2.py'
    spec=importlib.util.spec_from_file_location('v2_train_test',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    manifest={'schema':'ybyag_nn_closed_loop_v3','splits':{'train':['train.json'],
               'validation':['val.json'],'test':[]}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    (tmp_path/'train.json').write_text(json.dumps({'setup_id':'same','split':'train'}))
    (tmp_path/'val.json').write_text(json.dumps({'setup_id':'same','split':'validation'}))
    (tmp_path/'config.json').write_text('{}')
    with pytest.raises(ValueError,match='leakage'):
        module.pack(tmp_path,tmp_path/'config.json',tmp_path/'cache')
    assert not (tmp_path/'cache').exists()
