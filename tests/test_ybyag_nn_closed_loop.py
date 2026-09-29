"""Small deterministic checks before any 1080p production campaign."""

import json
import os
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from hoyag.propagation import Grid2D
from ybluag.camera_dataset import CameraSettings
from ybyag_dataset.distortions.camera import capture, sample_camera_setup
from ybyag_dataset.generator import (REQUIRED_CONCENTRATIONS, REQUIRED_TARGETS,
                                     camera_shape_loss, collocated_external_oracle,
                                     setup_plan, shard_setup_plan, trial_command)
from ybyag_dataset import generator as dataset_generator
from examples.check_ybyag_nn_dataset import check_trial_sequence


ROOT = Path(__file__).resolve().parents[1]


def test_collocated_external_oracle_restores_intentional_phase():
    target=np.array([[0.,np.pi/2],[-np.pi/2,2*np.pi-.1]])
    external=np.array([[.4,-.7],[.8,1.3]])
    command=collocated_external_oracle(target,external)
    residual=np.angle(np.exp(1j*(command+external-target)))
    assert np.max(abs(residual))<1e-14
    with pytest.raises(ValueError,match="aligned"):
        collocated_external_oracle(target,np.zeros((3,3)))


def _fake_process_setup_task(config_path, output_dir, counts, points, split, group_id):
    """Tiny importable task exercising the real spawned process pool."""
    stage=dataset_generator._stage_directory(output_dir,split,group_id)
    folder=stage/split/group_id
    folder.mkdir(parents=True,exist_ok=True)
    setup=folder/"setup.npz"
    setup.write_bytes(b"fixed setup")
    setup_sha=dataset_generator._sha(setup)
    config_sha=dataset_generator._sha(config_path)
    time.sleep(.15)
    for point in range(points):
        stem=f"point_{point:03d}"
        (folder/f"{stem}_measurements.npz").write_bytes(b"measured")
        (folder/f"{stem}_truth.npz").write_bytes(b"truth")
        (folder/f"{stem}.json").write_text(json.dumps({
            "split":split,"setup_id":group_id,"trial_index":point,
            "source_config_sha256":config_sha,"setup_npz_sha256":setup_sha,
            "measurements_file":f"{stem}_measurements.npz",
            "truth_file":f"{stem}_truth.npz","worker_pid":os.getpid()}))
    return split,group_id


def test_each_split_has_all_required_setup_combinations():
    config=json.loads((ROOT/"config/ybyag_nn_dataset.json").read_text())
    plan=setup_plan(config)
    expected={(d,t) for d in REQUIRED_CONCENTRATIONS for t in REQUIRED_TARGETS}
    for split in ("train","validation","test"):
        assert {(r["yb_at_percent"],r["target"]) for r in plan[split]} == expected
    ids=[r["setup_id"] for rows in plan.values() for r in rows]
    assert len(ids)==len(set(ids))
    with pytest.raises(ValueError,match="at least 12"):
        setup_plan(config,{"train":11,"validation":12,"test":12,"stress":0})


def test_machine_shards_keep_complete_setups_and_split_coverage():
    config=json.loads((ROOT/"config/ybyag_nn_dataset.json").read_text())
    counts={"train":36,"validation":36,"test":36,"stress":0}
    full=setup_plan(config,counts)
    local=shard_setup_plan(full,1,2)
    remote=shard_setup_plan(full,0,2)
    required={(d,t) for d in REQUIRED_CONCENTRATIONS for t in REQUIRED_TARGETS}
    for split in ("train","validation","test"):
        local_ids={row["setup_id"] for row in local[split]}
        remote_ids={row["setup_id"] for row in remote[split]}
        assert len(local_ids)==12 and len(remote_ids)==24
        assert not local_ids & remote_ids
        assert local_ids | remote_ids == {row["setup_id"] for row in full[split]}
        for shard in (local,remote):
            assert {(row["yb_at_percent"],row["target"])
                    for row in shard[split]} == required
    with pytest.raises(ValueError,match="full doping/target coverage"):
        shard_setup_plan(setup_plan(config),1,2)




def test_camera_feedback_loss_and_trial_command_are_measurement_driven():
    ideal=np.zeros((2,8,8))
    ideal[:,3:5,3:5]=1
    measured=np.rint(ideal*1000+32).astype(np.uint16)
    shifted=np.roll(measured,2,axis=2)
    assert camera_shape_loss(measured,ideal,32)<1e-20
    assert camera_shape_loss(shifted,ideal,32)>0.1
    grid=Grid2D.square(32,.012)
    target=np.zeros(grid.shape)
    baseline=trial_command(None,target,grid,0)
    plus=trial_command(baseline,target,grid,1)
    minus=trial_command(baseline,target,grid,2)
    assert np.array_equal(baseline,target)
    assert np.max(np.abs(np.angle(np.exp(1j*(plus-minus)))))>0


def test_multi_pulse_variation_changes_exposure_and_is_seeded():
    settings=CameraSettings(width=64,height=36,object_fov_width_mm=8,
        qe_at_signal=.2,optical_throughput=.001,pulses_per_exposure=100,
        psf_sigma_pixels=.5)
    setup=sample_camera_setup(settings,{"camera_shift_pixels":0,
        "camera_rotation_deg":0,"camera_scale_fraction":0},7,enabled=False)
    axis=np.linspace(-6e-3,6e-3,64)
    x,y=np.meshgrid(axis,axis)
    fluence=.01*np.exp(-2*(x*x+y*y)/(1e-3)**2)
    _,base,_=capture(fluence,axis,axis,1030e-9,settings,setup,11,enabled=False)
    variation={"energy_jitter_rms_fraction":.05,
               "pointing_jitter_rms_pixels":1.5}
    a,blurred,info=capture(fluence,axis,axis,1030e-9,settings,setup,11,
                           pulse_variation=variation)
    b,again,again_info=capture(fluence,axis,axis,1030e-9,settings,setup,11,
                               pulse_variation=variation)
    assert np.array_equal(a,b)
    assert np.array_equal(blurred,again)
    assert info==again_info
    assert info["pulse_pointing_blur_sigma_pixels"]>0
    assert blurred.max()<base.max()


def test_rejected_measured_trials_are_complete_and_not_relabelled():
    fixed={"setup_npz_sha256":"fixture","camera_settings":{"width":1920},
           "physical_parameters":{"yb_at_percent":5,
                                  "external_optics":{}},
           "selected_target":"Gaussian TEM00",
           "sensor_parameters":{"bias_K":[0]*5}}
    rows=[{"trial_index":0,"trial_outcome":"baseline_accepted",
           "measured_camera_shape_loss":.3,"trial_improvement_tolerance":.01,**fixed},
          {"trial_index":1,"trial_outcome":"accepted",
           "measured_camera_shape_loss":.2,"trial_improvement_tolerance":.01,**fixed},
          {"trial_index":2,"trial_outcome":"rejected",
           "measured_camera_shape_loss":.25,"trial_improvement_tolerance":.01,**fixed}]
    check_trial_sequence(rows)
    with pytest.raises(ValueError,match="rejected trial improved"):
        check_trial_sequence([rows[0],rows[1],{**rows[2],
                              "measured_camera_shape_loss":.1}])


def test_parallel_coordinator_commits_whole_setups_and_recovers(tmp_path, monkeypatch):
    output=tmp_path/"dataset"
    partial=output/"train"/"train_crystal_0000"
    partial.mkdir(parents=True)
    (partial/"previous_trial.txt").write_text("preserve this interrupted setup")
    running=0
    maximum=0
    calls=[]
    lock=threading.Lock()

    def fake_task(config_path, output_dir, counts, points, split, group_id):
        nonlocal running, maximum
        with lock:
            running+=1
            maximum=max(maximum,running)
            calls.append(group_id)
        time.sleep(.01)
        stage=dataset_generator._stage_directory(output_dir,split,group_id)
        folder=stage/split/group_id
        folder.mkdir(parents=True,exist_ok=True)
        setup=folder/"setup.npz"
        setup.write_bytes(b"fixed setup")
        setup_sha=dataset_generator._sha(setup)
        config_sha=dataset_generator._sha(config_path)
        for point in range(points):
            stem=f"point_{point:03d}"
            (folder/f"{stem}_measurements.npz").write_bytes(b"measured")
            (folder/f"{stem}_truth.npz").write_bytes(b"truth")
            (folder/f"{stem}.json").write_text(json.dumps({
                "split":split,"setup_id":group_id,"trial_index":point,
                "source_config_sha256":config_sha,"setup_npz_sha256":setup_sha,
                "measurements_file":f"{stem}_measurements.npz",
                "truth_file":f"{stem}_truth.npz"}))
        with lock:
            running-=1
        return split,group_id

    monkeypatch.setattr(dataset_generator,"_generate_setup_task",fake_task)
    monkeypatch.setattr(dataset_generator,"ProcessPoolExecutor",
                        lambda max_workers,mp_context: ThreadPoolExecutor(max_workers))
    config=ROOT/"config/ybyag_nn_dataset.json"
    dataset_generator.generate(config,output,points_per_setup=3,workers=4)
    manifest=json.loads((output/"manifest.json").read_text())
    assert maximum>=2
    assert len(calls)==36
    assert {split:len(rows) for split,rows in manifest["splits"].items()}=={
        "train":36,"validation":36,"test":36,"stress":0}
    assert (dataset_generator._stage_directory(output,"train","train_crystal_0000")/
            "previous_final_1"/"previous_trial.txt").read_text()=="preserve this interrupted setup"
    for split,files in manifest["splits"].items():
        assert all((output/name).is_file() for name in files)
        assert files==sorted(files)

    # A crash after moving a finished group but before manifest publication is
    # recovered from its complete files, without regenerating that setup.
    manifest["splits"]["train"]=manifest["splits"]["train"][3:]
    (output/"manifest.json").write_text(json.dumps(manifest))
    dataset_generator.generate(config,output,points_per_setup=3,workers=4,resume=True)
    recovered=json.loads((output/"manifest.json").read_text())
    assert len(recovered["splits"]["train"])==36
    assert len(calls)==36


def test_parallel_generator_uses_distinct_processes(tmp_path,monkeypatch):
    config=ROOT/"config/ybyag_nn_dataset.json"
    full=dataset_generator.setup_plan(json.loads(config.read_text()))
    plan={name:(rows[:2] if name=="train" else []) for name,rows in full.items()}
    monkeypatch.setattr(dataset_generator,"setup_plan",lambda *_:plan)
    monkeypatch.setattr(dataset_generator,"_generate_setup_task",_fake_process_setup_task)
    output=tmp_path/"spawned"
    dataset_generator.generate(config,output,points_per_setup=3,workers=2)
    manifest=json.loads((output/"manifest.json").read_text())
    assert len(manifest["splits"]["train"])==6
    pids={json.loads((output/name).read_text())["worker_pid"]
          for name in manifest["splits"]["train"]}
    assert len(pids)==2
