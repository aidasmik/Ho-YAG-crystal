import numpy as np

from ybyag_dataset.distortions.material import sample_material
from ybyag_dataset.distortions.thermal import sample_contact, sample_operating_point
from ybyag_dataset.distortions.optical import sample_optical_phase
from ybyag_dataset.distortions.slm import sample_slm_setup, apply_slm
from ybyag_dataset.distortions.camera import sample_camera_setup, capture
from ybluag.camera_dataset import CameraSettings


RANGES = {
    "yb_concentration_rms_fraction":.03,"thickness_rms_fraction":.01,
    "background_absorption_mean_m1":.1,"background_absorption_rms_fraction":.2,
    "surface_figure_rms_nm":2,"thermal_contact_fraction":.2,
    "pump_power_fraction":.1,"pump_radius_fraction":.1,
    "pump_offset_radius_fraction":.1,"coolant_temperature_K":.2,
    "thermal_drift_K_per_s":0,"external_residual_rms_waves":.02,
    "external_aberration_rms_waves":.25,"slm_global_gain_fraction":.02,
    "slm_spatial_gain_rms_fraction":.01,"slm_pixel_gain_rms_fraction":.005,
    "slm_bits":8,"slm_crosstalk_sigma_pixels":.2,"camera_shift_pixels":0,
    "camera_rotation_deg":0,"camera_scale_fraction":0,
}


def test_fixed_crystal_maps_are_smooth_and_reproducible():
    a=sample_material((64,64),RANGES,7)
    b=sample_material((64,64),RANGES,7)
    assert np.array_equal(a["yb_concentration_scale"],b["yb_concentration_scale"])
    assert np.std(np.diff(a["yb_concentration_scale"],axis=0)) < .01
    assert np.std(a["yb_concentration_scale"]) > .02
    contact=sample_contact((8,12),RANGES,4)
    assert contact.shape==(8,12) and np.all(contact>0)


def test_operating_point_perturbs_physical_inputs():
    nominal={"pump_W":20,"radius_mm":1,"coolant_temperature_C":20.5}
    a=sample_operating_point(nominal,RANGES,5)
    assert 18 <= a["pump_W"] <= 22
    assert .9 <= a["radius_mm"] <= 1.1
    assert np.hypot(*a["pump_center_m"]) <= .1e-3


def test_slm_preserves_requested_vortex_separately():
    y,x=np.mgrid[-1:1:64j,-1:1:64j]
    desired=np.mod(np.arctan2(y,x),2*np.pi)
    phase,details=sample_optical_phase(x*5e-3,y*5e-3,5e-3,1030e-9,RANGES,3)
    setup=sample_slm_setup(desired.shape,RANGES,5)
    actual,command=apply_slm(desired,setup)
    assert np.array_equal(command,desired)
    assert np.mean(abs(np.angle(np.exp(1j*(actual-desired))))) < .2
    assert phase.shape==desired.shape and details["achieved_rms_waves"] <= .25


def test_slm_update_delay_keeps_previous_command():
    old=np.zeros((8,8))
    new=np.full((8,8),np.pi)
    setup=sample_slm_setup(old.shape,RANGES,2,enabled=False)
    applied,requested=apply_slm(new,setup,previous_command=old,delayed=True)
    assert np.all(applied==0)
    assert np.all(requested==np.pi)


def test_camera_photons_then_fixed_pattern_and_readout():
    settings=CameraSettings(width=64,height=36,object_fov_width_mm=10,
        qe_at_signal=.2,optical_throughput=.001,pulses_per_exposure=100)
    setup=sample_camera_setup(settings,RANGES,5)
    axis=np.linspace(-6e-3,6e-3,32)
    xx,yy=np.meshgrid(axis,axis)
    fluence=.01*np.exp(-2*(xx*xx+yy*yy)/(1e-3)**2)
    a,clean,_=capture(fluence,axis,axis,1030e-9,settings,setup,1)
    b,clean_again,_=capture(fluence,axis,axis,1030e-9,settings,setup,2)
    assert a.shape==(36,64)
    assert np.array_equal(clean,clean_again)
    assert not np.array_equal(a,b)
    assert clean.max()>clean.min()
