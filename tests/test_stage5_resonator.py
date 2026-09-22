import numpy as np
import pytest
from hoyag.thermal import DiskThermalMesh
from hoyag.resonator import ThinDiskResonator
from hoyag.thermal_resonator import area_averaged_lg0,modal_laser_on_thermal_mesh,sample_cycle_heat


def test_shared_mesh_conserves_volume_and_gaussian_energy():
    mesh=DiskThermalMesh.disk(32,4);c=ThinDiskResonator();m=modal_laser_on_thermal_mesh(mesh,c)
    assert np.allclose(m.volume,mesh.volumes_m3.reshape(m.nz,m.ns),rtol=1e-14,atol=0)
    assert np.isclose(m.volume.sum(),np.pi*.005**2*.001,rtol=1e-14)
    assert np.isclose(m.pump@m.area,1,rtol=1e-14)
    for l in (0,1,2,-3): assert np.isclose(area_averaged_lg0(mesh,c.waist_m,l)@m.area,1,rtol=1e-14)


def test_asymmetric_mesh_shared_with_modal_primitive():
    mesh=DiskThermalMesh.disk(12,2,6);m=modal_laser_on_thermal_mesh(mesh,ThinDiskResonator())
    assert m.ns==72 and m.nc==144
    assert np.isclose(m.modes[0]@m.area,1,rtol=1e-14)


def test_cycle_heat_rejects_unconverged_optical_state_by_default():
    mesh=DiskThermalMesh.disk(16,2);m=modal_laser_on_thermal_mesh(mesh,ThinDiskResonator())
    state=m.run(1e-4,max_cycles=2,min_cycles=2)
    with pytest.raises(ValueError): sample_cycle_heat(m,state,1e-4)


def test_transient_heat_ledger_includes_pump_storage():
    mesh=DiskThermalMesh.disk(16,2);m=modal_laser_on_thermal_mesh(mesh,ThinDiskResonator())
    state=m.run(1e-4,max_cycles=2,min_cycles=2)
    heat=sample_cycle_heat(m,state,1e-4,cycles=1,allow_transient=True)
    b=heat.budget
    assert b['ion_storage_change_W']>0
    assert 0<b['heat_W']<b['pump_absorbed_W']
    assert b['local_ledger_L1_error_over_incident']<1e-6
    assert np.isclose(b['pump_absorbed_W'],b['heat_W']+b['stimulated_signal_W']+b['fluorescence_W']+b['ion_storage_change_W'],rtol=1e-5)
    assert np.isclose(b['pump_incident_W'],b['pump_absorbed_W']+b['pump_escaped_W']+b['pump_mirror_loss_W'],rtol=1e-10)


def test_modal_mesh_rejects_wrong_geometry_and_clipped_pump():
    c=ThinDiskResonator()
    with pytest.raises(ValueError): modal_laser_on_thermal_mesh(DiskThermalMesh.disk(radius_m=.001),c)
    with pytest.raises(ValueError): modal_laser_on_thermal_mesh(DiskThermalMesh.disk(),c,pump_waist_m=.02)
