from types import SimpleNamespace
import numpy as np

from hoyag.thermal import DiskThermalMesh
from hoyag.resonator import ThinDiskResonator
from hoyag.thermal_resonator import modal_laser_on_thermal_mesh,sample_cycle_heat
from hoyag.coupled_resonator import time_averaged_populations


def test_shared_heat_population_replay_matches_independent_trajectory():
    mesh=DiskThermalMesh.disk(4,2)
    model=modal_laser_on_thermal_mesh(mesh,ThinDiskResonator())
    optical=SimpleNamespace(periodic_converged=True,period_cycles=1,
        fractions_before_next_pump=np.zeros((3,model.nz,model.ns)),
        log_photon_number=np.zeros(model.nm))
    energy=1e-6
    separate=sample_cycle_heat(model,optical,energy)
    old_mean=time_averaged_populations(model,optical,energy,1e4)
    shared,new_mean=sample_cycle_heat(model,optical,energy,return_mean_fractions=True)
    np.testing.assert_allclose(shared.heat_W_m3,separate.heat_W_m3,rtol=0,atol=0)
    np.testing.assert_allclose(new_mean,old_mean,rtol=0,atol=1e-10)
    assert np.max(abs(new_mean.sum(axis=0)-1))<1e-10
