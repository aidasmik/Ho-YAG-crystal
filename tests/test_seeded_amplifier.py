from types import SimpleNamespace
import numpy as np

from hoyag.propagation import Grid2D
from hoyag.temporal import TimeGrid
from hoyag.inhomogeneity import uniform_ho_density_field
from hoyag.seeded_amplifier import (SharedCrystalState, DiskEncounter,
                                    apply_phase_modulator, amplify_seeded_pulse)


def test_ideal_phase_modulator_preserves_local_intensity():
    rng=np.random.default_rng(4)
    field=rng.normal(size=(8,4,4))+1j*rng.normal(size=(8,4,4))
    mask=rng.uniform(-10,10,size=(4,4))
    result=apply_phase_modulator(field,mask,mask/3)
    np.testing.assert_allclose(abs(result['field_after'])**2,abs(field)**2,rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(result['phi_requested'],np.mod(mask+mask/3,2*np.pi))
    assert result['idealized_response']


def test_repeated_encounters_share_updated_populations(monkeypatch):
    import hoyag.seeded_amplifier as amplifier
    grid=Grid2D.square(4,1e-3)
    time=TimeGrid.centered(4,20e-12)
    density=uniform_ho_density_field(grid,1,1e-3,1e26)
    populations=np.zeros((4,1,4,4))
    populations[3]=density.values_m3
    crystal=SharedCrystalState(density,populations)
    seen=[]
    def fake(field,grid,time,energy,state,density,params):
        seen.append(float(state[2,0,0,0]))
        final=state.copy()
        final[2]+=1e20
        final[3]-=1e20
        return SimpleNamespace(field_out=field.copy(),input_energy_J=energy,
            output_energy_J=energy,final_populations_by_slice=final,
            signal_energy_change_by_slice_J=np.array([0.]))
    monkeypatch.setattr(amplifier,'propagate_structured_signal_saturated',fake)
    result=amplify_seeded_pulse(np.ones((4,4,4),complex),grid,time,1e-12,
        crystal,[DiskEncounter(),DiskEncounter()])
    assert seen==[0.,1e20]
    assert result['crystal'] is crystal
    assert result['disk_encounters']==2 and result['material_traversals']==2
    assert not result['seeded_amplifier_configuration_complete']
