import numpy as np
import pytest

from hoyag.inhomogeneity import uniform_ho_density_field
from hoyag.population_state import validate_populations
from hoyag.propagation import Grid2D
from hoyag.seeded_periodic import PeriodicAmplifierSettings, solve_periodic_seeded_amplifier
from hoyag.refractive_response import HoIndexResponse


def test_pump_signal_share_state_and_close_optical_heat_ledger():
    grid=Grid2D.square(8,2e-3)
    density=uniform_ho_density_field(grid,2,1e-3,1.52e26)
    x,y=grid.mesh
    seed=np.exp(-(x*x+y*y)/(0.4e-3)**2)*np.exp(1j*np.arctan2(y,x))
    cfg=PeriodicAmplifierSettings(seed_energy_J=10e-9,pump_energy_J=.1e-3,
                                   signal_traversals=2,max_cycles=3)
    result=solve_periodic_seeded_amplifier(seed,grid,density,cfg)
    validate_populations(result['populations_before_pump'],density.values_m3)
    assert result['cycles']==3 and not result['converged']
    assert len(result['pass_records_J'])==2
    assert np.isclose(result['output_energy_J'],result['pass_records_J'][-1][1],rtol=1e-12)
    assert abs(result['heat_ledger_closure_J'])<1e-20
    assert np.isclose(result['heat_energy_J'],result['pump_absorbed_J']-
        result['signal_extracted_J']-result['population_change_energy_J']-
        result['fluorescence_energy_J'],atol=1e-18)
    assert np.isfinite(result['field_out']).all()


def test_invalid_seed_pulse_parameters_rejected():
    with pytest.raises(ValueError):
        PeriodicAmplifierSettings(seed_fwhm_s=1e-3)
    with pytest.raises(ValueError):
        PeriodicAmplifierSettings(signal_traversals=0)


def test_ho_index_increment_requires_coefficient_provenance_and_correct_depth_integral():
    grid=Grid2D.square(4,1e-3)
    density=uniform_ho_density_field(grid,2,1e-3,1e26)
    values=density.values_m3.copy()
    values[:,:,2:]=1.2e26
    from hoyag.inhomogeneity import HoDensityField
    density=HoDensityField(values,1e-3)
    with pytest.raises(ValueError,match='provenance'):
        HoIndexResponse(dn_dHo_m3=1e-30)
    response=HoIndexResponse(dn_dHo_m3=1e-30,provenance='test measurement')
    opd=response.single_pass_opd_m(density)
    np.testing.assert_allclose(opd[:,:2],-1e-8,rtol=1e-12)
    np.testing.assert_allclose(opd[:,2:],1e-8,rtol=1e-12)
