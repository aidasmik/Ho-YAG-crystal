from types import SimpleNamespace

import numpy as np
import pytest

from hoyag.propagation import Grid2D, optical_power
from hoyag.structured_beam_gallery import (GallerySettings, nonuniform_density,
    frozen_populations_on_grid, input_modes, phase_pattern)
from hoyag.seeded_amplifier import apply_phase_modulator
from hoyag.thermal import DiskThermalMesh


def test_declared_ho_map_is_nonuniform_positive_and_mean_preserving():
    grid=Grid2D.square(64,3e-3)
    z=np.linspace(0,1e-3,5)
    density=nonuniform_density(grid,z,1e-3)
    active=density.values_m3[density.values_m3>0]
    assert np.isclose(active.mean(),GallerySettings().mean_ho_density_m3,rtol=1e-14)
    assert active.min()>0 and active.max()/active.min()>1.1
    assert np.any(density.values_m3==0)
    assert density.values_m3[0].shape==grid.shape
    again=nonuniform_density(grid,z,1e-3)
    np.testing.assert_array_equal(density.values_m3,again.values_m3)
    alternate=nonuniform_density(grid,z,1e-3,GallerySettings(density_seed=18))
    assert np.max(abs(density.values_m3-alternate.values_m3))>1e23


def test_frozen_fractions_follow_local_density_and_structured_inputs_are_power_normalized():
    grid=Grid2D.square(32,3e-3)
    mesh=DiskThermalMesh.disk(nr=3,nz=2,nphi=4,radius_m=1e-3)
    density=nonuniform_density(grid,mesh.z_edges_m,1e-3)
    fractions=np.zeros((4,*mesh.shape))
    fractions[2]=.4;fractions[3]=.6
    saved=SimpleNamespace(arrays={'mean_fractions':fractions,
        'raw_heat_W_m3':np.zeros(mesh.shape),
        'r_edges_m':mesh.r_edges_m,'z_edges_m':mesh.z_edges_m})
    populations=frozen_populations_on_grid(saved,grid,density)
    np.testing.assert_allclose(populations.sum(axis=0),density.values_m3,rtol=1e-14)
    np.testing.assert_allclose(populations[2],.4*density.values_m3,rtol=1e-14)
    modes=input_modes(grid)
    assert len(modes)==6
    for field in modes.values():
        assert np.isclose(optical_power(field,grid),1.,rtol=1e-13)
    assert not np.allclose(modes['Helical LG(0,+1)'],modes['Double helix LG(0,+2)'])
    assert not np.allclose(modes['Needle Bessel-Gaussian'],modes['Flattop super-Gaussian'])
    needle_center=abs(modes['Needle Bessel-Gaussian'][grid.ny//2,grid.nx//2])
    flattop_center=abs(modes['Flattop super-Gaussian'][grid.ny//2,grid.nx//2])
    assert needle_center > 0 and flattop_center > 0


@pytest.mark.parametrize('name',('none','vortex+1','vortex-1','vortex+2',
                                  'defocus','astigmatic','axicon'))
def test_phase_mask_choices_are_finite_and_ideal_phase_only(name):
    grid=Grid2D.square(32,3e-3)
    pattern=phase_pattern(grid,GallerySettings(phase_mask_name=name))
    assert pattern.shape==grid.shape and np.all(np.isfinite(pattern))
    seed=input_modes(grid)['Gaussian TEM00']
    modulated=apply_phase_modulator(seed[None],pattern)
    np.testing.assert_allclose(abs(modulated['field_after'][0])**2,abs(seed)**2,
                               rtol=1e-14,atol=1e-14)
    if name=='none':assert np.all(pattern==0)
    else:assert np.ptp(pattern)>0


def test_vortex_mask_has_declared_charge_and_invalid_choice_is_rejected():
    grid=Grid2D.square(32,3e-3)
    x,y=grid.mesh
    for name,charge in [('vortex+1',1),('vortex-1',-1),('vortex+2',2)]:
        p=phase_pattern(grid,GallerySettings(phase_mask_name=name))
        np.testing.assert_allclose(np.exp(1j*p),
                                   np.exp(1j*charge*np.arctan2(y,x)))
    with pytest.raises(ValueError,match='phase mask'):
        GallerySettings(phase_mask_name='unknown')
