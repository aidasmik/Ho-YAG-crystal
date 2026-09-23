from types import SimpleNamespace

import numpy as np

from hoyag.propagation import Grid2D, optical_power
from hoyag.structured_beam_gallery import (GallerySettings, nonuniform_density,
    frozen_populations_on_grid, input_modes)
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
    assert len(modes)==4
    for field in modes.values():
        assert np.isclose(optical_power(field,grid),1.,rtol=1e-13)
    assert not np.allclose(modes['Helical LG(0,+1)'],modes['Double helix LG(0,+2)'])
