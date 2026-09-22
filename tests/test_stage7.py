"""Stage 7 tests exercise real repository operators; no copied physics harness."""
from dataclasses import replace
from types import SimpleNamespace
import json
from pathlib import Path
import numpy as np
import pytest
from hoyag.propagation import Grid2D
from hoyag.resonator import ThinDiskResonator, cavity_roundtrip, lg0_field, radial_laser
from hoyag.stress_optics import HotDiskScreens, hot_disk_cavity_roundtrip
from hoyag.thermal import DiskThermalMesh
from hoyag.thermal_resonator import area_averaged_lg0
from hoyag.populations import HoYAGFourLevelParams
from hoyag.vector_cavity import (VectorRoundTrip, PlaneExchange, normalize_vector,
    aligned_distance, subspace_distance, solve_vector_eigenfields, passive_mode_losses)
from hoyag.coupled_resonator import (FieldCoupledLaser, PlateAssembly, HotCavitySettings,
    initial_vector_modes, run_coupled_hot_cavity, _weighted_relative, time_averaged_populations)

ROOT=Path(__file__).resolve().parents[1]


def setup_field():
    grid=Grid2D.square(128,.012);c=ThinDiskResonator(air_gap_m=.25,output_transmission=.02)
    x,y=grid.mesh
    f=normalize_vector(np.stack([lg0_field(x,y,c.waist_m),np.zeros(grid.shape)]))
    return grid,c,f


def identity_screens(grid,c,opd=0.):
    shape=grid.shape;identity=np.broadcast_to(np.eye(2,dtype=complex),(*shape,2,2)).copy()
    zero=np.zeros(shape)
    return HotDiskScreens(identity,identity,np.full(shape,opd),zero,zero,zero,zero,zero,c.wavelength_m)


@pytest.mark.parametrize('ell',[0,1,-1,2])
def test_cached_operator_matches_cold_scalar_operator(ell):
    grid,c,_=setup_field();x,y=grid.mesh
    f=normalize_vector(np.stack([lg0_field(x,y,c.waist_m,ell),np.zeros(grid.shape)]))
    a,b=VectorRoundTrip(grid,c).propagate(f)
    aa,bb=cavity_roundtrip(f[0],grid,c)
    assert np.allclose(a[0],aa,rtol=1e-11,atol=1e-13)
    assert np.allclose(b[0],bb,rtol=1e-11,atol=1e-13)
    assert np.max(abs(a[1]))==0


def test_cached_vector_operator_matches_stage6_with_nondiagonal_jones():
    grid,c,f=setup_field();screens=identity_screens(grid,c,2e-8)
    angle=.03
    h=np.array([[np.cos(angle),1j*np.sin(angle)],[1j*np.sin(angle),np.cos(angle)]])
    screens.inward_jones=np.broadcast_to(h,(*grid.shape,2,2)).copy()
    screens.outward_jones=screens.inward_jones.copy()
    x,y=grid.mesh;gain=.02*np.exp(-2*(x*x+y*y)/(.001**2))
    a,b=VectorRoundTrip(grid,c,screens,gain).propagate(f)
    aa,bb=hot_disk_cavity_roundtrip(f,grid,c,screens,single_pass_log_gain=gain)
    assert np.allclose(a,aa,rtol=1e-11,atol=1e-13)
    assert np.allclose(b,bb,rtol=1e-11,atol=1e-13)


def test_roundtrip_gain_not_double_counted():
    grid,c,f=setup_field();g=.04
    a,_=VectorRoundTrip(grid,c,single_pass_log_gain=g).propagate(f)
    assert np.isclose(np.sum(abs(a)**2),c.passive_power_retention*np.exp(2*g),rtol=1e-10)


def test_passive_losses_recover_configured_mirror_loss():
    grid,c,f=setup_field()
    loss,out=passive_mode_losses(VectorRoundTrip(grid,c),f[None])
    assert np.isclose(loss[0],c.logarithmic_loss,rtol=1e-10)
    assert np.isclose(out[0],c.output_transmission,rtol=1e-10)


def test_passive_loss_function_rejects_gain():
    grid,c,f=setup_field()
    with pytest.raises(ValueError):
        passive_mode_losses(VectorRoundTrip(grid,c,single_pass_log_gain=1),f[None])


def test_global_phase_does_not_break_convergence():
    _,_,f=setup_field()
    assert aligned_distance(f,f*np.exp(1.789j))<3e-8
    assert aligned_distance(f,f[::-1])>1.0


def test_subspace_residual_is_basis_independent():
    _,_,f=setup_field();g=f[::-1];fields=np.stack([f,g])
    rotated=np.stack([(f+g)/np.sqrt(2),(f-g)/np.sqrt(2)])
    assert subspace_distance(fields,rotated)<3e-8


@pytest.mark.parametrize('nphi',[1,4,12])
def test_plane_quadrature_preserves_constant_intensity_and_annular_area(nphi):
    grid,c,_=setup_field();mesh=DiskThermalMesh.disk(16,2,nphi=nphi)
    bridge=PlaneExchange(grid,mesh,order=6)
    areas=bridge.matrix@np.ones(grid.nx*grid.ny)
    assert np.allclose(areas,mesh.face_areas_m2.ravel(),rtol=1e-12,atol=1e-20)
    assert np.isclose(areas.sum(),np.pi*(.005)**2,rtol=1e-12)


def test_projected_mode_has_unit_power_and_positive_intensity():
    grid,c,f=setup_field();mesh=DiskThermalMesh.disk(24,2,nphi=12)
    bridge=PlaneExchange(grid,mesh,order=6)
    m,error=bridge.mode_intensity(f)
    assert np.isclose(m@bridge.area,1,rtol=1e-12)
    assert np.min(m)>=0 and error<.03


def test_projection_rejects_field_outside_material():
    grid,c,_=setup_field();mesh=DiskThermalMesh.disk(12,2)
    f=np.ones((2,*grid.shape),complex)
    with pytest.raises(ValueError,match='quadrature mismatch'):
        PlaneExchange(grid,mesh).mode_intensity(f)


def test_constant_gain_remapping_inside_and_outside_disk():
    grid,c,_=setup_field();mesh=DiskThermalMesh.disk(12,2,nphi=8)
    result=PlaneExchange(grid,mesh).surface_on_grid(np.ones(mesh.nr*mesh.nphi)*3)
    x,y=grid.mesh;inside=x*x+y*y<=.005**2
    assert np.allclose(result[inside],3) and np.all(result[~inside]==0)


def test_pass_visit_averaging_preserves_power():
    grid,c,f=setup_field();mesh=DiskThermalMesh.disk(20,2,nphi=8)
    bridge=PlaneExchange(grid,mesh,order=6)
    m,error=bridge.visit_averaged_mode(f,VectorRoundTrip(grid,c))
    assert np.isclose(m@bridge.area,1,rtol=1e-12)
    assert error<.03


def test_eigenfields_full_grid_residual_and_polarization_degeneracy():
    grid,c,f=setup_field();x,y=grid.mesh
    gain=.05*np.exp(-2*(x*x+y*y)/(.0007**2))
    op=VectorRoundTrip(grid,c,single_pass_log_gain=gain)
    r=solve_vector_eigenfields(op,f[None],candidates=2,tolerance=2e-6,maxiter=400)
    assert r.converged,r.status
    assert max(r.residuals)<2e-6
    assert np.isclose(np.linalg.norm(r.fields[0]),1,rtol=1e-12)
    assert r.operator_calls>0


def test_arnoldi_failure_is_not_marked_converged():
    grid,c,f=setup_field();x,y=grid.mesh
    r=solve_vector_eigenfields(VectorRoundTrip(grid,c,single_pass_log_gain=.01*np.cos(x/.001)),
          f[None],candidates=2,tolerance=1e-13,maxiter=1)
    assert not r.converged


def test_extra_cavity_loss_changes_photons_not_material_or_jacobian():
    c=ThinDiskResonator();mesh=DiskThermalMesh.disk(6,2)
    area=mesh.face_areas_m2.ravel();modes=area_averaged_lg0(mesh,c.waist_m)[None]
    pump=area_averaged_lg0(mesh,.0005);density=np.full((mesh.nz,len(area)),1.52e26)
    m=FieldCoupledLaser(c,area,density,modes,pump)
    f=np.zeros((3,m.nz,m.ns));f[2]=.2
    y=np.r_[f.ravel(),np.log(1e8),0.]
    base=m.rhs(0,y);jac=m.jacobian(0,y)
    m.set_roundtrip_losses([c.logarithmic_loss+.01])
    actual=m.rhs(0,y)
    assert np.allclose(actual[:3*m.nc],base[:3*m.nc],rtol=0,atol=0)
    assert np.isclose(actual[3*m.nc]-base[3*m.nc],-.01/m.trt,rtol=1e-12)
    assert np.allclose(m.jacobian(0,y).toarray(),jac.toarray(),rtol=0,atol=0)
    assert actual[-1]==base[-1]
    with pytest.raises(ValueError):m.set_roundtrip_losses([0.])


def test_unrelaxed_source_residual_cannot_be_hidden_by_small_mixing():
    old=np.ones((2,3,4));new=old*2;volume=np.ones_like(old)
    raw=_weighted_relative(new,old,volume)
    damped=_weighted_relative(old+.001*(new-old),old,volume)
    assert np.isclose(raw,.5) and damped<.002


def test_stage7_assembly_retains_finite_plate_and_interface():
    cfg=json.loads((ROOT/'config/stage6_assembly.json').read_text())
    cfg['numerics']['mechanical'].update(nr=2,outer_rings=1,ntheta=8,nz_disk=2,nz_plate=2)
    cfg['numerics']['plate_thermal_nz']=3
    mesh=DiskThermalMesh.disk(6,3,nphi=4);grid=Grid2D.square(64,.012)
    assembly=PlateAssembly(mesh,grid,cfg)
    q=np.full(mesh.shape,1e6);q[:,:,0]*=1.2
    temp,disp,screens=assembly.solve(q)
    assert temp.disk_temperature_K.max()>temp.plate_temperature_K.max()>293.15
    assert abs(temp.balance_error_W)/temp.input_heat_W<1e-8
    assert np.max(abs(disp.interface_jump_m))>0
    assert screens.inward_jones.shape==(*grid.shape,2,2)
    assert np.max(abs(screens.geometry_roundtrip_opd_m))>0


def test_settings_validate_limits_and_independent_initial_modes():
    with pytest.raises(ValueError):HotCavitySettings(heat_relaxation=2)
    with pytest.raises(ValueError):HotCavitySettings(eigen_maxiter=2.5)
    grid,c,_=setup_field();fields=initial_vector_modes(grid,c,(0,0,1))
    assert np.linalg.matrix_rank(fields.reshape(3,-1))==3


def test_cycle_mean_requires_actual_periodic_state():
    with pytest.raises(ValueError):
        time_averaged_populations(None,SimpleNamespace(periodic_converged=False),1e-3,1e4)
