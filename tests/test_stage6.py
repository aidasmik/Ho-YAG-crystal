"""Independent physical limits for the finite plate, bond, FEM and Jones optics."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from hoyag.thermal import DiskThermalMesh,ThermalBoundary
from hoyag.cooling_plate import (DiskPlateHeatSolver,cooling_plate_mesh,ThermalMaterial,
                                 COPPER_THERMAL,sample_temperature)
from hoyag.thermomechanics import (DiskPlateMesh,ElasticMaterial,YAG_ELASTIC,COPPER_ELASTIC,
    BondedInterface,solve_disk_plate,tetra_kinematics,assemble_body,_recover,von_mises,stress_tensor)
from hoyag.stress_optics import (crystal_axes_111,stress_impermeability,photoelastic_index_matrix,
    geometric_roundtrip_opd,ordered_jones,apply_jones,HotDiskScreens,hot_disk_cavity_roundtrip,
    build_hot_disk_screens)


def mesh_small():
    return DiskPlateMesh.make(nr=2,outer_rings=1,ntheta=8,nz_disk=2,nz_plate=2)


def thermal_small(nphi=1):
    d=DiskThermalMesh.disk(6,6,nphi=nphi,radial_exponent=1.)
    p=cooling_plate_mesh(d,nz=6,outer_rings=3)
    return d,p


def test_assembly_geometry_and_matching_interface():
    m=mesh_small()
    assert m.disk_thickness_m==.001 and m.radius_m==.005
    assert np.array_equal(m.disk.xy_m,m.plate.xy_m[:m.disk.nxy])
    assert np.allclose(m.disk.nodes_m[m.disk.bottom_nodes],m.plate.nodes_m[:m.disk.nxy])
    vd,_=tetra_kinematics(m.disk);vp,_=tetra_kinematics(m.plate)
    # Exact volume for the polygon, not a claim of exact circular FEM geometry.
    assert np.isclose(vd.sum(),.5*8*.005**2*np.sin(2*np.pi/8)*.001,rtol=1e-12)
    assert np.isclose(vp.sum(),.5*8*.010**2*np.sin(2*np.pi/8)*.003,rtol=1e-12)


def test_zero_heat_gives_coolant_temperature_in_both_bodies():
    d,p=thermal_small();s=DiskPlateHeatSolver(d,p);t=s.steady(0.)
    assert np.allclose(t.disk_temperature_K,293.15,atol=1e-10,rtol=0)
    assert np.allclose(t.plate_temperature_K,293.15,atol=1e-10,rtol=0)


def test_two_layer_thermal_resistance_against_analytic_slab():
    d=DiskThermalMesh.disk(4,20,radial_exponent=1.)
    p=cooling_plate_mesh(d,radius_m=.005,nz=20)
    q=1e6;h=1e5;hc=1e4;kd=14.;kp=COPPER_THERMAL.conductivity_W_mK
    s=DiskPlateHeatSolver(d,p,contact_conductance_W_m2K=h,coolant=ThermalBoundary(293.15,hc))
    t=s.steady(q)
    z=d.z_m
    exact=293.15+q*.001*(1/h+1/hc+.003/kp)+q/(2*kd)*(.001**2-z**2)
    # Finite-volume point value has the standard O(dz^2) source-cell correction.
    maxerr=np.max(abs(t.disk_temperature_K[:,0,0]-exact))
    assert maxerr < q*(.001/20)**2/(4*kd)
    assert t.relative_balance_error<1e-10
    assert np.allclose(t.interface_flux_W_m2,1000.,rtol=1e-9)
    assert np.allclose(t.disk_contact_temperature_K-t.plate_contact_temperature_K,1000/h,atol=1e-9,rtol=0)


def test_interface_heat_leaving_disk_enters_plate():
    d,p=thermal_small(4);s=DiskPlateHeatSolver(d,p)
    q=np.ones(d.shape)*1e6;q[:,:,0]*=2
    t=s.steady(q)
    transfer=np.sum(t.interface_flux_W_m2*d.face_areas_m2)
    assert abs(transfer-t.input_heat_W)/t.input_heat_W<1e-10
    assert abs(t.outward_heat_W['plate_rear']-transfer)/transfer<1e-10
    assert not np.allclose(t.disk_temperature_K[:,:,0],t.disk_temperature_K[:,:,2],rtol=0,atol=1e-5)


def test_lower_contact_conductance_increases_disk_temperature():
    d,p=thermal_small()
    a=DiskPlateHeatSolver(d,p,contact_conductance_W_m2K=1e5).steady(1e6)
    b=DiskPlateHeatSolver(d,p,contact_conductance_W_m2K=1e3).steady(1e6)
    assert b.disk_temperature_K.max()>a.disk_temperature_K.max()+.1


def test_plate_is_finite_not_isothermal_boundary():
    d,p=thermal_small()
    a=DiskPlateHeatSolver(d,p).steady(1e6)
    b=DiskPlateHeatSolver(d,p,plate_material=ThermalMaterial(10.,8940.,385.2)).steady(1e6)
    assert np.ptp(a.plate_temperature_K)>0
    assert b.disk_temperature_K.max()>a.disk_temperature_K.max()


def test_two_body_transient_energy_budget_including_plate_storage():
    d,p=thermal_small();s=DiskPlateHeatSolver(d,p)
    t=s.advance(293.15,293.15,1e6,1e-3)
    assert t.stored_energy_change_J>0
    assert t.relative_balance_error<1e-9
    ep=np.sum(s.ps.capacity_J_K*(t.plate_temperature_K-293.15))
    assert ep>0


def test_contact_map_zero_region_not_automatically_tied():
    d,p=thermal_small(4);h=np.full((d.nr,d.nphi),1e5);h[:,0]=0.
    s=DiskPlateHeatSolver(d,p,contact_conductance_W_m2K=h);t=s.steady(1e6)
    assert np.all(t.interface_flux_W_m2[:,0]==0.)
    assert t.relative_balance_error<1e-10


def test_perfect_thermal_contact_has_continuous_face_temperature():
    d,p=thermal_small();s=DiskPlateHeatSolver(d,p,contact_conductance_W_m2K=np.inf)
    t=s.steady(1e6)
    assert np.max(abs(t.disk_contact_temperature_K-t.plate_contact_temperature_K))<1e-9


def test_reject_disconnected_or_bad_thermal_interface():
    d,p=thermal_small()
    with pytest.raises(ValueError):DiskPlateHeatSolver(d,p,contact_conductance_W_m2K=0).steady(1e6)
    with pytest.raises(ValueError):DiskPlateHeatSolver(d,p,contact_conductance_W_m2K=-1.)


def test_temperature_sampling_constant_and_coordinate_offset():
    d,p=thermal_small()
    pts=np.array([[0,0,.002],[.003,0,.003]])
    assert np.allclose(sample_temperature(p,310.,pts,z_offset_m=.001),310.)


def test_no_thermal_or_external_load_has_no_stress_or_displacement():
    m=mesh_small();s=solve_disk_plate(m,293.15,293.15)
    assert np.max(abs(s.disk_u_m))==0 and np.max(abs(s.plate_u_m))==0
    assert np.max(abs(s.disk_stress_Pa))==0


def test_free_uniform_expansion_identical_materials_no_stress():
    m=mesh_small();s=solve_disk_plate(m,303.15,303.15,plate_material=YAG_ELASTIC,support='free')
    origin=m.plate.nodes_m[m.plate.bottom_nodes[0]]
    exact=7e-6*10*(m.disk.nodes_m-origin)
    assert np.max(abs(s.disk_u_m-exact))<1e-17
    assert np.max(abs(s.disk_stress_Pa))<.01
    assert np.max(abs(s.plate_stress_Pa))<.01
    assert s.strain_and_bond_energy_J<1e-20


def test_clamped_uniform_temperature_generates_constraint_stress():
    m=mesh_small();s=solve_disk_plate(m,303.15,303.15)
    assert np.max(abs(s.disk_stress_Pa))>1e6
    assert np.max(abs(s.plate_u_m[m.plate.bottom_nodes]))==0
    assert s.free_residual_relative<1e-10


def test_cte_mismatch_generates_stress_with_only_rigid_body_gauges():
    m=mesh_small()
    a=solve_disk_plate(m,303.15,303.15,plate_material=YAG_ELASTIC,support='free')
    b=solve_disk_plate(m,303.15,303.15,support='free')
    assert np.max(von_mises(b.disk_stress_Pa))>1e6
    assert np.max(von_mises(b.disk_stress_Pa))>100*np.max(von_mises(a.disk_stress_Pa))


def test_sliding_limit_removes_interface_shear_not_normal_contact():
    m=mesh_small()
    s=solve_disk_plate(m,303.15,303.15,interface=BondedInterface(1e14,0.),support='free')
    assert np.max(abs(s.interface_traction_on_disk_Pa[:,:2]))==0.
    assert np.max(von_mises(s.disk_stress_Pa))<.01


def test_external_pressure_and_interface_action_reaction_balance():
    m=mesh_small();pressure=1e5
    s=solve_disk_plate(m,293.15,293.15,front_pressure_Pa=pressure)
    vd,_=tetra_kinematics(m.disk);force=pressure*vd.sum()/m.disk_thickness_m
    assert np.isclose(s.support_reaction_N[2],-force,rtol=1e-8)
    assert np.isclose(s.interface_force_on_disk_N[2],-force,rtol=1e-8)
    assert np.linalg.norm(s.support_reaction_N[:2])<1e-8


def test_virtual_work_identity():
    m=mesh_small();s=solve_disk_plate(m,310.,295.)
    assert np.isclose(2*s.displacement_quadratic_energy_J,s.thermal_load_work_J,rtol=1e-10)


def test_fully_blocked_tetra_stress_matches_hooke_thermal_stress():
    body=mesh_small().disk;mat=YAG_ELASTIC
    _,_,cache=assemble_body(body,mat,303.15)
    _,stress,_=_recover(body,np.zeros_like(body.nodes_m),mat,cache)
    expected=-mat.young_Pa*mat.expansion_K1*10/(1-2*mat.poisson)
    assert np.allclose(stress[:,:3],expected,rtol=1e-12)
    assert np.max(abs(stress[:,3:]))==0


def test_von_mises_uniaxial_and_hydrostatic():
    assert von_mises(np.array([10.,0,0,0,0,0]))==10.
    assert von_mises(np.array([10.,10,10,0,0,0]))==0.


def test_cubic_rotation_and_hydrostatic_optical_isotropy():
    q=crystal_axes_111(.37)
    assert np.allclose(q.T@q,np.eye(3)) and np.isclose(np.linalg.det(q),1.)
    db=stress_impermeability([1e7,1e7,1e7,0,0,0],crystal_axes=q)
    assert np.max(abs(db-np.eye(3)*np.trace(db)/3))<1e-19


def test_uniaxial_cubic_elasto_optic_formula():
    st=np.array([1e7,0,0,0,0,0]);m=YAG_ELASTIC
    db=stress_impermeability(st,crystal_axes=np.eye(3))
    bx=(-.029-2*m.poisson*.0091)*st[0]/m.young_Pa
    by=(.0091*(1-m.poisson)-m.poisson*(-.029))*st[0]/m.young_Pa
    assert np.isclose(db[0,0],bx,rtol=1e-12,atol=0)
    assert np.isclose(db[1,1],by,rtol=1e-12,atol=0)


def test_photoelastic_shear_factor():
    m=YAG_ELASTIC;shear=1e7
    db=stress_impermeability([0,0,0,shear,0,0],crystal_axes=np.eye(3))
    mu=m.young_Pa/(2*(1+m.poisson))
    assert np.isclose(db[0,1],-.0615*shear/mu,rtol=1e-12,atol=0)


def test_zero_stress_adds_no_extra_free_thermal_expansion_index():
    dn=photoelastic_index_matrix(np.zeros((10,6)))
    assert np.max(abs(dn))<1e-14


def test_geometric_opd_rigid_translation_and_fixed_rear_expansion():
    n=1.8
    assert np.isclose(geometric_roundtrip_opd(1e-7,1e-7,index=n),2e-7,rtol=1e-12,atol=0)
    assert np.isclose(geometric_roundtrip_opd(-1e-7,0,index=n),2*(n-1)*1e-7,rtol=1e-12,atol=0)


def test_ordered_jones_reciprocity_and_unitarity():
    dn=np.zeros((2,2,2));dn[0]=[[1e-4,0],[0,-1e-4]];dn[1]=[[0,1e-4],[1e-4,0]]
    inward,outward=ordered_jones(dn,[1e-3,1e-3],2e-6)
    assert np.allclose(inward.T,outward,rtol=0,atol=1e-14)
    assert not np.allclose(inward,outward,rtol=0,atol=1e-4)
    assert np.allclose(inward.conj().T@inward,np.eye(2),rtol=0,atol=1e-14)
    f=np.array([1.,1j])/np.sqrt(2)
    assert np.isclose(np.vdot(apply_jones(f,inward),apply_jones(f,inward)).real,1.,rtol=1e-12)


def test_uniform_birefringence_double_pass_retardance():
    dn=np.array([[[1e-4,0],[0,0]]]);ji,jo=ordered_jones(dn,[.001],2e-6)
    assert np.allclose(jo@ji,np.diag([np.exp(2j*np.pi*.1),1]),atol=1e-14)


def test_full_map_zero_load_and_outside_identity():
    m=mesh_small();s=solve_disk_plate(m,293.15,293.15)
    tm=DiskThermalMesh.disk(4,3)
    xy=np.array([[0.,0.],[.003,0],[.006,0]])
    screens=build_hot_disk_screens(m,s,tm,293.15,xy)
    assert np.allclose(screens.inward_jones,np.eye(2),rtol=0,atol=1e-12)
    assert np.max(abs(screens.mean_roundtrip_opd_m))<1e-15
    assert np.array_equal(screens.outward_jones[-1],np.eye(2))


def test_physical_parameter_validation():
    with pytest.raises(ValueError):ElasticMaterial(1.,.5,1.)
    with pytest.raises(ValueError):BondedInterface(-1.,1.)
    with pytest.raises(ValueError):DiskPlateMesh.make(plate_radius_m=.003)
    with pytest.raises(ValueError):ordered_jones(np.ones((1,2,2)),[-1],1.)
    with pytest.raises(ValueError):stress_impermeability(np.zeros(6),crystal_axes=np.ones((3,3)))


def test_hot_roundtrip_recovers_cold_cavity_and_counts_both_faces():
    from hoyag.resonator import ThinDiskResonator,cavity_roundtrip,lg0_field
    n=128;size=.012;d=size/n;x=(np.arange(n)-(n-1)/2)*d
    grid=SimpleNamespace(nx=n,ny=n,dx=d,dy=d,shape=(n,n),mesh=np.meshgrid(x,x,indexing='xy'),fx=np.fft.fftfreq(n,d),fy=np.fft.fftfreq(n,d))
    c=ThinDiskResonator();xx,yy=grid.mesh
    u=lg0_field(xx,yy,c.waist_m);field=np.stack([u,np.zeros_like(u)])
    zero=np.zeros(grid.shape);identity=np.broadcast_to(np.eye(2,dtype=complex),grid.shape+(2,2)).copy()
    screens=HotDiskScreens(identity,identity,zero,zero,zero,zero,zero,zero,c.wavelength_m)
    hot,oc=hot_disk_cavity_roundtrip(field,grid,c,screens)
    cold,coldoc=cavity_roundtrip(u,grid,c)
    assert np.allclose(hot[0],cold,rtol=1e-12,atol=1e-12)
    assert np.allclose(oc[0],coldoc,rtol=1e-12,atol=1e-12)
    opd=2e-8;screens.geometry_roundtrip_opd_m=np.full(grid.shape,opd)
    hot,_=hot_disk_cavity_roundtrip(field,grid,c,screens)
    assert np.allclose(hot[0],cold*np.exp(2j*np.pi*opd/c.wavelength_m),rtol=1e-10,atol=1e-12)
