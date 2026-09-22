import numpy as np
import pytest
from dataclasses import replace
from types import SimpleNamespace
from hoyag.populations import HoYAGFourLevelParams,H,C0,I5,I6,I7,I8
from hoyag.propagation import Grid2D
from hoyag.heat import ManifoldEnergies, heat_rates, stored_energy_density, pulse_heat_from_states
from hoyag.thermal import DiskHeatSolver,ThermalBoundary
from hoyag.thermal_optics import (thermal_opd, phase_from_opd,fit_thermal_lens,
    thermal_cavity_roundtrip, gaussian_hot_cavity,radial_source_to_volume)
from hoyag.resonator import ThinDiskResonator,cavity_roundtrip,lg0_field,radial_laser
from hoyag.thermal_resonator import measure_heat_cycle,cycle_averaged_heat


def isolated():
    return HoYAGFourLevelParams(tau5_s=1e99,tau6_s=1e99,tau7_s=1e99,
        M56_s1=0,M67_s1=0,M78_s1=0,k75_m3_s=0,k76_m3_s=0,C57_m3_s=0,C67_m3_s=0)


def test_centroids_match_existing_database():
    expected=[125106/11,114484/13,80118/15,5003/17]
    assert np.allclose(ManifoldEnergies().centers_cm1,expected,rtol=1e-14,atol=0)
    assert np.isclose(1e6*H*C0/ManifoldEnergies().photons_J[-1],1.9814120241,rtol=2e-10)


def test_dark_ground_state_generates_no_heat():
    p=HoYAGFourLevelParams();n=np.array([0,0,0,p.N_total_m3])
    for a in heat_rates(n,p).values():assert np.all(a==0)


def test_heat_first_law_with_all_channels_and_light():
    p=HoYAGFourLevelParams();rng=np.random.default_rng(34)
    n=rng.random((4,3,4));n*=p.N_total_m3/n.sum(axis=0)
    q=heat_rates(n,p,pump_intensity_W_m2=4e10,signal_intensity_W_m2=7e9,
                 cavity_spontaneous_fraction=3e-8)
    scale=max(np.max(abs(q['pump_absorbed'])),np.max(abs(q['signal_extracted'])))
    assert np.max(abs(q['balance_residual']))/scale<1e-13
    assert np.allclose(q['heat'],sum(q[k] for k in ['multiphonon','etu','cross_relaxation','pump_quantum_defect','laser_quantum_defect','spontaneous_quantum_defect']))


def test_pure_radiative_decay_is_not_heat_and_uses_own_lifetime():
    p=replace(isolated(),tau5_s=.004,tau6_s=.007,tau7_s=.009)
    for level in [I5,I6,I7]:
        n=np.zeros(4);n[level]=1e24;n[I8]=1e26
        rates=heat_rates(n,p)
        assert abs(rates['heat'])<1e-10
        assert np.isclose(rates['fluorescence_escape'],-rates['stored_rate'],rtol=1e-14)


def test_nonradiative_decay_releases_manifold_energy():
    p=replace(isolated(),M78_s1=10.)
    n=np.array([0,0,1e24,1e26]);q=heat_rates(n,p)
    expected=10*n[I7]*ManifoldEnergies().joules[I7]
    assert np.isclose(q['heat'],expected,rtol=1e-14)


def test_signed_upconversion_does_not_clip_phonon_assisted_cooling():
    p=replace(isolated(),k75_m3_s=3.8e-24)
    q=heat_rates(np.array([0.,0.,1e25,1e26]),p)
    assert q['etu']<0 and q['heat']<0
    assert abs(q['balance_residual'])<1e-9*abs(q['heat'])


def test_quantum_defect_pump_then_signal_equals_optical_energy_difference():
    n=np.array([0.,0.,0.,1e26]);excited=n.copy();dn=1e23
    excited[I7]+=dn;excited[I8]-=dn;p=isolated()
    ep,es=H*C0/p.pump_wavelength_m,H*C0/p.laser_wavelength_m
    prompt=pulse_heat_from_states(n,excited,dn*ep)
    extraction=pulse_heat_from_states(excited,n,0.,dn*es)
    assert 0<prompt<dn*ep
    assert np.isclose(prompt+extraction,dn*(ep-es),rtol=1e-13)


def test_changing_energy_zero_does_not_change_heat():
    e=ManifoldEnergies();e2=replace(e,centers_cm1=tuple(np.array(e.centers_cm1)+12345))
    n=np.array([1e23,2e23,3e24,1e26])
    assert np.isclose(heat_rates(n,energies=e)['heat'],heat_rates(n,energies=e2)['heat'],rtol=1e-13)


def test_spontaneous_cavity_photons_are_not_counted_twice():
    p=replace(isolated(),tau7_s=.01);n=np.array([0,0,1e24,1e26]);fraction=.2
    q=heat_rates(n,p,cavity_spontaneous_fraction=fraction)
    assert np.isclose(q['spontaneous_to_cavity'],fraction*n[I7]/p.tau7_s*H*C0/p.laser_wavelength_m)
    assert np.isclose(q['fluorescence_escape'],(1-fraction)*n[I7]/p.tau7_s*ManifoldEnergies().photons_J[-1])
    assert abs(q['balance_residual'])<5e-14*abs(q['fluorescence_escape'])


def solver(n=17,nz=8,h=np.inf):
    return DiskHeatSolver(Grid2D.square(n,.01),nz=nz,
                           boundary=ThermalBoundary(rear_h_W_m2K=h))


def test_zero_heat_returns_sink_temperature():
    s=solver();o=s.steady(0.)
    assert np.all(o.temperature_K==s.boundary.sink_temperature_K)
    assert o.boundary_power_W==0


def test_internal_heat_flux_cancels_and_matrix_is_symmetric():
    s=solver();assert np.max(abs((s.K-s.K.T).data),initial=0)<1e-14
    assert np.max(abs(np.asarray(s.K.sum(axis=1)).ravel()-s.boundary_conductance))<1e-13


@pytest.mark.parametrize('h',[np.inf,1e4,1e5])
def test_uniform_rear_cooled_slab_solution(h):
    s=solver(n=11,nz=16,h=h);q=2e7;o=s.steady(q)
    expected=q*(s.thickness_m**2-s.z_m**2)/28
    if np.isfinite(h):expected+=q*s.thickness_m/h
    actual=o.temperature_K[:,5,5]-s.boundary.sink_temperature_K
    error=abs(actual-expected)
    assert error.max()<q*s.dz**2/(8*14)*1.01
    assert abs(o.residual_W)/o.deposited_power_W<1e-8


def test_slab_axial_error_is_second_order():
    errors=[]
    for nz in [4,8,16]:
        s=solver(n=7,nz=nz);q=1e7;o=s.steady(q)
        exact=q*(s.thickness_m**2-s.z_m**2)/28
        errors.append(np.max(abs(o.temperature_K[:,3,3]-293.15-exact)))
    assert 3.95<errors[0]/errors[1]<4.05
    assert 3.95<errors[1]/errors[2]<4.05


def test_insulated_transient_conserves_heat_and_uniform_temperature():
    s=solver(n=11,nz=4,h=0.);q=1e7;dt=.003
    out=s.step(293.15,q,dt)
    expected=293.15+q*dt/(4560*680)
    assert np.allclose(out.temperature_K[s.mask],expected,rtol=1e-12)
    assert abs(out.stored_energy_change_J-out.deposited_power_W*dt)<1e-11
    assert abs(out.residual_W)<1e-8
    with pytest.raises(ValueError):s.steady(q)


def test_implicit_step_cools_without_overshoot_and_closes_energy():
    s=solver(n=13,nz=4);o=s.step(303.15,0.,.1)
    assert np.max(o.temperature_K)<303.15 and np.min(o.temperature_K)>=293.15-1e-9
    assert abs(o.residual_W)<1e-7
    assert o.stored_energy_change_J<0 and o.boundary_power_W>0


def test_transient_approaches_steady_state():
    s=solver(n=9,nz=5,h=1e5);q=3e7;t=np.full(s.shape,293.15)
    for _ in range(110):t=s.step(t,q,.02).temperature_K
    expected=s.steady(q).temperature_K
    assert np.max(abs(t-expected))<1e-6


def test_heat_impulse_preserves_deposited_energy():
    s=solver(n=9,nz=4);e=1234.;out=s.deposit_energy(293.15,e)
    expected=293.15+e/(4560*680)
    assert np.allclose(out[s.mask],expected,rtol=1e-14)


def test_host_geometry_is_independent_of_ho_doping():
    s=solver();assert s.material_volume_m3>0
    # No Ho input is needed for bulk conduction in undoped YAG.
    assert s.steady(1e7).deposited_power_W>0


def test_nonuniform_conductivity_harmonic_flux_and_invalid_inputs():
    base=solver(n=9,nz=6);k=np.full(base.shape,14.);k[3:]=7.
    s=DiskHeatSolver(base.grid,nz=6,conductivity_W_mK=k)
    out=s.steady(1e7)
    assert abs(out.residual_W)/out.deposited_power_W<1e-8
    with pytest.raises(ValueError):DiskHeatSolver(base.grid,conductivity_W_mK=0.)
    with pytest.raises(ValueError):ThermalBoundary(rear_h_W_m2K=np.nan)
    with pytest.raises(ValueError):s.step(293.,1.,-1.)


def test_uniform_temperature_opd_double_pass_and_phase():
    s=solver();t=np.full(s.shape,298.15)
    op=thermal_opd(t,s);expected=9.1e-6*5e-3
    assert np.allclose(op[s.mask[0]],expected,rtol=1e-12)
    assert np.all(op[~s.mask[0]]==0)
    assert np.array_equal(thermal_opd(t,s,passes=2),2*op)
    assert np.allclose(phase_from_opd(op,2e-6),2*np.pi*op/2e-6)


def test_lens_fit_recovers_astigmatic_quadratic_with_tilt():
    g=Grid2D.square(81,.004);x,y=g.mesh
    power=np.array([[1.2,.15],[.15,.7]])
    opd=1e-7+2e-5*x-1e-5*y-.5*(power[0,0]*x*x+2*power[0,1]*x*y+power[1,1]*y*y)
    fit=fit_thermal_lens(opd,g,fit_radius_m=.001)
    assert np.allclose(fit.power_matrix_m1,power,rtol=1e-12,atol=1e-12)
    assert fit.residual_rms_m<1e-20


def test_zero_thermal_phase_reproduces_cold_cavity():
    g=Grid2D.square(128,.012);c=ThinDiskResonator();x,y=g.mesh
    f=lg0_field(x,y,c.waist_m,1)
    cold,oc=cavity_roundtrip(f,g,c)
    hot,hotoc=thermal_cavity_roundtrip(f,g,c,np.zeros(g.shape))
    assert np.allclose(hot,cold,rtol=1e-13,atol=1e-13)
    assert np.allclose(hotoc,oc,rtol=1e-13,atol=1e-13)


def test_thermal_piston_is_counted_twice_not_four_times():
    g=Grid2D.square(128,.012);c=ThinDiskResonator();x,y=g.mesh
    f=lg0_field(x,y,c.waist_m);cold,oc=cavity_roundtrip(f,g,c)
    piston=.1e-6;hot,hotoc=thermal_cavity_roundtrip(f,g,c,np.full(g.shape,piston))
    phi=2*np.pi*piston/c.wavelength_m
    assert np.allclose(hot,cold*np.exp(2j*phi),rtol=1e-10,atol=1e-13)
    assert np.allclose(hotoc,oc*np.exp(1j*phi),rtol=1e-10,atol=1e-13)


def test_hot_quadratic_mode_matches_fft_roundtrip():
    c=ThinDiskResonator();g=Grid2D.square(256,.012);x,y=g.mesh
    cold=gaussian_hot_cavity(c,0.)
    assert np.isclose(cold['waist_m'],c.waist_m,rtol=1e-13)
    power=.7;hot=gaussian_hot_cavity(c,power)
    f=lg0_field(x,y,hot['waist_m'])
    out,oc=thermal_cavity_roundtrip(f,g,c,-.5*power*(x*x+y*y))
    a=np.vdot(f,out)/np.vdot(f,f)
    assert np.linalg.norm(out-a*f)/np.linalg.norm(out)<3e-7
    assert np.isclose(abs(a)**2,c.passive_power_retention,rtol=1e-9)
    assert not gaussian_hot_cavity(c,100.)['stable']


def test_radial_mapping_conserves_signed_heat_with_different_z_resolution():
    m,r=radial_laser(radial_points=36,z_slices=3)
    s=solver(n=81,nz=8)
    q=np.broadcast_to(1e8*np.exp(-2*(r/.0005)**2)-1000.,m.density.shape).copy()
    q*=np.arange(1,4)[:,None]
    out,meta=radial_source_to_volume(r,q,m.area,m.dz,s)
    assert np.isclose(meta['power_before_W'],meta['power_after_W'],rtol=1e-13)
    assert np.all(out[~s.mask]==0)


def test_population_cycle_heat_and_total_optical_budget_close():
    m,r=radial_laser(radial_points=24,z_slices=2)
    f=np.zeros((3,m.nz,m.ns));f[2]=.3*np.exp(-2*(r/.0006)**2)
    ans=measure_heat_cycle(m,f,np.log([1e8]),3e-4,1e4,rtol=3e-7)
    b=ans['budget_J']
    assert abs(b['pump_map_residual'])<1e-15
    assert abs(b['local_first_law_residual'])/b['absorbed_pump']<3e-5
    assert abs(b['whole_system_residual'])/b['absorbed_pump']<3e-5
    assert b['output_coupler']>=0
    assert b['all_cavity_optical_losses']>b['output_coupler']


def test_cycle_average_refuses_unconverged_steady_state_claim():
    with pytest.raises(ValueError):cycle_averaged_heat(None,SimpleNamespace(periodic_converged=False))


def test_off_axis_source_produces_off_axis_3d_temperature():
    s=solver(n=65,nz=8,h=1e5);x,y=s.grid.mesh
    xy=5e7*np.exp(-2*((x-.8e-3)**2+y*y)/(.4e-3)**2)
    out=s.steady(xy[None])
    iz,iy,ix=np.unravel_index(np.argmax(out.temperature_K),s.shape)
    assert abs(s.grid.x[ix]-.8e-3)<s.grid.dx
    assert abs(s.grid.y[iy])<s.grid.dy
    assert iz==0
    assert np.min(out.temperature_K)>=293.15-1e-8
    assert abs(out.residual_W)/out.deposited_power_W<1e-8


def test_unsupported_branching_and_negative_intensity_are_rejected():
    n=np.array([0.,0.,1e24,1e26])
    with pytest.raises(ValueError):heat_rates(n,pump_intensity_W_m2=-1.)
    with pytest.raises(ValueError):heat_rates(n,replace(isolated(),beta78=.5))
    with pytest.raises(ValueError):ManifoldEnergies(centers_cm1=(1.,2.,3.,4.))
