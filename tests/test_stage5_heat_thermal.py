import numpy as np
import pytest
from dataclasses import replace
from hoyag.heat import HeatSpectroscopy, heat_budget, ion_energy_density, fluorescence_power_density, pump_kick_budget
from hoyag.populations import HoYAGFourLevelParams, H,C0,I5,I6,I7,I8
from hoyag.thermal import DiskThermalMesh,DiskCooling,ThermalBoundary,DiskHeatSolver,nonlinear_steady
from hoyag.thermal_optics import thermal_opd,fit_radial_thermal_lens,hot_gaussian_mode,thermal_cavity_roundtrip,normalized_overlap
from hoyag.resonator import ThinDiskResonator,lg0_field,cavity_roundtrip
from types import SimpleNamespace


def isolated():
    return HoYAGFourLevelParams(tau5_s=1e99,tau6_s=1e99,tau7_s=1e99,M56_s1=0.,M67_s1=0.,M78_s1=0.,k75_m3_s=0.,k76_m3_s=0.,C57_m3_s=0.,C67_m3_s=0.)

def optical_grid(n=256):
    size=.012;dx=size/n;x=(np.arange(n)-(n-1)/2)*dx
    return SimpleNamespace(nx=n,ny=n,dx=dx,dy=dx,shape=(n,n),mesh=np.meshgrid(x,x,indexing='xy'),fx=np.fft.fftfreq(n,dx),fy=np.fft.fftfreq(n,dx))


def test_ground_state_zero_heat():
    p=HoYAGFourLevelParams();n=np.array([0.,0.,0.,p.N_total_m3])
    assert heat_budget(n,p).heat_W_m3==0


def test_pump_storage_not_counted_as_heat():
    p=isolated();n=np.array([0.,0.,0.,p.N_total_m3]);s=HeatSpectroscopy()
    b=heat_budget(n,p,s,pump_intensity_W_m2=1e8)
    expected=b.pump_net_W_m3*(1-s.manifold_energy_J[I7]/(H*C0/p.pump_wavelength_m))
    assert np.isclose(b.heat_W_m3,expected,rtol=1e-12)
    assert b.heat_W_m3<b.pump_net_W_m3


def test_stimulated_emission_and_storage_balance():
    p=isolated();n=np.array([0,0,.6,.4])*p.N_total_m3;s=HeatSpectroscopy()
    b=heat_budget(n,p,s,signal_intensity_W_m2=1e10)
    assert b.signal_net_W_m3>0 and b.stored_rate_W_m3<0
    assert np.isclose(b.heat_W_m3,-b.stored_rate_W_m3-b.signal_net_W_m3,rtol=1e-12)


def test_radiative_lifetimes_are_upper_manifold_specific():
    p=isolated();p=replace(p,tau5_s=.001,tau6_s=.003,tau7_s=.01)
    s=HeatSpectroscopy();n=np.array([.1,.2,.3,.4])*p.N_total_m3
    expected=n[I5]*s.manifold_energy_J[I5]/p.tau5_s+n[I6]*s.manifold_energy_J[I6]/p.tau6_s+n[I7]*s.manifold_energy_J[I7]/p.tau7_s
    # Telescoping branch energies give weighted differences, not E_i alone when final manifold excited.
    b=s.branch_photon_energy_J
    expected=n[0]*(p.beta56*b[0]+p.beta57*b[1]+p.beta58*b[2])/p.tau5_s+n[1]*(p.beta67*b[3]+p.beta68*b[4])/p.tau6_s+n[2]*b[5]/p.tau7_s
    assert np.isclose(fluorescence_power_density(n,p,s),expected,rtol=1e-14)
    assert abs(heat_budget(n,p,s).heat_W_m3)<1e-6


def test_nonradiative_i7_decay_is_heat():
    p=replace(isolated(),M78_s1=20.9);s=HeatSpectroscopy();n=np.array([0,0,.2,.8])*p.N_total_m3
    assert np.isclose(heat_budget(n,p,s).heat_W_m3,p.M78_s1*n[I7]*s.manifold_energy_J[I7],rtol=1e-12)


def test_etu_heat_channel_is_not_clipped():
    p=replace(isolated(),k75_m3_s=3.8e-24);s=HeatSpectroscopy();n=np.array([0,0,.3,.7])*p.N_total_m3
    expected=(2*s.manifold_energy_J[I7]-s.manifold_energy_J[I5])*p.k75_m3_s*n[I7]**2
    assert expected<0
    assert np.isclose(heat_budget(n,p,s).heat_W_m3,expected,rtol=1e-12)


def test_random_full_heat_ledger_closes():
    p=HoYAGFourLevelParams();rng=np.random.default_rng(3);n=rng.random((4,3,4));n*=p.N_total_m3/n.sum(axis=0)
    b=heat_budget(n,p,pump_intensity_W_m2=3e9,signal_intensity_W_m2=2e9,background_W_m3=5.)
    assert np.max(abs(b.closure_W_m3))/np.max(abs(b.pump_net_W_m3))<1e-13


def test_short_pump_kick_heat_conserves_energy():
    p=isolated();a=np.array([0,0,0,1.])*p.N_total_m3;b=np.array([0,0,.1,.9])*p.N_total_m3
    k=pump_kick_budget(a,b,H*C0/p.pump_wavelength_m)
    assert np.isclose(k['pump_J_m3'],k['stored_J_m3']+k['heat_J_m3'],rtol=1e-14)


def test_circular_volume_exact():
    for nphi in (1,8):
        m=DiskThermalMesh.disk(12,5,nphi)
        assert np.isclose(m.volumes_m3.sum(),np.pi*.005**2*.001,rtol=1e-14)


def test_no_heating_returns_bath_temperature():
    m=DiskThermalMesh.disk(12,6)
    r=DiskHeatSolver(m).steady(0.)
    assert np.allclose(r.temperature_K,293.15,rtol=0,atol=2e-8)


@pytest.mark.parametrize('h',[np.inf,1e5,1e3])
def test_uniform_heat_matches_1d_thin_disk(h):
    m=DiskThermalMesh.disk(4,64);q=1e8;k=14.;L=.001;Tb=293.15
    cool=DiskCooling(rear=ThermalBoundary(Tb,h))
    r=DiskHeatSolver(m,k,cool).steady(q)
    expected=Tb+(0 if np.isinf(h) else q*L/h)+q*(L**2-m.z_m**2)/(2*k)
    err=np.max(abs(r.temperature_K[:,0,0]-expected))
    assert err<q*(L/64)**2/(5*k)
    assert r.relative_balance_error<1e-10
    assert np.isclose(r.boundary_heat_W['rear'],q*m.volumes_m3.sum(),rtol=1e-10)


def test_radial_uniform_heat_rim_cooling_parabola():
    m=DiskThermalMesh.disk(120,2,radial_exponent=1.)
    cool=DiskCooling(rear=ThermalBoundary(),rim=ThermalBoundary(293.15,np.inf))
    q=1e7;k=14.;r=DiskHeatSolver(m,k,cool).steady(q)
    expected=293.15+q*(.005**2-m.r_m**2)/(4*k)
    assert max(abs(r.temperature_K[0,:,0]-expected))<.001


def test_3d_azimuthal_heat_is_not_averaged_away():
    m=DiskThermalMesh.disk(12,5,12)
    q=1e7*(1+.4*np.cos(m.phi_rad))[None,None,:]
    r=DiskHeatSolver(m).steady(q)
    assert np.ptp(r.temperature_K[0,-2])>.005
    assert r.relative_balance_error<1e-9
    # Rotating heat rotates temperature on the periodic phi mesh.
    rr=DiskHeatSolver(m).steady(np.roll(np.broadcast_to(q,m.shape),3,axis=2))
    assert np.allclose(rr.temperature_K,np.roll(r.temperature_K,3,axis=2),atol=2e-8)


def test_transient_insulated_energy_conservation():
    m=DiskThermalMesh.disk(8,5);cool=DiskCooling(rear=ThermalBoundary())
    solver=DiskHeatSolver(m,cooling=cool);q=1e7;dt=.003
    r=solver.advance(293.15,q,dt)
    expected=293.15+dt*q/(4560*680)
    assert np.allclose(r.temperature_K,expected,atol=1e-9)
    assert r.relative_balance_error<1e-8
    with pytest.raises(ValueError): solver.steady(q)


def test_transient_cooling_budget_and_convergence():
    m=DiskThermalMesh.disk(5,5);solver=DiskHeatSolver(m);q=1e7
    target=solver.steady(q).temperature_K;t=np.full(m.shape,293.15)
    for _ in range(60):
        r=solver.advance(t,q,.02);t=r.temperature_K
        assert r.relative_balance_error<1e-8
    assert np.max(abs(t-target))<1e-3


def test_nonlinear_constant_k_reproduces_linear():
    m=DiskThermalMesh.disk(8,6)
    r,it=nonlinear_steady(m,1e7,lambda t:14+0*t)
    assert np.max(abs(r.temperature_K-DiskHeatSolver(m).steady(1e7).temperature_K))<1e-6


def test_opd_units_and_double_pass():
    m=DiskThermalMesh.disk(8,6)
    a=thermal_opd(m,303.15)
    assert np.allclose(a,9.1e-6*10*.001,rtol=1e-14,atol=0)
    assert np.array_equal(thermal_opd(m,303.15,passes=2),2*a)


def test_lens_fit_recovers_sign_and_focal_length():
    r=np.linspace(0,1e-3,30);opd=2e-7-r*r/(2*.8)
    f=fit_radial_thermal_lens(r,opd,fit_radius_m=.001)
    assert np.isclose(f.single_pass_focal_length_m,.8,rtol=1e-13)
    assert f.weighted_rms_residual_m<1e-20


def test_hot_mode_zero_lens_is_cold_cavity():
    c=ThinDiskResonator();h=hot_gaussian_mode(c,0)
    assert np.isclose(h.disk_radius_m,c.waist_m,rtol=1e-14)
    assert np.isclose(h.output_radius_m,c.output_mirror_spot_m,rtol=1e-14)


def test_zero_phase_and_double_traversal_piston():
    c=ThinDiskResonator();g=optical_grid();x,y=g.mesh;a=lg0_field(x,y,c.waist_m)
    cold,oc=cavity_roundtrip(a,g,c)
    back,out=thermal_cavity_roundtrip(a,g,c,0.)
    assert np.array_equal(back,cold) and np.array_equal(out,oc)
    phase=.3;opd=phase*c.wavelength_m/(2*np.pi)
    back,out=thermal_cavity_roundtrip(a,g,c,opd)
    assert np.allclose(back,cold*np.exp(2j*phase),rtol=1e-10,atol=1e-12)
    assert np.allclose(out,oc*np.exp(1j*phase),rtol=1e-10,atol=1e-12)


def test_parabolic_hot_fft_eigenmode():
    c=ThinDiskResonator();g=optical_grid();x,y=g.mesh;kappa=1.
    h=hot_gaussian_mode(c,kappa)
    a=lg0_field(x,y,h.disk_radius_m)*np.exp(1j*np.pi*(x*x+y*y)/c.wavelength_m*np.real(1/h.q_at_disk_m))
    back,_=thermal_cavity_roundtrip(a,g,c,-.5*kappa*(x*x+y*y))
    assert 1-normalized_overlap(a,back)<1e-10


def test_invalid_input_rejected():
    with pytest.raises(ValueError): DiskThermalMesh.disk(nphi=2)
    with pytest.raises(ValueError): ThermalBoundary(conductance_W_m2K=-1)
    with pytest.raises(ValueError): DiskHeatSolver(DiskThermalMesh.disk(2,2),conductivity_W_mK=0.)
    with pytest.raises(ValueError): heat_budget([0,0,0,1],pump_intensity_W_m2=-1)
    with pytest.raises(ValueError): hot_gaussian_mode(ThinDiskResonator(),100.)


def test_front_surface_heat_gives_linear_axial_temperature():
    m=DiskThermalMesh.disk(4,16);flux=1e4;k=14.
    cool=DiskCooling(rear=ThermalBoundary(293.15,np.inf))
    sol=DiskHeatSolver(m,k,cool,front_heat_flux_W_m2=flux).steady(0.)
    expected=293.15+flux*(.001-m.z_m)/k
    assert np.allclose(sol.temperature_K[:,0,0],expected,atol=1e-9)
    assert sol.relative_balance_error<1e-10


def test_rear_coating_heat_respects_surface_contact_partition():
    m=DiskThermalMesh.disk(4,12);flux=1e4;h=2e4
    cool=DiskCooling(rear=ThermalBoundary(293.15,h))
    sol=DiskHeatSolver(m,cooling=cool,rear_heat_flux_W_m2=flux).steady(0.)
    assert np.allclose(sol.temperature_K,293.15+flux/h,atol=1e-9)
    assert sol.relative_balance_error<1e-10
    perfect=DiskHeatSolver(m,cooling=DiskCooling(rear=ThermalBoundary(293.15,np.inf)),rear_heat_flux_W_m2=flux).steady(0.)
    assert np.array_equal(perfect.temperature_K,np.full(m.shape,293.15))
    assert perfect.relative_balance_error==0.


def test_heterogeneous_conductivity_conserves_heat_flux():
    m=DiskThermalMesh.disk(3,20);k=np.full(m.shape,14.);k[:10]=7.;flux=1e4
    sol=DiskHeatSolver(m,k,DiskCooling(rear=ThermalBoundary(293.15,np.inf)),front_heat_flux_W_m2=flux).steady(0.)
    expected=293.15+np.where(m.z_m<.0005,flux*((.0005-m.z_m)/7+.0005/14),flux*(.001-m.z_m)/14)
    assert np.allclose(sol.temperature_K[:,0,0],expected,atol=1e-9)
    assert sol.relative_balance_error<1e-10


def test_3d_manufactured_solution_refines_toward_analytic_field():
    errors=[]
    for nr,nz,nphi in ((8,6,12),(16,12,24)):
        m=DiskThermalMesh.disk(nr,nz,nphi,radial_exponent=1.)
        R=m.r_edges_m[-1];L=m.z_edges_m[-1];a=40/(R**4*L**2)
        r=m.r_m[None,:,None];z=m.z_m[:,None,None];phi=m.phi_rad[None,None,:]
        exact=293.15+a*(R**2-r**2)*r**2*np.cos(2*phi)*(L**2-z**2)
        q=14*a*(12*r**2*(L**2-z**2)+2*r**2*(R**2-r**2))*np.cos(2*phi)
        cool=DiskCooling(rear=ThermalBoundary(293.15,np.inf),rim=ThermalBoundary(293.15,np.inf))
        out=DiskHeatSolver(m,cooling=cool).steady(q)
        errors.append(np.max(abs(out.temperature_K-exact)))
    assert errors[1]<.5*errors[0] and errors[1]<.2
