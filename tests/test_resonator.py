import math
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from hoyag.resonator import (ThinDiskResonator, cavity_roundtrip, lg0_field,
                            fluence_transfer, radial_laser, ModalThinDiskLaser)
from hoyag.populations import HoYAGFourLevelParams, H, C0, I7


def grid2d(n=256,size=.012):
    d=size/n
    x=(np.arange(n)-(n-1)/2)*d
    return SimpleNamespace(nx=n,ny=n,dx=d,dy=d,shape=(n,n),
                           mesh=np.meshgrid(x,x,indexing='xy'),
                           fx=np.fft.fftfreq(n,d),fy=np.fft.fftfreq(n,d))


def test_cavity_stability_and_lengths():
    c=ThinDiskResonator()
    assert c.disk_diameter_m==.01 and c.disk_thickness_m==.001
    assert c.stable and 0<c.stability_product<1
    assert np.isclose(c.reduced_length_m,.2+.001/c.host_index)
    assert np.isclose(c.roundtrip_time_s,2*(.2+c.host_group_index*.001)/C0)
    assert not replace(c,air_gap_m=.6).stable
    with pytest.raises(ValueError):
        _=replace(c,air_gap_m=.6).waist_m


@pytest.mark.parametrize('charge',[0,1,2,-1])
def test_fft_roundtrip_recovers_lg_eigenmode(charge):
    c=ThinDiskResonator(); gr=grid2d();x,y=gr.mesh
    f=lg0_field(x,y,c.waist_m,charge)
    out,oc=cavity_roundtrip(f,gr,c)
    amplitude=np.vdot(f,out)/np.vdot(f,f)
    mismatch=np.linalg.norm(out-amplitude*f)/np.linalg.norm(out)
    expected=np.sqrt(c.passive_power_retention)*np.exp(-2j*(abs(charge)+1)*np.arctan(c.reduced_length_m/c.rayleigh_range_m))
    assert mismatch<2e-7
    assert abs(amplitude-expected)<2e-7
    assert np.isclose(np.sum(abs(oc)**2)/np.sum(abs(f)**2),c.output_transmission,rtol=1e-10)


def test_roundtrip_counts_two_gain_traversals():
    c=ThinDiskResonator();gr=grid2d();x,y=gr.mesh
    f=lg0_field(x,y,c.waist_m)
    back,_=cavity_roundtrip(f,gr,c,c.threshold_gain_m1*c.disk_thickness_m)
    assert np.isclose(np.sum(abs(back)**2)/np.sum(abs(f)**2),1,rtol=1e-10)


def test_cold_cavity_ringdown_and_lifetime():
    c=ThinDiskResonator();gr=grid2d(128);x,y=gr.mesh
    f=lg0_field(x,y,c.waist_m);p0=np.sum(abs(f)**2)
    for _ in range(40):
        f,_=cavity_roundtrip(f,gr,c)
    ratio=np.sum(abs(f)**2)/p0
    assert np.isclose(ratio,np.exp(-40*c.roundtrip_time_s/c.photon_lifetime_s),rtol=1e-8)


def test_fluence_map_weak_limit_and_zero():
    fs=5e4
    f=np.array([0.,1e-7,.01])
    out=fluence_transfer(f,-.2,fs)
    assert out[0]==0
    assert np.allclose(out[1:],f[1:]*np.exp(-.2),rtol=1e-7,atol=0)


def test_fluence_map_high_gain_and_high_fluence_stable():
    f=np.array([1e-8,5.,1e6]);fs=10.
    assert np.allclose(fluence_transfer(f,0,fs),f,rtol=1e-12)
    assert np.all(fluence_transfer(f,-2,fs)<=f)
    assert np.all(fluence_transfer(f,2,fs)>=f)
    with pytest.raises(ValueError):
        fluence_transfer(-1,0,fs)


def test_radial_quadrature_covers_whole_disk_without_beam_renormalization():
    m,r=radial_laser(radial_points=36)
    assert np.isclose(np.sum(m.area),np.pi*(.005)**2,rtol=1e-12)
    assert np.allclose(m.modes@m.area,1,rtol=2e-7)
    assert np.isclose(m.pump@m.area,1,rtol=2e-7)
    assert np.max(r)>.0049


def test_pump_kick_conserves_photons_and_energy():
    m,r=radial_laser(radial_points=24,z_slices=4)
    z=np.zeros((3,m.nz,m.ns));f,absorbed,escaped,loss=m.pump_kick(z,1e-3)
    assert np.isclose(absorbed+escaped+loss,1e-3,rtol=1e-12)
    stored=np.sum(f[2]*m.density*m.volume)*m.ep
    assert np.isclose(stored,absorbed,rtol=1e-12)
    assert f[0].max()==0 and f[1].max()==0
    assert absorbed>0 and escaped>0 and loss>0


def test_pump_weak_double_pass_absorption():
    m,r=radial_laser(radial_points=24,z_slices=4)
    f,absorbed,escaped,loss=m.pump_kick(np.zeros((3,m.nz,m.ns)),1e-10)
    t=np.exp(-m.sa*m.params.N_total_m3*m.cavity.disk_thickness_m)
    expected=(1-t)*(1+m.cavity.pump_hr_reflectivity*t)
    assert abs(absorbed/1e-10-expected)<1e-6


def test_pump_local_fraction_capped_by_pump_transparency():
    m,r=radial_laser(radial_points=24,z_slices=4)
    f,ab,esc,loss=m.pump_kick(np.zeros((3,m.nz,m.ns)),10.)
    limit=m.sa/(m.sa+m.params.sigma_em_pump_m2)
    assert np.min(f)>=0 and np.max(f[2])<=limit+1e-10


def test_uniform_threshold_net_photon_growth_zero():
    m,r=radial_laser(radial_points=24,z_slices=2)
    p=m.params
    f7=(m.cavity.threshold_gain_m1/p.N_total_m3+p.sigma_abs_laser_m2)/(p.sigma_em_laser_m2+p.sigma_abs_laser_m2)
    st=np.zeros((3,m.nz,m.ns));st[2]=f7
    assert np.allclose(m.log_roundtrip_gain(st),m.cavity.logarithmic_loss,rtol=2e-7)


def test_stimulated_energy_coupling_has_double_pass_factor():
    p=HoYAGFourLevelParams(tau5_s=1e99,tau6_s=1e99,tau7_s=1e99,
                         M56_s1=0,M67_s1=0,M78_s1=0,
                         k75_m3_s=0,k76_m3_s=0,C57_m3_s=0,C67_m3_s=0)
    m,r=radial_laser(radial_points=24,z_slices=2,params=p)
    f=np.zeros((3,m.nz,m.ns));f[2]=.3
    photons=1e10
    y=np.r_[f.ravel(),np.log(photons),0.]
    dy=m.rhs(0,y)
    dn7=dy[2*m.nc:3*m.nc].reshape(m.nz,m.ns)*m.density
    population_power=-np.sum(dn7*m.volume)*m.es
    optical_power=(m.log_roundtrip_gain(f)[0]/m.trt)*photons*m.es
    assert np.isclose(population_power,optical_power,rtol=1e-12)


def test_modes_compete_via_shared_population_and_degenerate_sign():
    m,r=radial_laser(radial_points=36,z_slices=2,charges=(0,1,-1))
    assert np.array_equal(m.modes[1],m.modes[2])
    f=np.zeros((3,m.nz,m.ns));f[2]=.3*np.exp(-(r/.0005)**2)
    gg=m.log_roundtrip_gain(f)
    assert gg[0]>gg[1] and np.isclose(gg[1],gg[2],rtol=1e-14,atol=0)


def test_short_dynamic_run_and_integrated_outcoupling():
    m,r=radial_laser(radial_points=24,z_slices=1)
    res=m.run(1e-4,max_cycles=3,min_cycles=3,rtol=5e-6)
    assert np.all(res.history[:,10:]>0)
    assert not res.periodic_converged
    m.check_fractions(res.fractions_before_next_pump)
    assert np.all(np.isfinite(res.waveform))
    assert np.all(res.history[:,2]+res.history[:,3]+res.history[:,4]-1e-4<1e-12)


def test_reject_long_pulses_and_invalid_parameters():
    m,_=radial_laser(radial_points=24,z_slices=1)
    with pytest.raises(ValueError):
        m.run(1e-4,max_cycles=1,min_cycles=1,pump_fwhm_s=1e-6)
    with pytest.raises(ValueError):
        ThinDiskResonator(output_transmission=0)
    with pytest.raises(ValueError):
        radial_laser(charges=(0,0))
    with pytest.raises(ValueError):
        ThinDiskResonator(disk_thickness_m=float('nan'))


def test_analytic_jacobian_matches_finite_difference():
    m,r=radial_laser(radial_points=24,z_slices=1,charges=(0,1))
    f=np.zeros((3,m.nz,m.ns));f[0]=1e-4;f[1]=2e-3;f[2]=.2
    yy=np.r_[f.ravel(),np.log([1e9,1e8]),[0.,0.]]
    jac=m.jacobian(0,yy).toarray()
    numeric=np.empty_like(jac)
    for i in range(m.nvar):
        step=1e-6 if i<3*m.nc else 1e-5
        hi=yy.copy();lo=yy.copy();hi[i]+=step;lo[i]-=step
        numeric[:,i]=(m.rhs(0,hi)-m.rhs(0,lo))/(2*step)
    assert np.max(np.abs(numeric-jac)/(1+np.abs(jac)))<2e-3
