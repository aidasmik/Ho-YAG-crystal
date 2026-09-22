"""Independent whole-project audit. Run against unmodified Stage 0--7 source.
This script reports failures without changing model source or existing tests.
"""
from __future__ import annotations
import json, subprocess, time, traceback
from pathlib import Path
import numpy as np
from hoyag.propagation import Grid2D, gaussian_beam, laguerre_gaussian, normalize_power, angular_spectrum_propagate
from hoyag.temporal import TimeGrid, gaussian_temporal_envelope, combine_spatial_temporal, apply_gdd, temporal_energy
from hoyag.populations import HoYAGFourLevelParams, four_level_rhs, H, C0, I5, I6, I7, I8
from hoyag.inhomogeneity import uniform_ho_density_field, propagate_single_pulse_inhomogeneous, smooth_random_ho_density_field
from hoyag.signal import validate_population_field, propagate_structured_signal_small_signal, propagate_structured_signal_saturated
from hoyag.spectroscopy import pump_absorption_cross_section_295K, effective_pump_absorption_cross_section_295K
from hoyag.heat import heat_budget, HeatSpectroscopy, BRANCHES
from hoyag.resonator import fluence_transfer, ThinDiskResonator
from hoyag.thermomechanics import DiskPlateMesh, tetra_kinematics, assemble_body, _recover, YAG_ELASTIC, solve_disk_plate
from hoyag.stress_optics import CubicElastoOptic, stress_impermeability, geometric_roundtrip_opd, HotDiskScreens
from hoyag.vector_cavity import VectorRoundTrip, normalize_vector
from hoyag.thermal import DiskThermalMesh
from hoyag.cooling_plate import DiskPlateHeatSolver, cooling_plate_mesh

ROOT=Path(__file__).resolve().parents[1]
RESULTS=[]
def check(name, fn):
    begin=time.perf_counter()
    try:
        ok,data=fn()
        result={'check':name,'passed':bool(ok),'data':data}
    except Exception as exc:
        result={'check':name,'passed':False,'error':repr(exc),'traceback':traceback.format_exc()}
    result['seconds']=time.perf_counter()-begin
    RESULTS.append(result)
    print('AUDIT_CHECK '+json.dumps(result,allow_nan=False),flush=True)


def isolated_params():
    return HoYAGFourLevelParams(tau5_s=1e99,tau6_s=1e99,tau7_s=1e99,M56_s1=0.,M67_s1=0.,M78_s1=0.,
                               k75_m3_s=0.,k76_m3_s=0.,C57_m3_s=0.,C67_m3_s=0.)


def plane_wave_exactness():
    grid=Grid2D.square(96,.004);x,y=grid.mesh;wave=2.0903e-6;n=1.799104526293235;z=.047
    kx=2*np.pi*7/.004;ky=-2*np.pi*11/.004
    field=np.exp(1j*(kx*x+ky*y));k=2*np.pi*n/wave
    out=angular_spectrum_propagate(field,grid,wave,z,refractive_index=n)
    exact=field*np.exp(1j*np.sqrt(k*k-kx*kx-ky*ky)*z)
    err=float(np.linalg.norm(out-exact)/np.linalg.norm(exact))
    return err<1e-10,{'relative_error':err}


def evanescent_backward_mask():
    grid=Grid2D.square(32,1e-6);f=gaussian_beam(grid,1e-7)
    try:
        with np.errstate(over='raise',invalid='raise'):
            result=angular_spectrum_propagate(f,grid,2e-6,-1e-3,bandlimit=True)
        finite=bool(np.all(np.isfinite(result)))
        return finite,{'finite':finite}
    except FloatingPointError as err:
        return False,{'error':str(err),'case':'evanescent modes are masked, yet exponential evaluated first'}


def gdd_analytic_complex_envelope():
    tg=TimeGrid.centered(4096,40e-12);T=1e-12;D=.5e-24;a=2*np.log(2)/T**2
    f=np.exp(-a*tg.tau**2).astype(complex)
    exact=(1-2j*a*D)**(-.5)*np.exp(-a*tg.tau**2/(1-2j*a*D))
    out=apply_gdd(f,tg,D)
    err=float(np.linalg.norm(out-exact)/np.linalg.norm(exact))
    return err<1e-10,{'complex_field_relative_error':err,'energy_error':float(abs(temporal_energy(out,tg)/temporal_energy(f,tg)-1))}


def parameter_validation():
    accepted=[]
    for values in [{'sigma_abs_laser_m2':-1e-24},{'tau7_s':-1e-3},{'N_total_m3':float('nan')},{'beta78':.2},{'M56_s1':-1.}]:
        try:HoYAGFourLevelParams(**values);accepted.append({k:str(v) for k,v in values.items()})
        except ValueError:pass
    return not accepted,{'unphysical_inputs_accepted':accepted}


def independent_rate_stoichiometry_and_heat():
    p=HoYAGFourLevelParams();N=np.array([.03,.04,.26,.67])*p.N_total_m3
    e=np.array(HeatSpectroscopy().manifold_energy_J);ip=2e8;il=3e8
    ep=H*C0/p.pump_wavelength_m;es=H*C0/p.laser_wavelength_m
    netp=ip*(p.sigma_abs_pump_m2*N[I8]-p.sigma_em_pump_m2*N[I7])/ep
    nets=il*(p.sigma_em_laser_m2*N[I7]-p.sigma_abs_laser_m2*N[I8])/es
    rates=[N[I5]*p.M56_s1,N[I6]*p.M67_s1,N[I7]*p.M78_s1,N[I7]**2*p.k75_m3_s,
           N[I7]**2*p.k76_m3_s,N[I5]*N[I8]*p.C57_m3_s,N[I6]*N[I8]*p.C67_m3_s]
    changes=[[-1,1,0,0],[0,-1,1,0],[0,0,-1,1],[1,0,-2,1],[0,1,-2,1],[-1,0,2,-1],[0,-1,2,-1]]
    rhs=np.array(changes).T@np.array(rates)
    heat=-float(e@rhs)+(ep-(e[I7]-e[I8]))*netp+((e[I7]-e[I8])-es)*nets
    branchrate=[p.beta56/p.tau5_s,p.beta57/p.tau5_s,p.beta58/p.tau5_s,p.beta67/p.tau6_s,p.beta68/p.tau6_s,p.beta78/p.tau7_s]
    for (upper,lower),rate in zip(BRANCHES,branchrate):
        rhs[upper]-=N[upper]*rate;rhs[lower]+=N[upper]*rate
    rhs[I7]+=netp-nets;rhs[I8]-=netp-nets
    actual=four_level_rhs(N,p,ip*p.sigma_abs_pump_m2/ep,ip*p.sigma_em_pump_m2/ep,
                         il*p.sigma_abs_laser_m2/es,il*p.sigma_em_laser_m2/es)
    re=float(np.linalg.norm(actual-rhs)/np.linalg.norm(rhs))
    he=float(abs(heat_budget(N,p,pump_intensity_W_m2=ip,signal_intensity_W_m2=il).heat_W_m3-heat)/abs(heat))
    return re<1e-12 and he<1e-12,{'reaction_stoichiometry_relative_error':re,'independent_event_heat_relative_error':he}


def population_layout_handoff():
    p=HoYAGFourLevelParams();grid=Grid2D.square(3,.001);t=TimeGrid.centered(96,60e-12)
    pulse=combine_spatial_temporal(normalize_power(np.ones(grid.shape),grid),gaussian_temporal_envelope(t,10e-12),grid,t)
    density=uniform_ho_density_field(grid,4,.001,p.N_total_m3)
    result=propagate_single_pulse_inhomogeneous(pulse,grid,t,1e-4,density,p,
                  spectral_absorption=False,include_passive_propagation=False,store_full_populations=True)
    raw=result.final_populations_by_slice
    if getattr(result,'population_axes',None)==('manifold','z','y','x'):
        reference=raw.copy()
    else:
        reference=np.moveaxis(raw,1,0)
    try:interpreted=validate_population_field(raw,density,grid)
    except ValueError as exc:return False,{'handoff_rejected':str(exc),'pump_axis_order':'z,manifold,y,x','signal_axis_order':'manifold,z,y,x'}
    error=float(np.max(abs(interpreted-reference))/p.N_total_m3)
    return error<1e-10,{'maximum_population_fraction_error':error,'actual_I7_mean_fraction':float(reference[I7].mean()/p.N_total_m3),
                       'interpreted_I7_mean_fraction':float(interpreted[I7].mean()/p.N_total_m3),'ambiguous_shape':list(raw.shape)}


def vortex_reference_phase():
    p=isolated_params();grid=Grid2D.square(96,.0016);t=TimeGrid.centered(32,80e-12)
    density=uniform_ho_density_field(grid,6,.018,p.N_total_m3);x,y=grid.mesh
    f7=.06+.52*np.exp(-2*(x*x+y*y)/(.000095**2))
    state=np.zeros((4,density.nz,*grid.shape));state[I7]=density.values_m3*f7;state[I8]=density.values_m3*(1-f7)
    f=laguerre_gaussian(grid,0,2,.000065)
    pulse=combine_spatial_temporal(f,gaussian_temporal_envelope(t,10e-12),grid,t)
    small=propagate_structured_signal_small_signal(f,grid,state,density,p)
    result=propagate_structured_signal_saturated(pulse,grid,t,1e-14,state,density,p,beta2_s2_per_m=0.)
    physical=abs(result.energy_gain/small.power_gain-1);reference=abs(result.small_signal_power_gain_reference/small.power_gain-1)
    return physical<1e-5 and reference<1e-5,{'true_complex_weak_gain':small.power_gain,'weak_energy_gain':result.energy_gain,
          'reported_reference_gain':result.small_signal_power_gain_reference,'reference_relative_error':reference,
          'actual_signal_solver_weak_limit_relative_error':physical,'wavelength_m':p.laser_wavelength_m,'waist_m':.000065,'length_m':.018}


def frantz_nodvik_convergence():
    p=isolated_params();grid=Grid2D.square(2,.001);t=TimeGrid.centered(512,80e-12)
    f=combine_spatial_temporal(normalize_power(np.ones(grid.shape),grid),gaussian_temporal_envelope(t,10e-12),grid,t)
    Fin=3e4;area=grid.nx*grid.dx*grid.ny*grid.dy;L=.018;fraction=.6
    g=p.N_total_m3*(p.sigma_em_laser_m2*fraction-p.sigma_abs_laser_m2*(1-fraction))
    fs=H*C0/p.laser_wavelength_m/(p.sigma_abs_laser_m2+p.sigma_em_laser_m2)
    exact=float(fluence_transfer(Fin,g*L,fs)/Fin);rows=[]
    for nz in [1,4,16,64]:
        density=uniform_ho_density_field(grid,nz,L,p.N_total_m3)
        state=np.zeros((4,nz,*grid.shape));state[I7]=fraction*density.values_m3;state[I8]=(1-fraction)*density.values_m3
        r=propagate_structured_signal_saturated(f,grid,t,Fin*area,state,density,p,include_passive_propagation=False)
        released=float(np.sum(state[I7]-r.final_populations_by_slice[I7])*grid.dx*grid.dy*L/nz*(H*C0/p.laser_wavelength_m))
        emitted=r.output_energy_J-r.input_energy_J
        rows.append({'nz':nz,'gain':r.energy_gain,'gain_error':abs(r.energy_gain/exact-1),'energy_mismatch':abs(emitted-released)/abs(released)})
    return rows[-1]['gain_error']<1e-3,{'exact_gain':exact,'resolution_sweep':rows}


def random_density_mean_contract():
    grid=Grid2D.square(12,.004);requested=1.52e26
    a=smooth_random_ho_density_field(grid,5,.001,requested,1.2,seed=12,minimum_fraction=.6)
    ratio=a.values_m3.mean()/requested
    return abs(ratio-1)<1e-10,{'requested_relative_rms':1.2,'minimum_fraction':.6,'mean_over_requested':float(ratio),
                              'realized_relative_rms':float(a.values_m3.std()/requested)}


def narrowband_absorption_approximation():
    p=HoYAGFourLevelParams();nu0=C0/p.pump_wavelength_m;rows=[]
    for ps in [1.,10.]:
        bw=2*np.log(2)/np.pi/(ps*1e-12);nu=nu0+np.linspace(-5*bw,5*bw,20001)
        w=np.exp(-4*np.log(2)*((nu-nu0)/bw)**2);w/=np.sum(w)
        sigma=pump_absorption_cross_section_295K(C0/nu);effective=float(w@sigma)
        for L in [.001,.002,.018]:
            exact=float(w@np.exp(-p.N_total_m3*L*sigma));scalar=float(np.exp(-p.N_total_m3*L*effective))
            rows.append({'duration_ps':ps,'length_m':L,'sigma_eff_m2':effective,'spectral_T':exact,'scalar_T':scalar,'relative_transmission_bias':(scalar-exact)/exact})
    return True,{'classification':'quantified approximation, not exact spectral propagation','cases':rows}


def carrier_detuning_convention():
    p=HoYAGFourLevelParams();t=TimeGrid.centered(8192,100e-12);pulse=gaussian_temporal_envelope(t,10e-12)
    delta=2e11;signal=pulse*np.exp(-2j*np.pi*delta*t.tau)
    actual=effective_pump_absorption_cross_section_295K(signal,t,p.pump_wavelength_m)
    nu0=C0/p.pump_wavelength_m;spectrum=abs(np.fft.fft(signal))**2;nu=nu0-t.frequency_hz
    expected=float(np.sum(spectrum*pump_absorption_cross_section_295K(C0/nu))/np.sum(spectrum))
    return abs(actual/expected-1)<1e-8,{'convention':'physical exp(ikz-iwt); positive detuning multiplies envelope by exp(-i deltaomega t)',
        'carrier_relative_detuning_Hz':delta,'calculated_sigma_m2':actual,'expected_sigma_m2':expected,'relative_error':float(actual/expected-1)}


def mechanical_affine_patch():
    mesh=DiskPlateMesh.make(nr=3,outer_rings=2,ntheta=12,nz_disk=2,nz_plate=2)
    A=np.array([[1e-4,2e-5,-3e-5],[4e-5,-7e-5,2e-5],[5e-5,8e-5,9e-5]])
    u=mesh.disk.nodes_m@A.T+np.array([2e-8,-3e-8,4e-8]);mat=YAG_ELASTIC
    k,f,cache=assemble_body(mesh.disk,mat,293.15);strain,stress,_=_recover(mesh.disk,u,mat,cache)
    expected=np.r_[np.diag(A),A[0,1]+A[1,0],A[0,2]+A[2,0],A[1,2]+A[2,1]]
    error=float(np.max(abs(strain-expected)))
    return error<1e-14,{'max_engineering_strain_error':error}


def unheated_pressure_compliance():
    mesh=DiskPlateMesh.make(nr=3,outer_rings=1,ntheta=12,nz_disk=2,nz_plate=2);pressure=2e5
    r=solve_disk_plate(mesh,293.15,293.15,front_pressure_Pa=pressure)
    V,_=tetra_kinematics(mesh.disk);force=pressure*V.sum()/.001
    e=float(abs(r.interface_force_on_disk_N[2]+force)/force)
    return e<1e-9,{'force_balance_relative_error':e,'max_interface_jump_m':float(np.linalg.norm(r.interface_jump_m,axis=-1).max())}


def isotropic_photoelastic_rotation_covariance():
    rng=np.random.default_rng(3);q=np.linalg.qr(rng.normal(size=(3,3)))[0]
    if np.linalg.det(q)<0:q[:,0]*=-1
    p=CubicElastoOptic(-.029,.0091,(-.029-.0091)/2);stress=np.array([1e7,-2e7,3e7,4e6,-3e6,2e6])
    a=stress_impermeability(stress,coefficients=p,crystal_axes=np.eye(3));b=stress_impermeability(stress,coefficients=p,crystal_axes=q)
    err=float(np.linalg.norm(a-b)/np.linalg.norm(a))
    return err<1e-12,{'relative_error':err,'interpretation':'checks p44 tensor-shear convention independently'}


def body_translation_optical_path():
    n=1.7991;u=1e-7;a=geometric_roundtrip_opd(u,u,index=n);b=geometric_roundtrip_opd(-u,0,index=n)
    return abs(a-2*u)<1e-20 and abs(b-2*(n-1)*u)<1e-20,{'rigid_translation_opd_m':float(a),'fixed_rear_expansion_opd_m':float(b)}


def thermal_network_energy():
    disk=DiskThermalMesh.disk(12,6,nphi=8);plate=cooling_plate_mesh(disk,nz=6,outer_rings=5)
    h=np.full((disk.nr,disk.nphi),1e5);h[:,0]=0.;h[:,1]=1e3
    model=DiskPlateHeatSolver(disk,plate,contact_conductance_W_m2K=h)
    q=np.full(disk.shape,3e5);q[:,:,1]*=2
    steady=model.steady(q);transient=model.advance(293.15,293.15,q,.01)
    err=max(steady.relative_balance_error,transient.relative_balance_error)
    return err<1e-9,{'steady_error':steady.relative_balance_error,'transient_error':transient.relative_balance_error,
        'insulated_patch_flux':float(abs(steady.interface_flux_W_m2[:,0]).max()),'plate_nonisothermal_range_K':float(np.ptp(steady.plate_temperature_K))}


def strong_contact_is_not_unilateral_contact():
    mesh=DiskPlateMesh.make(nr=3,outer_rings=1,ntheta=12,nz_disk=2,nz_plate=2)
    result=solve_disk_plate(mesh,310.,293.15,support='clamped');normal=result.interface_traction_on_disk_Pa[:,2]
    return True,{'classification':'documented bilateral bonded-interface limit','normal_traction_min_Pa':float(normal.min()),
                 'normal_traction_max_Pa':float(normal.max()),'has_both_traction_signs':bool(normal.min()<0<normal.max())}


def optics_grid_aliasing():
    c=ThinDiskResonator(air_gap_m=.25);L=.012;radius=.005;rows=[]
    for n in [128,192,256,384]:
        dx=L/n;worst=4*np.pi*radius*dx/(c.wavelength_m*c.output_mirror_radius_m)
        rows.append({'n':n,'spacing_m':dx,'mirror_phase_step_rad_at_disk_radius':worst,'nyquist_over_entire_disk':bool(worst<=np.pi)})
    return True,{'classification':'sampling risk; central low-order mode can still be resolved','cases':rows}


def optical_roundtrip_is_compatible_with_moving_reference():
    c=ThinDiskResonator(air_gap_m=.25);grid=Grid2D.square(128,.012)
    f=normalize_vector(np.stack([gaussian_beam(grid,c.waist_m),np.zeros(grid.shape)]))
    eye=np.broadcast_to(np.eye(2,dtype=complex),(*grid.shape,2,2)).copy();z=np.zeros(grid.shape)
    screens=HotDiskScreens(eye,eye,np.ones(grid.shape)*1e-7,z,z,z,z,z,c.wavelength_m)
    a,b=VectorRoundTrip(grid,c).propagate(f);aa,bb=VectorRoundTrip(grid,c,screens).propagate(f)
    error=float(np.linalg.norm(aa-a*np.exp(2j*np.pi*1e-7/c.wavelength_m)))
    return error<1e-12,{'field_error':error,'power_difference':float(abs(np.sum(abs(aa)**2)-np.sum(abs(a)**2)))}


def main():
    checks=[plane_wave_exactness,evanescent_backward_mask,gdd_analytic_complex_envelope,parameter_validation,
      independent_rate_stoichiometry_and_heat,population_layout_handoff,vortex_reference_phase,
      frantz_nodvik_convergence,random_density_mean_contract,narrowband_absorption_approximation,
      carrier_detuning_convention,mechanical_affine_patch,unheated_pressure_compliance,
      isotropic_photoelastic_rotation_covariance,body_translation_optical_path,thermal_network_energy,
      strong_contact_is_not_unilateral_contact,optics_grid_aliasing,optical_roundtrip_is_compatible_with_moving_reference]
    for fn in checks:check(fn.__name__,fn)
    report={'original_audited_base':'b55a1faf462f6433d4e9401a000682426de46467',
      'run_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'checks':RESULTS,'failed_checks':[v['check'] for v in RESULTS if not v['passed']]}
    out=ROOT/'results/deep_audit';out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('AUDIT_FINAL '+json.dumps({k:v for k,v in report.items() if k!='checks'}),flush=True)
    if report['failed_checks']:
        raise SystemExit('Independent audit failures: '+', '.join(report['failed_checks']))
if __name__=='__main__':main()
