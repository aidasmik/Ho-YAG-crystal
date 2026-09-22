"""Regressions for independently reproduced Stage 0--7 audit failures."""
from pathlib import Path
import inspect
import json
import numpy as np
import pytest
from hoyag.population_state import validate_populations, convert_population_layout, PopulationField
from hoyag.pump_source import PumpSource, GAUSSIAN_TBP
from hoyag.populations import (HoYAGFourLevelParams, I7, I8, H, C0,
    propagate_single_pulse_hoyag, stimulated_rates_from_intensity)
from hoyag.inhomogeneity import (uniform_ho_density_field, propagate_single_pulse_inhomogeneous,
    smooth_random_ho_density_field, ho_density_statistics)
from hoyag.propagation import (Grid2D, gaussian_beam, laguerre_gaussian, normalize_power,
    angular_spectrum_propagate, angular_spectrum_transfer)
from hoyag.temporal import (TimeGrid, gaussian_temporal_envelope, combine_spatial_temporal,
    propagate_spatiotemporal)
from hoyag.signal import (validate_population_field, propagate_structured_signal_small_signal,
    propagate_structured_signal_saturated, propagate_structured_signal_frozen)
from hoyag.spectroscopy import (effective_pump_absorption_cross_section_295K,
    pump_absorption_cross_section_295K, apply_frozen_spectral_absorption, spectral_attenuation_diagnostic)
from hoyag.density_statistics import bounded_mean_rms
from hoyag.numerical_quality import check_spectral_scalar_limit, ModelValidityWarning, cavity_sampling_diagnostic
from hoyag.resonator import ThinDiskResonator, radial_laser
from hoyag.thermal import DiskThermalMesh
from hoyag.thermal_resonator import modal_laser_on_thermal_mesh
from hoyag.coupled_resonator import run_coupled_hot_cavity


def pulse(grid,time,duration=10e-12):
    return combine_spatial_temporal(normalize_power(np.ones(grid.shape),grid),
           gaussian_temporal_envelope(time,duration),grid,time)


def isolated_params():
    return HoYAGFourLevelParams(tau5_s=1e99,tau6_s=1e99,tau7_s=1e99,
        M56_s1=0.,M67_s1=0.,M78_s1=0.,k75_m3_s=0.,k76_m3_s=0.,C57_m3_s=0.,C67_m3_s=0.)


@pytest.mark.parametrize('nz',[3,4,5])
@pytest.mark.parametrize('inhomogeneous',[False,True])
def test_single_pump_to_signal_actual_handoff(nz,inhomogeneous):
    p=HoYAGFourLevelParams();g=Grid2D.square(3,.001);t=TimeGrid.centered(96,60e-12)
    d=uniform_ho_density_field(g,nz,.001,p.N_total_m3);field=pulse(g,t)
    opts=dict(params=p,spectral_absorption=False,include_passive_propagation=False,store_full_populations=True)
    if inhomogeneous:
        result=propagate_single_pulse_inhomogeneous(field,g,t,1e-4,d,**opts)
    else:
        result=propagate_single_pulse_hoyag(field,g,t,1e-4,.001,nz,**opts)
    assert result.population_axes==('manifold','z','y','x')
    assert result.final_populations_by_slice.shape==(4,nz,*g.shape)
    state=validate_population_field(result.final_populations_by_slice,d,g)
    upper_bound=p.sigma_abs_pump_m2*100/(H*C0/p.pump_wavelength_m)
    assert 0 < state[I7].mean()/p.N_total_m3 < upper_bound*1.001
    assert state[I7].max()/p.N_total_m3 < .002
    out=propagate_structured_signal_small_signal(np.ones(g.shape),g,state,d,p,include_passive_propagation=False)
    assert out.power_gain<1


def test_four_by_four_legacy_array_rejected_and_explicitly_convertible():
    p=HoYAGFourLevelParams();g=Grid2D.square(3,.001);d=uniform_ho_density_field(g,4,.001,p.N_total_m3)
    correct=np.zeros((4,4,*g.shape));correct[I7]=.001*d.values_m3;correct[I8]=.999*d.values_m3
    legacy=np.moveaxis(correct,0,1)
    assert legacy.shape==correct.shape
    with pytest.raises(ValueError,match='local population sum'):
        validate_population_field(legacy,d,g)
    migrated=convert_population_layout(legacy,source_layout='z,manifold,y,x')
    tagged=PopulationField.from_legacy_z_major(legacy,d.values_m3)
    assert np.allclose(migrated,correct,rtol=0,atol=0)
    assert np.allclose(validate_population_field(tagged,d,g),correct,rtol=1e-12)
    assert not tagged.values_m3.flags.writeable


def test_no_silent_normalization_or_input_mutation():
    density=np.full((2,3,3),1.52e26);n=np.zeros((4,*density.shape));n[I8]=density
    before=n.copy();out=validate_populations(n,density)
    assert np.array_equal(n,before) and out is not n
    n[I8]*=1.0001
    with pytest.raises(ValueError):validate_populations(n,density)
    n[I8]=density*(1+1e-13)
    assert np.allclose(validate_populations(n,density).sum(axis=0),density,rtol=1e-15)


@pytest.mark.parametrize('kind',['nan','negative','undoped'])
def test_population_validation_checks_local_physical_constraints(kind):
    d=np.ones((2,2,2))*1e26;n=np.zeros((4,*d.shape));n[I8]=d
    if kind=='nan':n[0,0,0,0]=np.nan
    elif kind=='negative':n[0,0,0,0]=-1e24;n[I8,0,0,0]+=1e24
    else:d[0,0,0]=0
    with pytest.raises(ValueError):validate_populations(n,d)


@pytest.mark.parametrize('nz',[3,4,5])
def test_layout_conversion_requires_declared_axes(nz):
    array=np.zeros((nz,4,2,2))
    assert convert_population_layout(array,source_layout='z,manifold,y,x').shape==(4,nz,2,2)
    with pytest.raises(TypeError):convert_population_layout(array)
    with pytest.raises(ValueError):convert_population_layout(array,source_layout='guess')


def test_vortex_frozen_reference_keeps_original_complex_phase():
    p=isolated_params();g=Grid2D.square(96,.0016);t=TimeGrid.centered(32,80e-12)
    d=uniform_ho_density_field(g,6,.018,p.N_total_m3);x,y=g.mesh
    excited=.06+.52*np.exp(-2*(x*x+y*y)/(.000095**2))
    n=np.zeros((4,d.nz,*g.shape));n[I7]=d.values_m3*excited;n[I8]=d.values_m3*(1-excited)
    f=laguerre_gaussian(g,0,2,.000065)
    field=combine_spatial_temporal(f,gaussian_temporal_envelope(t,10e-12),g,t)
    expected=propagate_structured_signal_small_signal(f,g,n,d,p)
    sat=propagate_structured_signal_saturated(field,g,t,1e-14,n,d,p,beta2_s2_per_m=0.)
    assert np.isclose(sat.energy_gain,expected.power_gain,rtol=1e-9)
    assert np.isclose(sat.small_signal_power_gain_reference,expected.power_gain,rtol=1e-9)


def test_nonseparable_spacetime_reference_keeps_phase_and_gvd():
    p=isolated_params();g=Grid2D.square(24,.003);t=TimeGrid.centered(64,40e-12)
    d=uniform_ho_density_field(g,3,.001,p.N_total_m3);x,y=g.mesh
    f7=.3+.1*np.tanh(x/.0004)
    n=np.zeros((4,d.nz,*g.shape));n[I7]=d.values_m3*f7;n[I8]=d.values_m3*(1-f7)
    a=gaussian_temporal_envelope(t,3e-12,delay_s=-4e-12)
    b=gaussian_temporal_envelope(t,3e-12,delay_s=4e-12,gdd_s2=.1e-24)
    field=a[:,None,None]*laguerre_gaussian(g,0,1,.0004)+b[:,None,None]*laguerre_gaussian(g,0,-1,.0005)*1j
    frozen=propagate_structured_signal_frozen(field,g,t,1e-14,n,d,p,beta2_s2_per_m=1e-21)
    sat=propagate_structured_signal_saturated(field,g,t,1e-14,n,d,p,beta2_s2_per_m=1e-21)
    assert np.isclose(frozen.energy_gain,sat.energy_gain,rtol=1e-9)
    assert np.linalg.norm(frozen.field_out-sat.field_out)/np.linalg.norm(frozen.field_out)<1e-9
    assert np.isclose(sat.small_signal_power_gain_reference,frozen.energy_gain,rtol=1e-12)


@pytest.mark.parametrize('detuning',[2e11,-2e11])
def test_positive_and_negative_optical_detuning(detuning):
    lam=1.9077e-6;t=TimeGrid.centered(8192,100e-12)
    a=gaussian_temporal_envelope(t,10e-12)*np.exp(-2j*np.pi*detuning*t.tau)
    sigma=effective_pump_absorption_cross_section_295K(a,t,lam)
    physical_center=C0/(C0/lam+detuning)
    expected=PumpSource(physical_center,10e-12).effective_absorption_m2()
    assert np.isclose(sigma,expected,rtol=2e-8,atol=0)


@pytest.mark.parametrize('duration',[1e-12,10e-12])
def test_shared_source_agrees_with_fft_envelope(duration):
    t=TimeGrid.centered(8192,100e-12);a=gaussian_temporal_envelope(t,duration)
    expected=effective_pump_absorption_cross_section_295K(a,t,1.9077e-6)
    spec=PumpSource(1.9077e-6,duration)
    assert np.isclose(spec.effective_absorption_m2(),expected,rtol=2e-8,atol=0)
    assert spec.summary()['duration_fwhm_s']==duration


def test_chirped_duration_does_not_narrow_the_spectrum():
    bw=GAUSSIAN_TBP/1e-12;gdd=.5e-24
    duration=1e-12*np.sqrt(1+(4*np.log(2)*gdd/1e-24)**2)
    chirped=PumpSource(duration_fwhm_s=duration,bandwidth_fwhm_hz=bw,gdd_s2=gdd)
    tl=PumpSource(duration_fwhm_s=1e-12)
    assert np.isclose(chirped.effective_absorption_m2(),tl.effective_absorption_m2(),rtol=1e-13)
    assert chirped.effective_absorption_m2()<PumpSource(duration_fwhm_s=duration).effective_absorption_m2()
    with pytest.raises(ValueError):PumpSource(duration_fwhm_s=duration,gdd_s2=gdd)
    with pytest.raises(ValueError):PumpSource(duration_fwhm_s=10e-12,bandwidth_fwhm_hz=bw)


def test_custom_source_and_documented_override():
    freq=(C0/1.9077e-6,C0/1.9283e-6);weights=(.7,.3)
    source=PumpSource(spectrum_frequency_hz=freq,spectrum_energy_weights=weights)
    assert np.isclose(source.effective_absorption_m2(),np.array(weights)@pump_absorption_cross_section_295K(C0/np.array(freq)),rtol=1e-12,atol=0)
    with pytest.raises(ValueError):PumpSource(effective_absorption_override_m2=1e-24)
    explicit=PumpSource(effective_absorption_override_m2=1.2e-24,override_note='Measured effective cross section example')
    assert explicit.effective_absorption_m2()==1.2e-24


@pytest.mark.parametrize('factory',['thermal','radial'])
def test_modal_and_thermal_sources_bind_pulse_duration(factory):
    c=ThinDiskResonator();mesh=DiskThermalMesh.disk(6,2);models=[]
    for duration in [1e-12,10e-12]:
        if factory=='thermal':
            m=modal_laser_on_thermal_mesh(mesh,c,pump_duration_fwhm_s=duration)
        else:
            m,_=radial_laser(c,radial_points=36,z_slices=2,pump_duration_fwhm_s=duration)
        models.append(m)
        assert np.isclose(m.sa,PumpSource(duration_fwhm_s=duration).effective_absorption_m2(),rtol=1e-12,atol=0)
        assert m.pump_source.duration_fwhm_s==duration
    assert models[0].sa < .7*models[1].sa
    with pytest.raises(ValueError,match='durations differ'):
        models[0].run(1e-4,pump_fwhm_s=10e-12,max_cycles=1,min_cycles=1)


def test_stage7_source_binding_before_expensive_solve(monkeypatch):
    # Inspect the actual material constructor then deliberately stop. This tests
    # parameter plumbing, not an invented oscillator/thermal result.
    import hoyag.coupled_resonator as cr
    class Reached(Exception):pass
    captures=[]
    def fake_model(*args,**kwargs):
        captures.append(kwargs['pump_source']);raise Reached()
    grid=Grid2D.square(128,.012);mesh=DiskThermalMesh.disk(16,2,4)
    cavity=ThinDiskResonator(air_gap_m=.25)
    cfg=json.loads((Path(__file__).resolve().parents[1]/'config/stage6_assembly.json').read_text())
    cfg['numerics']['mechanical'].update(nr=2,ntheta=8,outer_rings=1,nz_disk=2,nz_plate=2)
    cfg['numerics']['plate_thermal_nz']=2
    monkeypatch.setattr(cr,'FieldCoupledLaser',fake_model)
    for duration in [1e-12,10e-12]:
        with pytest.raises(Reached):
            run_coupled_hot_cavity(grid,mesh,cavity,1e-3,cfg,pump_duration_s=duration)
    assert captures[0].effective_absorption_m2()<.7*captures[1].effective_absorption_m2()
    assert 'pump_absorption_m2=1.223454786e-24' not in inspect.getsource(run_coupled_hot_cavity)


@pytest.mark.parametrize('updates',[
    {'N_total_m3':np.nan},{'N_total_m3':np.inf},{'tau7_s':-1.},{'tau5_s':0.},
    {'sigma_abs_laser_m2':-1e-24},{'sigma_em_pump_m2':np.nan},{'M56_s1':-1.},
    {'k75_m3_s':np.inf},{'C57_m3_s':-1e-23},{'beta78':.2},
    {'beta56':-.1},{'beta57':1.3},{'beta68':np.nan},{'beta68':.8},
    {'pump_wavelength_m':0.},{'laser_wavelength_m':np.inf}])
def test_unphysical_material_parameters_rejected(updates):
    with pytest.raises(ValueError):HoYAGFourLevelParams(**updates)


def test_zero_rates_and_finite_isolated_lifetimes_still_supported():
    isolated_params();HoYAGFourLevelParams(sigma_abs_pump_m2=0.,sigma_em_pump_m2=0.)
    with pytest.raises(ValueError):stimulated_rates_from_intensity(np.nan,1e-6,1e-24,1e-24)
    with pytest.raises(ValueError):stimulated_rates_from_intensity(1.,1e-6,-1e-24,1e-24)


def test_backward_evanescent_mask_does_not_evaluate_overflowing_bins():
    g=Grid2D.square(32,1e-6);a=gaussian_beam(g,1e-7)
    with np.errstate(over='raise',invalid='raise'):
        out=angular_spectrum_propagate(a,g,2e-6,-.001,bandlimit=True)
    assert np.all(np.isfinite(out))
    with pytest.raises(ValueError,match='evanescent'):
        angular_spectrum_propagate(a,g,2e-6,-.001,bandlimit=False)


def test_spatiotemporal_path_uses_same_safe_masked_transfer():
    g=Grid2D.square(16,1e-6);t=TimeGrid.centered(8,40e-12)
    a=combine_spatial_temporal(gaussian_beam(g,1e-7),gaussian_temporal_envelope(t,10e-12),g,t)
    with np.errstate(over='raise',invalid='raise'):
        out=propagate_spatiotemporal(a,g,t,2e-6,-.001,bandlimit=True)
    assert np.all(np.isfinite(out))
    assert np.allclose(out[3],angular_spectrum_propagate(a[3],g,2e-6,-.001),rtol=1e-12)


def test_forward_evanescent_continuation_only_decays():
    g=Grid2D.square(32,1e-6);h=angular_spectrum_transfer(g,2e-6,.001,bandlimit=False)
    assert np.all(np.isfinite(h)) and np.all(abs(h)<=1+1e-15)
    with pytest.raises(ValueError):angular_spectrum_transfer(g,2e-6,np.nan)


@pytest.mark.parametrize('n',[0,1,2.5,True])
def test_invalid_spatial_grids_rejected_before_division(n):
    with pytest.raises(ValueError):Grid2D.square(n,.004)


def test_legacy_grid_centering_and_zero_distance_preserved():
    for n in [128,129]:
        g=Grid2D.square(n,.004);a=gaussian_beam(g,.0004)
        assert np.allclose(g.x,-g.x[::-1])
        assert np.array_equal(angular_spectrum_propagate(a,g,2e-6,0),a)


@pytest.mark.parametrize('rms,floor',[(.08,.5),(1.2,.6),(2.,.1),(.1,.99)])
def test_density_mean_rms_and_floor_all_hold(rms,floor):
    g=Grid2D.square(12,.004)
    a=smooth_random_ho_density_field(g,5,.001,1.52e26,rms,seed=12,minimum_fraction=floor)
    b=smooth_random_ho_density_field(g,5,.001,1.52e26,rms,seed=12,minimum_fraction=floor)
    stats=ho_density_statistics(a)
    assert np.array_equal(a.values_m3,b.values_m3)
    assert np.isclose(stats['mean_density_m3']/1.52e26,1,rtol=2e-12,atol=0)
    assert np.isclose(stats['relative_rms'],rms,rtol=2e-10,atol=0)
    assert a.values_m3.min()>=floor*1.52e26


def test_density_infeasible_or_degenerate_statistics_explicit():
    s=np.random.default_rng(1).normal(size=(2,3,4))
    with pytest.raises(ValueError):bounded_mean_rms(s,1.,.1,1.)
    with pytest.raises(ValueError):bounded_mean_rms(s,1.,.1,1.1)
    with pytest.raises(ValueError):bounded_mean_rms(s,1.,100.,.1)
    assert np.all(bounded_mean_rms(s,0.,0.,0.)==0)
    assert np.all(bounded_mean_rms(s,2.,0.,1.)==2)
    with pytest.raises(ValueError):bounded_mean_rms(s,0.,.1,0.)


def test_resolved_linear_absorption_and_effective_sigma_validity_limits():
    lam=1.9077e-6;time=TimeGrid.centered(8192,100e-12)
    a=gaussian_temporal_envelope(time,1e-12);column=1.52e26*.018
    diag=spectral_attenuation_diagnostic(a,time,lam,column)
    b=apply_frozen_spectral_absorption(a,time,lam,column)
    T=np.sum(abs(b)**2)/np.sum(abs(a)**2)
    assert np.isclose(T,diag['spectral_transmission'],rtol=1e-12)
    assert diag['relative_transmission_bias']<-.4
    with pytest.raises(ValueError):check_spectral_scalar_limit(diag,policy='error')
    with pytest.warns(ModelValidityWarning):assert not check_spectral_scalar_limit(diag)
    safe=PumpSource(duration_fwhm_s=10e-12).attenuation_diagnostic(1.52e26*.002)
    assert check_spectral_scalar_limit(safe,policy='error')


def test_cavity_sampling_is_not_confused_with_eigenpair_convergence():
    c=ThinDiskResonator(air_gap_m=.25)
    coarse=cavity_sampling_diagnostic(Grid2D.square(128,.012),c)
    refined=cavity_sampling_diagnostic(Grid2D.square(256,.012),c)
    assert not coarse['nyquist_satisfied_over_radius']
    assert refined['nyquist_satisfied_over_radius']
    assert not refined['mesh_convergence_verified']
