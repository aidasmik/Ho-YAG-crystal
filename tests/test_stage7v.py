"""Stage 7V diagnostic correctness, not merely self-comparison of simulations."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from pathlib import Path
import json
import numpy as np
import pytest
from hoyag.validation_metrics import (fingerprint,sample_cartesian,vector_overlap,compare_fields,
    piston_removed_opd_error,oam_spectrum,ContinuousDopant,conservative_heat_remap,edge_power_fraction)
from hoyag.validation_campaign import (make_plan,physical_spec,compile_report,verify_saved,
    write_json,file_sha256,case_identity,probe_frozen_disk,compare_cases,THRESHOLDS)
from hoyag.thermal import DiskThermalMesh
from hoyag.propagation import Grid2D
from hoyag.resonator import ThinDiskResonator,lg0_field
from hoyag.stress_optics import HotDiskScreens

ROOT=Path(__file__).resolve().parents[1]


def vector(n=128,width=.006,ell=1,waist=.0005):
    g=Grid2D.square(n,width);x,y=g.mesh
    f=np.stack([lg0_field(x,y,waist,ell),np.zeros(g.shape,complex)])
    return g,f


def test_stable_fingerprint_and_nonfinite_rejection():
    assert fingerprint({'x':1,'y':[2]})==fingerprint({'y':[2],'x':1})
    with pytest.raises(ValueError):fingerprint({'x':float('nan')})


def test_complex_interpolation_does_not_interpolate_wrapped_phase():
    g,f=vector();x,y=g.mesh
    out=sample_cartesian(f,g.x,g.y,x,y)
    assert np.max(abs(out-f))<1e-12
    assert np.iscomplexobj(out)


@pytest.mark.parametrize('ell',[-3,-1,0,1,3])
def test_analytic_lg_azimuthal_spectrum(ell):
    g,f=vector(n=192,ell=ell)
    s=oam_spectrum(f,g.x,g.y,radius_m=.002,radial_points=140,angular_points=192)
    assert s['charge_fractions'][str(ell)]>.9999
    assert abs(s['mean_azimuthal_order']-ell)<2e-4
    assert abs(s['pupil_power_over_cartesian_power']-1)<.002


def test_oam_mixture_powers_not_contour_winding():
    g,a=vector(n=192,ell=1);_,b=vector(n=192,ell=-2)
    a/=np.linalg.norm(a);b/=np.linalg.norm(b)
    s=oam_spectrum(np.sqrt(.7)*a+np.sqrt(.3)*b,g.x,g.y,radius_m=.002)
    assert abs(s['charge_fractions']['1']-.7)<1e-3
    assert abs(s['charge_fractions']['-2']-.3)<1e-3
    assert abs(s['mean_azimuthal_order']-.1)<.003


def test_oam_unreported_orders_not_renormalized_away():
    g,f=vector(n=192,ell=4)
    s=oam_spectrum(f,g.x,g.y,radius_m=.002,charges=(-1,0,1))
    assert s['reported_fraction']<1e-4
    assert s['unreported_fraction']>.9999


def test_vector_polarization_not_discarded():
    _,a=vector();b=a[::-1]
    assert vector_overlap(a,b)==0.
    assert np.isclose(vector_overlap(a,a*np.exp(2j)),1.)


def test_cross_resolution_uses_metres_and_preserves_vortex():
    ga,a=vector(n=128);gb,b=vector(n=256)
    result=compare_fields(a[None],ga.x,ga.y,b[None],gb.x,gb.y)
    assert result['minimum_matched_overlap']>.99999
    ar=np.stack([a,a[::-1]]);br=np.stack([(b+b[::-1])/np.sqrt(2),(b-b[::-1])/np.sqrt(2)])
    rotated=compare_fields(ar,ga.x,ga.y,br,gb.x,gb.y)
    assert rotated['minimum_subspace_capture']>.99999
    assert .49<rotated['minimum_matched_overlap']<.51


def test_differing_window_does_not_crop_away_the_field():
    ga,a=vector(n=128,width=.006);gb,b=vector(n=256,width=.012)
    c=compare_fields(a[None],ga.x,ga.y,b[None],gb.x,gb.y)
    assert c['minimum_matched_overlap']>.99999


def test_beam_displacement_is_not_aligned_away():
    g,a=vector();x,y=g.mesh
    b=np.stack([lg0_field(x-.0001,y,.0005,1),np.zeros(g.shape)])
    c=compare_fields(a[None],g.x,g.y,b[None],g.x,g.y)
    assert c['minimum_matched_overlap']<.95


def test_opd_piston_removed_but_tilt_and_defocus_retained():
    g,_=vector();x,y=g.mesh;zero=np.zeros(g.shape)
    piston=piston_removed_opd_error(zero,g.x,g.y,zero+1e-6,g.x,g.y)
    tilt=piston_removed_opd_error(zero,g.x,g.y,1e-4*x,g.x,g.y)
    focus=piston_removed_opd_error(zero,g.x,g.y,.1*(x*x+y*y),g.x,g.y)
    assert piston['rms_difference_m']<1e-20
    assert tilt['rms_difference_m']>1e-8 and focus['rms_difference_m']>1e-9


def test_dopant_is_same_continuous_specimen_on_every_grid():
    d=ContinuousDopant(amplitude_bound=.2,seed=4)
    x=np.array([0.,.001]);v=d.value(x,0.,.0005)
    assert np.array_equal(v,d.value(x,0.,.0005))
    assert np.all((v>=.8*d.mean_m3)&(v<=1.2*d.mean_m3))
    a=DiskThermalMesh.disk(12,4,8);b=DiskThermalMesh.disk(24,8,16)
    av=d.cell_averages(a,order=4);bv=d.cell_averages(b,order=4)
    mean_a=np.sum(av*a.volumes_m3)/a.volumes_m3.sum()
    mean_b=np.sum(bv*b.volumes_m3)/b.volumes_m3.sum()
    assert abs(mean_a/mean_b-1)<1e-5
    assert fingerprint(asdict(d))==fingerprint(asdict(ContinuousDopant(amplitude_bound=.2,seed=4)))


def test_uniform_dopant_cell_averages_exact():
    m=DiskThermalMesh.disk(7,3,4);d=ContinuousDopant()
    assert np.allclose(d.cell_averages(m),d.mean_m3,rtol=1e-14)


@pytest.mark.parametrize('nphi',[1,4,7])
def test_heat_remap_conserves_nonuniform_source(nphi):
    a=DiskThermalMesh.disk(8,4,nphi);b=DiskThermalMesh.disk(13,7,12)
    q=np.random.default_rng(7).uniform(.1,2,a.shape)
    mapped=conservative_heat_remap(q,a,b)
    assert np.isclose(np.sum(q*a.volumes_m3),np.sum(mapped*b.volumes_m3),rtol=1e-12,atol=0)
    assert mapped.min()>=0


def test_constant_heat_remap_and_domain_guard():
    a=DiskThermalMesh.disk(8,4,4);b=DiskThermalMesh.disk(13,7,12)
    q=conservative_heat_remap(np.full(a.shape,3.),a,b)
    assert np.allclose(q,3.,rtol=1e-12)
    with pytest.raises(ValueError):conservative_heat_remap(np.ones(a.shape),a,DiskThermalMesh.disk(radius_m=.006))


def test_plan_has_separate_axes_and_scope_and_full_required_groups():
    p=make_plan('pilot');full=make_plan('full')
    assert len({c['id'] for c in full['cases']})==len(full['cases'])
    assert set(full['required_groups'])<=set(g['id'] for g in full['groups'])
    assert any(c['kind']=='frozen_assembly' for c in p['cases'])
    optical=[c for c in p['cases'] if c['id'] in ('base256','opt384','opt512')]
    assert [c['grid']['optical_n'] for c in optical]==[256,384,512]
    assert len({fingerprint(physical_spec(c,ROOT)) for c in optical})==1
    assert len({fingerprint(c['dopant']) for c in full['cases']})==1


def test_missing_runs_never_qualify(tmp_path):
    report=compile_report(make_plan('pilot'),tmp_path)
    assert not report['numerical_qualification_passed']
    assert not report['approved_for_trusted_nn_targets']
    assert 'joint' in report['unqualified_required_groups']
    assert all(c['status']=='NOT_RUN' for c in report['cases'])


def write_minimal_case(folder,case,status='COMPLETED'):
    folder.mkdir(parents=True)
    g,f=vector(n=32);z=np.zeros(g.shape)
    np.savez(folder/'state.npz',x_m=g.x,y_m=g.y,fields=f[None],mean_opd_m=z)
    s={'case':case,'status':status,'identity':{'physics_fingerprint':'same','solver_source_sha256':'code','source_state_sha256':'source'},
       'scalars':{'output_W':1.,'heat_W':.6,'absorbed_W':2.,'disk_peak_K':302.,'plate_peak_K':294.,
                  'thermal_balance_relative':0.,'mechanical_residual':0.,'energy_ledger_relative':0.},
       'state_sha256':file_sha256(folder/'state.npz'),'elapsed_s':1.,'edge_power_fraction':0.,
       'mirror_sampling':{'nyquist_satisfied_over_radius':True}}
    write_json(folder/'summary.json',s)
    return s


def test_corrupt_or_wrong_fingerprint_cache_rejected(tmp_path):
    c=make_plan()['cases'][0];s=write_minimal_case(tmp_path/'x',c)
    assert verify_saved(tmp_path/'x',s['identity']) is not None
    assert verify_saved(tmp_path/'x',{'different':'identity'}) is None
    with (tmp_path/'x/state.npz').open('ab') as stream:stream.write(b'corrupt')
    assert verify_saved(tmp_path/'x') is None


def test_nonconvergence_and_mixed_physics_cannot_pass(tmp_path):
    c=make_plan()['cases'][0]
    a=write_minimal_case(tmp_path/'a',c);b=write_minimal_case(tmp_path/'b',c,'NONCONVERGED')
    assert compare_cases(tmp_path/'a',tmp_path/'b')['status']=='FAIL'
    b['status']='COMPLETED';b['identity']['physics_fingerprint']='different'
    write_json(tmp_path/'b/summary.json',b)
    assert compare_cases(tmp_path/'a',tmp_path/'b')['status']=='FAIL'
    assert compare_cases(tmp_path/'a',tmp_path/'b',kind='sensitivity')['status']=='MEASURED'


def test_coupled_and_frozen_source_cannot_be_claimed_same_validation(tmp_path):
    c=make_plan()['cases'][0];a=write_minimal_case(tmp_path/'a',c)
    b=write_minimal_case(tmp_path/'b',{**c,'kind':'frozen_assembly'})
    assert compare_cases(tmp_path/'a',tmp_path/'b')['status']=='FAIL'


def test_uniform_reflected_probe_gain_and_oam():
    g=Grid2D.square(128,.012);c=ThinDiskResonator(air_gap_m=.25)
    identity=np.broadcast_to(np.eye(2,dtype=complex),(*g.shape,2,2)).copy();zero=np.zeros(g.shape)
    s=HotDiskScreens(identity,identity,zero,zero,zero,zero,zero,zero,c.wavelength_m)
    probes,_=probe_frozen_disk(g,c,s,np.full(g.shape,.04),charges=(-1,1),waist_m=.0005)
    for p in probes:
        assert np.isclose(p['double_pass_power_gain'],c.disk_hr_reflectivity*np.exp(.08),rtol=1e-12)
        assert p['vector_overlap_with_input']>.999999
        assert p['output_azimuthal_purity']>.999
    assert abs(probes[0]['double_pass_power_gain']-probes[1]['double_pass_power_gain'])<1e-14


def test_finite_aperture_and_boundary_energy_reported():
    _,f=vector();assert edge_power_fraction(f)<1e-8
    assert edge_power_fraction(np.ones_like(f))>.2
