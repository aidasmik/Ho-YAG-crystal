"""Stage 7V metrics, campaign integrity and classification tests.

Synthetic arrays below test bookkeeping/analytical limits; they are never
reported as solved Ho:YAG operating points.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import pytest

from hoyag.thermal import DiskThermalMesh
from hoyag.validation_metrics import (DiagnosticGrid, grid_from_axes, ContinuousDensity,
    interval_average_matrix, conservative_remap, ComparisonDomain, compare_fields,
    compare_opd, vector_overlap, modal_mixture_fidelity, sample_cartesian,
    oam_spectrum, closed_phase_winding, stable_hash, relative_change)
from hoyag.validation_backend import lg_seed, sha256_file, solve_assembly, weak_double_pass_probe
from hoyag.validation_plan import make_plan, validate_plan, Acceptance
from hoyag.validation_campaign import (execute_case, load_record, quality_gates, write_json,
    compare_records, build_report, json_safe)

ROOT=Path(__file__).resolve().parents[1]


def physical():
    assembly=json.loads((ROOT/'config/stage6_assembly.json').read_text())
    assembly.pop('numerics',None)
    return {'assembly':assembly,
       'cavity':{'disk_diameter_m':.01,'disk_thickness_m':.001,'disk_hr_reflectivity':.9995,
                 'wavelength_m':2.0903e-6,'output_mirror_radius_m':.5},
       'pump':{'energy_J':.001,'duration_s':1e-11,'waist_m':.0005,'repetition_rate_Hz':10000},
       'density':asdict(ContinuousDensity()),
       'probe':{'waist_m':.000408,'charges':[0,1,-1,2],
                'model':'weak seeded double-pass probe of frozen mean populations'}}


def plan(kind='frozen'):
    return make_plan(physical(),kind=kind,reference={'state_sha256':'a'*64,'summary_sha256':'b'*64})


@pytest.mark.parametrize('n',[32,33,256])
def test_grid_preserves_repository_half_pixel_convention(n):
    g=DiagnosticGrid.square(n,.012)
    assert np.allclose(g.x,-g.x[::-1],rtol=0,atol=1e-16)
    restored=grid_from_axes(g.x,g.y)
    assert np.allclose(restored.mesh,g.mesh)
    assert np.isclose(g.dx*n,.012)


@pytest.mark.parametrize('bad',[True,1,3.2,-2])
def test_grid_rejects_invalid_n(bad):
    with pytest.raises(ValueError):DiagnosticGrid.square(bad,.012)


def test_continuous_map_same_physical_points_independent_of_mesh():
    d=ContinuousDensity(contrast_bound=.08)
    point=(.00031,-.00047,.00066)
    a=d(*point);b=ContinuousDensity(**asdict(d))(*point)
    assert a==b and d.fingerprint==ContinuousDensity(**asdict(d)).fingerprint
    q1=d.cell_average(DiskThermalMesh.disk(12,4,8))
    q2=d.cell_average(DiskThermalMesh.disk(24,8,24))
    assert q1.shape!=(q2.shape)
    assert np.all(q1>0) and np.all(q2>0)
    assert d(*point)==a


def test_continuous_density_analytical_volume_mean_not_grid_renormalized():
    d=ContinuousDensity(contrast_bound=.2)
    m=DiskThermalMesh.disk(24,8,24)
    q=d.cell_average(m,order=6)
    mean=np.sum(q*m.volumes_m3)/m.volumes_m3.sum()
    assert abs(mean/d.mean_m3-1)<2e-9
    assert np.ptp(q)>0


def test_constant_density_exact_on_every_mesh():
    d=ContinuousDensity()
    for mesh in [DiskThermalMesh.disk(2,1,1),DiskThermalMesh.disk(7,3,4)]:
        q=d.cell_average(mesh)
        assert np.allclose(q,d.mean_m3,rtol=2e-15)


def test_different_random_seed_is_different_physical_problem():
    a=ContinuousDensity(contrast_bound=.08,seed=1)
    b=ContinuousDensity(contrast_bound=.08,seed=2)
    assert a.fingerprint!=b.fingerprint
    assert a(.001,.002,.0003)!=b(.001,.002,.0003)


def test_nonnegative_exact_overlap_remapping_conserves_heat_and_means():
    old=DiskThermalMesh.disk(5,3,4);new=DiskThermalMesh.disk(13,7,9)
    rng=np.random.default_rng(1);q=rng.random((2,*old.shape))*1e6
    r=conservative_remap(q,old,new)
    assert r.shape==(2,*new.shape) and r.min()>=0
    assert np.allclose(np.sum(q*old.volumes_m3,axis=(1,2,3)),
                       np.sum(r*new.volumes_m3,axis=(1,2,3)),rtol=1e-13)
    assert np.allclose(conservative_remap(np.ones(old.shape),old,new),1,rtol=1e-14)


def test_remapping_forbids_a_different_crystal_volume():
    with pytest.raises(ValueError):interval_average_matrix([0,1],[0,2])
    with pytest.raises(ValueError):interval_average_matrix([0,1,1],[0,1])


def test_axisymmetric_source_expands_without_an_angular_pattern():
    a=DiskThermalMesh.disk(5,3,1);b=DiskThermalMesh.disk(9,6,12)
    r=conservative_remap(np.arange(15).reshape(a.shape),a,b)
    assert np.allclose(r,r[:,:,:1])


def test_phase_preserving_sampling_handles_modes_and_polarizations():
    g=DiagnosticGrid.square(128,.006);x,y=g.mesh
    f=np.stack([x+1j*y,2*x-3j*y])
    r=sample_cartesian(f,g,np.array([.00013,-.00047]),np.array([.00029,.00033]))
    assert r.shape==(2,2)
    assert np.allclose(r[0],[.00013+1j*.00029,-.00047+1j*.00033],rtol=1e-12,atol=1e-18)
    modes=np.stack([f,2j*f])
    assert sample_cartesian(modes,g,np.zeros((3,4)),np.zeros((3,4))).shape==(2,2,3,4)


def test_opd_comparison_removes_piston_not_tilt_or_defocus():
    g=DiagnosticGrid.square(256,.004);x,y=g.mesh;domain=ComparisonDomain()
    a=np.zeros(g.shape);b=a+87e-9
    r=compare_opd(a,g,b,g,domain)
    assert r['piston_removed_difference_rms_m']<1e-22
    tilted=b+4e-5*x
    r=compare_opd(a,g,tilted,g,domain)
    assert r['piston_removed_difference_rms_m']>2e-9
    assert r['removed_terms']==['piston'] and not r['tilt_or_defocus_removed']


def test_native_grid_overlap_global_phase_and_amplitude_invariant():
    g=DiagnosticGrid.square(128,.004);a=lg_seed(g,.0004,1)
    assert np.isclose(vector_overlap(a,3*a*np.exp(.78j)),1,atol=1e-14)
    assert vector_overlap(a,lg_seed(g,.0004,-1))<1e-12


def test_cross_grid_comparison_has_same_physical_coordinates():
    ga=DiagnosticGrid.square(128,.004);gb=DiagnosticGrid.square(256,.004)
    a=lg_seed(ga,.0004,1);b=lg_seed(gb,.0004,1)*np.exp(1.2j)
    result=compare_fields(a,ga,b,gb,ComparisonDomain(.0015,.000408,96,256))
    assert result['single_mode_overlap']>.9999
    assert min(result['roi_power_fraction_a'])>.99999


def test_mode_mixture_comparison_does_not_add_incoherent_fields():
    g=DiagnosticGrid.square(64,.004)
    f=lg_seed(g,.0004,0);h=lg_seed(g,.0004,0,1)
    a=np.stack([f,h]);b=np.stack([(f+h)/np.sqrt(2),(f-h)/np.sqrt(2)])
    w=np.ones(g.shape)*g.dx*g.dy
    assert modal_mixture_fidelity(a,b,[.5,.5],[.5,.5],w)>1-1e-12
    assert modal_mixture_fidelity(a,a,[.9,.1],[.1,.9],w)<.5
    assert modal_mixture_fidelity(a,a[::-1],[.9,.1],[.1,.9],w)>1-1e-12


def test_unequal_retained_mode_counts_are_compared_as_mixtures():
    g=DiagnosticGrid.square(64,.004);f=lg_seed(g,.0004,0);h=lg_seed(g,.0004,1)
    fidelity=modal_mixture_fidelity(f[None],np.stack([f,h]),[1.],[.9,.1],np.ones(g.shape))
    assert np.isclose(fidelity,.9,rtol=1e-12)


@pytest.mark.parametrize('charge',[-3,-1,0,1,3])
def test_oam_sign_and_closed_contour_winding(charge):
    g=DiagnosticGrid.square(256,.004);f=lg_seed(g,.0004,charge)
    s=oam_spectrum(f,g)
    assert s['dominant_charge']==charge
    assert s['fractions'][s['charges'].index(charge)]>.9999
    r=closed_phase_winding(f[0],g,.0004)
    assert r['resolved'] and abs(r['charge']-charge)<1e-10


def test_unreported_oam_tail_is_not_renormalized_away():
    g=DiagnosticGrid.square(256,.004)
    s=oam_spectrum(lg_seed(g,.00025,8),g,max_charge=3)
    assert s['unreported_charge_tail']>.999
    assert sum(s['fractions'])<.001


def test_zero_amplitude_winding_is_unresolved():
    g=DiagnosticGrid.square(32,.004)
    r=closed_phase_winding(np.zeros(g.shape),g,.001)
    assert not r['resolved'] and r['charge'] is None


def test_plan_contains_independent_refinement_and_physical_sensitivity():
    p=validate_plan(plan('coupled'))
    names={g['name'] for g in p['groups']}
    assert {'optical_resolution','window_padding','material_resolution','mechanical_resolution',
            'joint_refinement','integration_tolerances','retained_modes','candidate_spectrum',
            'initialization','cooling_and_density_sensitivity'}<=names
    cases={c['id']:c for c in p['cases']}
    base=cases['reference']
    for key in ['optical_384','material_1','mechanical_1','joint_1','modes_2','start_lg_plus']:
        assert cases[key]['physics_hash']==base['physics_hash']
    assert cases['sensitivity_contact_0.5']['physics_hash']!=base['physics_hash']
    a,b=cases['reference']['numerics'],cases['window_384']['numerics']
    assert np.isclose(a['optical_window_m']/a['optical_n'],b['optical_window_m']/b['optical_n'])


def test_plan_rejects_silent_configuration_edits():
    p=plan();p['cases'][0]['numerics']['optical_n']=1024
    with pytest.raises(ValueError,match='configuration changed'):validate_plan(p)


def fake_record(case,directory,manifest,temperature=300.,phase=0.):
    # Deterministic synthetic arrays solely for acceptance-logic testing.
    g=DiagnosticGrid.square(64,.004);x,y=g.mesh
    f=np.stack([lg_seed(g,.0004,l) for l in [0,1,-1,2]])
    d=Path(directory)/case['id'];d.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(d/'state.npz',fields_used=f,x_m=g.x,y_m=g.y,mean_roundtrip_opd_m=phase*(x*x+y*y))
    rec={'id':case['id'],'spec_hash':case['spec_hash'],'physics_hash':case['physics_hash'],
         'source_hash':manifest['source_hash'],'kind':'frozen','status':'completed',
         'field_semantics':'independent_weak_probes','probe_labels':[0,1,-1,2],
         'metrics':{'heat_W':.67,'peak_disk_K':temperature,'peak_plate_K':293.7,
                    'thermal_energy_error_relative':1e-12,'mechanical_residual':1e-12,
                    'heat_remap_relative_error':1e-15},
         'probes':[{'power_gain':1.1,'target_oam_fraction':1.,'crossed_polarization_fraction':0.}]*4,
         'state_sha256':sha256_file(d/'state.npz')}
    write_json(d/'summary.json',rec)
    return rec


def test_missing_cases_are_incomplete_not_passed(tmp_path):
    p=plan();m={'source_hash':'abc'}
    fake_record(p['cases'][0],tmp_path,m)
    report=build_report(p,tmp_path,m)
    assert report['status']=='incomplete'
    assert not report['coupled_numerically_qualified'] and not report['dataset_ready']
    assert set(report['case_status'].values())=={'completed','not_run'}


def test_full_frozen_campaign_can_never_certify_coupled_model(tmp_path):
    p=plan();m={'source_hash':'abc'}
    for case in p['cases']:fake_record(case,tmp_path,m)
    report=build_report(p,tmp_path,m)
    assert report['component_numerical_checks_passed']
    assert report['status']=='component_checks_passed_not_coupled_validation'
    assert not report['coupled_numerically_qualified'] and not report['dataset_ready']


def test_acceptance_catches_temperature_and_opd_not_just_power(tmp_path):
    p=plan();m={'source_hash':'abc'}
    a=fake_record(p['cases'][0],tmp_path,m)
    b=fake_record(p['cases'][1],tmp_path,m,temperature=300.2,phase=.08)
    r=compare_records(a,b,tmp_path,Acceptance(),ComparisonDomain())
    assert r['status']=='fail' and not r['gates']['peak_disk_K'] and not r['gates']['piston_removed_opd']


def test_sensitivity_is_not_a_numerical_comparison(tmp_path):
    p=plan();m={'source_hash':'abc'}
    a=fake_record(p['cases'][0],tmp_path,m)
    sensitive=next(c for c in p['cases'] if c['purpose']=='sensitivity')
    b=fake_record(sensitive,tmp_path,m)
    r=compare_records(a,b,tmp_path,Acceptance(),ComparisonDomain())
    assert r['status']=='invalid_comparison'


def test_wrong_source_and_corrupted_state_never_resume(tmp_path):
    p=plan();c=p['cases'][0];m={'source_hash':'abc'}
    fake_record(c,tmp_path,m)
    assert load_record(c,tmp_path,{'source_hash':'different'})['status']=='stale'
    (tmp_path/c['id']/'state.npz').write_bytes(b'broken')
    assert load_record(c,tmp_path,m)['status']=='corrupt'


def test_case_failure_saved_and_not_reported_as_complete(tmp_path,monkeypatch):
    import hoyag.validation_campaign as vc
    def fail(*a,**k):raise RuntimeError('intentional unit-test failure')
    monkeypatch.setattr(vc,'frozen_case',fail)
    c=plan()['cases'][0]
    record=execute_case(c,tmp_path,{'source_hash':'abc'},reference_directory=tmp_path)
    assert record['status']=='failed' and 'intentional unit-test failure' in record['error']
    assert not quality_gates(record,Acceptance())['execution_completed']


def test_eigensolver_residual_alone_does_not_make_good_case():
    r={'status':'completed','kind':'coupled','metrics':{'thermal_energy_error_relative':0.,
      'mechanical_residual':0.,'coupled_converged':True,'optical_population_error_relative':0.,
      'eigen_residual':1e-14},'sampling':{'mirror_nyquist_over_disk_radius':False}}
    assert not all(quality_gates(r,Acceptance()).values())


def test_nonfinite_report_numbers_rejected():
    with pytest.raises(ValueError):json_safe({'metric':np.nan})
    with pytest.raises(ValueError):relative_change(np.inf,2.)


def test_weak_double_pass_uses_two_gain_traversals_and_hr_once():
    from hoyag.stress_optics import HotDiskScreens
    g=DiagnosticGrid.square(32,.004);zero=np.zeros(g.shape)
    eye=np.broadcast_to(np.eye(2,dtype=complex),(*g.shape,2,2)).copy()
    screens=HotDiskScreens(eye,eye,zero,zero,zero,zero,zero,zero,2e-6)
    f=lg_seed(g,.0004,1);log_gain=.1;hr=.999
    out=weak_double_pass_probe(f,g,screens,log_gain,hr,np.ones(g.shape))
    assert np.isclose(np.sum(abs(out)**2)/np.sum(abs(f)**2),hr*np.exp(2*log_gain),rtol=1e-13)
    assert vector_overlap(f,out)>1-1e-13


def test_small_real_finite_plate_assembly_runs_without_mock_physics():
    cfg=json.loads((ROOT/'config/stage6_assembly.json').read_text())
    cfg['numerics']['mechanical'].update(nr=2,outer_rings=1,ntheta=8,nz_disk=2,nz_plate=2)
    cfg['numerics']['plate_thermal_nz']=3
    g=DiagnosticGrid.square(32,.012);mesh=DiskThermalMesh.disk(6,3,4)
    q=np.ones(mesh.shape)*1e6;q[:,:,0]*=1.3
    t,u,s,fem=solve_assembly(mesh,g,q,cfg)
    assert t.relative_balance_error<1e-9
    assert u.free_residual_relative<1e-9
    assert t.disk_temperature_K.max()>t.plate_temperature_K.max()>293.15
    assert np.max(abs(s.geometry_roundtrip_opd_m))>0


def test_comparison_rejects_extrapolated_optical_domain():
    g=DiagnosticGrid.square(64,.001);f=lg_seed(g,.0001)
    with pytest.raises(ValueError,match='outside'):
        compare_fields(f,g,f,g,ComparisonDomain(radius_m=.001))
    with pytest.raises(ValueError,match='outside'):
        compare_opd(np.zeros(g.shape),g,np.zeros(g.shape),g,ComparisonDomain(radius_m=.001))


def test_mixed_initial_guess_seed_is_actually_used():
    from types import SimpleNamespace
    from hoyag.validation_backend import initial_modes
    g=DiagnosticGrid.square(64,.004);c=SimpleNamespace(waist_m=.0004)
    n={'mode_count':2,'initial_guess':'mixed','initial_seed':7}
    a=initial_modes(g,c,n);b=initial_modes(g,c,n)
    n['initial_seed']=8;other=initial_modes(g,c,n)
    assert np.array_equal(a,b) and not np.allclose(a,other)


def test_source_cache_identity_includes_numerical_runtime(tmp_path,monkeypatch):
    from hoyag.validation_backend import source_manifest
    import scipy
    p=tmp_path/'src/hoyag';p.mkdir(parents=True);(p/'one.py').write_text('x=1\n')
    a=source_manifest(tmp_path)
    monkeypatch.setattr(scipy,'__version__','different')
    b=source_manifest(tmp_path)
    assert a['files']==b['files'] and a['source_hash']!=b['source_hash']


def test_expanded_plan_covers_eight_modes_and_lg2():
    from hoyag.validation_plan import make_plan, physics_from_repository
    from pathlib import Path
    plan=make_plan(physics_from_repository(Path(__file__).resolve().parents[1]))
    cases={c['id']:c for c in plan['cases']}
    assert cases['modes_8']['numerics']['mode_count']==8
    assert cases['modes_8']['numerics']['settings']['eigen_candidates']>=8
    assert cases['start_lg_two']['numerics']['initial_guess']=='lg_two'
    from hoyag.validation_backend import initial_modes
    from hoyag.resonator import ThinDiskResonator
    from hoyag.validation_metrics import DiagnosticGrid
    f=initial_modes(DiagnosticGrid.square(64,.012),ThinDiskResonator(),cases['start_lg_two']['numerics'])
    assert f.shape==(2,2,64,64)


def test_new_frozen_mechanical_level_keeps_two_successive_pairs():
    from hoyag.validation_plan import make_plan, physics_from_repository
    from pathlib import Path
    plan=make_plan(physics_from_repository(Path(__file__).resolve().parents[1]),kind='frozen',
                   reference={'state_sha256':'example','summary_sha256':'example'})
    groups={g['name']:g for g in plan['groups']}
    assert groups['mechanical_resolution']['case_ids']==['reference','mechanical_1','mechanical_2','mechanical_3']
    assert groups['mechanical_resolution']['required_successive_pairs']==2
    assert groups['joint_refinement']['case_ids'][-1]=='joint_3'
