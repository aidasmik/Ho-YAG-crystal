"""Stage 7V manifest, execution and qualification of numerical validation cases.

This module never marks a dataset physically calibrated. It distinguishes full
hot-cavity reruns from frozen-source assembly/probe diagnostics. Missing work is
INCOMPLETE, not a pass. Completed files carry code/configuration/content hashes.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time
import traceback
import numpy as np
from .validation_metrics import (ContinuousDopant, fingerprint, compare_fields,
    piston_removed_opd_error, oam_spectrum, edge_power_fraction, vector_overlap,
    conservative_heat_remap)

BASE_REVISION = '0ffead9eddd004bd08ba45aad870f960f8ee6183'
THRESHOLDS = dict(power_relative=.01, heat_relative=.01, absorbed_relative=.01,
                  temperature_K=.05, opd_rms_m=2e-9, field_overlap=.999,
                  edge_power_fraction=1e-6, energy_ledger_relative=1e-4,
                  thermal_balance_relative=1e-8, mechanical_residual=1e-7)
REQUIRED_GROUPS = ('optical','window','material','mechanical_coupled','joint',
                   'integration','mode_count','initialization')


def serializable(value):
    if isinstance(value, dict): return {str(k): serializable(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [serializable(v) for v in value]
    if isinstance(value, np.ndarray): return serializable(value.tolist())
    if isinstance(value, np.generic): return serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write_json(path, value):
    """Atomic status replacement makes interrupted cases distinguishable."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(serializable(value), indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1<<20), b''): h.update(chunk)
    return h.hexdigest()


def source_digest(root):
    root = Path(root); h = hashlib.sha256()
    paths = list((root/'src/hoyag').glob('*.py'))
    if not paths: raise ValueError('repository source is missing')
    for path in sorted(paths):
        h.update(str(path.relative_to(root)).encode()); h.update(path.read_bytes())
    return h.hexdigest()


def physical_spec(case, root):
    root = Path(root)
    cavity = json.loads((root/case['cavity_config']).read_text())['cavity']
    assembly = json.loads((root/case['assembly_config']).read_text())
    for section, changes in case.get('assembly_overrides', {}).items():
        assembly[section] = _merge(assembly[section], changes)
    # Numerical mesh and plot metadata must not define a different crystal.
    physics_assembly = {k: assembly[k] for k in ('geometry','thermal','mechanical','optics')}
    return {'cavity': cavity, 'assembly': physics_assembly,
            'pump': case['pump'], 'dopant': case['dopant']}


def _merge(base, changes):
    out = deepcopy(base)
    for key, value in changes.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else deepcopy(value)
    return out


def make_plan(profile='pilot'):
    """Same physical case, separate discretization axes; no synthetic outcomes."""
    if profile not in ('pilot','full'): raise ValueError('profile must be pilot or full')
    base = {'id':'base256','kind':'coupled','cavity_config':'config/thin_disk_resonator_250mm_2pct.json',
            'assembly_config':'config/stage6_assembly.json',
            'pump':{'energy_J':.001,'repetition_rate_Hz':1e4,'duration_s':1e-11,'waist_m':.0005},
            'dopant':asdict(ContinuousDopant()),
            'grid':{'optical_n':256,'window_m':.012,'nr':16,'nz':2,'nphi':4},
            'mechanical':{'nr':4,'outer_rings':2,'ntheta':16,'nz_disk':2,'nz_plate':2,'radial_exponent':1.6},
            'plate_thermal_nz':6,'mode_count':1,'initial_charges':[0],
            'settings':{'max_outer_iterations':14,'field_tolerance':.001,
                'heat_relative_tolerance':.002,'temperature_tolerance_K':.02,
                'displacement_tolerance_m':2e-10,'power_relative_tolerance':.002,
                'optical_max_cycles':440,'optical_min_cycles':16,'eigen_candidates':2},
            'probes':[-3,-2,-1,0,1,2,3], 'probe_waist_m':.0004,
            'timeout_s':900}
    cases = [base]
    def add(name, **changes):
        c = _merge(base, changes); c['id'] = name; cases.append(c); return c
    add('opt384', grid={'optical_n':384})
    add('opt512', grid={'optical_n':512}, timeout_s=1200)
    add('modes2', mode_count=2, initial_charges=[0,0], settings={'eigen_candidates':4}, timeout_s=1200)
    # Assembly-only changes are explicitly fixed to the SAME archived source.
    for name, changes in [('mount_base',{}), ('contact_low',{'thermal':{'interface_conductance_W_m2K':1e4}}),
             ('contact_high',{'thermal':{'interface_conductance_W_m2K':1e6}}),
             ('shear_soft',{'mechanical':{'bond':{'tangential_stiffness_Pa_m':1e12}}}),
             ('roller',{'mechanical':{'plate_support':'roller'}}),
             ('coolant_low',{'thermal':{'coolant_conductance_W_m2K':1e3}})]:
        add(name, kind='frozen_assembly', source_case='base256', assembly_overrides=changes)
    add('fem_mid', kind='frozen_assembly', source_case='base256',
        mechanical={'nr':6,'outer_rings':3,'ntheta':24,'nz_disk':3,'nz_plate':3})
    add('fem_fine', kind='frozen_assembly', source_case='base256',
        mechanical={'nr':8,'outer_rings':4,'ntheta':32,'nz_disk':4,'nz_plate':4})
    groups = [dict(id='optical', cases=['base256','opt384','opt512'], kind='refinement',minimum_levels=3),
              dict(id='mode_count',cases=['base256','modes2','modes4'],kind='mode_comparison',minimum_levels=3),
              dict(id='frozen_mechanics',cases=['mount_base','fem_mid','fem_fine'],kind='refinement',minimum_levels=3),
              dict(id='mount_sensitivity',cases=['mount_base','contact_low','contact_high','shear_soft','roller','coolant_low'],
                   kind='sensitivity',minimum_levels=6)]
    if profile == 'full':
        add('window384',grid={'optical_n':384,'window_m':.018})
        add('window512',grid={'optical_n':512,'window_m':.024})
        add('material_mid',grid={'nr':24,'nz':4,'nphi':8},plate_thermal_nz=12,timeout_s=1800)
        add('material_fine',grid={'nr':32,'nz':8,'nphi':16},plate_thermal_nz=24,timeout_s=3600)
        add('mechanical_mid',mechanical={'nr':6,'outer_rings':3,'ntheta':24,'nz_disk':3,'nz_plate':3})
        add('mechanical_fine',mechanical={'nr':8,'outer_rings':4,'ntheta':32,'nz_disk':4,'nz_plate':4})
        add('joint_mid',grid={'optical_n':384,'nr':24,'nz':4,'nphi':8},plate_thermal_nz=12,
            mechanical={'nr':6,'outer_rings':3,'ntheta':24,'nz_disk':3,'nz_plate':3},timeout_s=2400)
        add('joint_fine',grid={'optical_n':512,'nr':32,'nz':8,'nphi':16},plate_thermal_nz=24,
            mechanical={'nr':8,'outer_rings':4,'ntheta':32,'nz_disk':4,'nz_plate':4},timeout_s=5400)
        add('tolerances_mid',settings={'optical_rtol':1e-6,'field_tolerance':5e-4,'heat_relative_tolerance':1e-3,
            'power_relative_tolerance':1e-3,'temperature_tolerance_K':.01,'eigen_tolerance':1e-7,'max_outer_iterations':20})
        add('tolerances_fine',settings={'optical_rtol':5e-7,'field_tolerance':2e-4,'heat_relative_tolerance':5e-4,
            'power_relative_tolerance':5e-4,'temperature_tolerance_K':.005,'eigen_tolerance':5e-8,'max_outer_iterations':24})
        add('modes4', mode_count=4,initial_charges=[0,0,1,-1],settings={'eigen_candidates':8},timeout_s=2400)
        add('seed_lg_plus',initial_charges=[1],timeout_s=1200)
        add('seed_lg_minus',initial_charges=[-1],timeout_s=1200)
        add('seed_mixed',initial_charges=[0],initial_mix=.15,timeout_s=1200)
        for name, axis, levels in [
             ('window','refinement',['base256','window384','window512']),
             ('material','refinement',['base256','material_mid','material_fine']),
             ('mechanical_coupled','refinement',['base256','mechanical_mid','mechanical_fine']),
             ('joint','refinement',['base256','joint_mid','joint_fine']),
             ('integration','refinement',['base256','tolerances_mid','tolerances_fine']),
             ('initialization','initialization',['base256','seed_lg_plus','seed_lg_minus','seed_mixed'])]:
            groups.append(dict(id=name,kind=axis,cases=levels,minimum_levels=len(levels)))
    # Gate lists full requirements even when the pilot intentionally omits them.
    return {'schema_version':1,'profile':profile,'base_revision':BASE_REVISION,
            'cases':cases,'groups':groups,'required_groups':list(REQUIRED_GROUPS),
            'thresholds':THRESHOLDS,'physical_calibration_verified':False,
            'required_measurements':['mount_sensitivity','seeded_probes']}


def case_identity(case, root, *, source_state_hash=None):
    return {'case_fingerprint':fingerprint(case),'physics_fingerprint':fingerprint(physical_spec(case, root)),
            'solver_source_sha256':source_digest(root),'source_state_sha256':source_state_hash}


def verify_saved(folder, expected=None):
    """A stale or partial cache cannot be reused or qualify a configuration."""
    folder=Path(folder)
    if not (folder/'summary.json').is_file(): return None
    summary=json.loads((folder/'summary.json').read_text())
    if summary.get('status') not in ('COMPLETED','NONCONVERGED'): return None
    if expected is not None and summary.get('identity') != expected: return None
    state=folder/'state.npz'
    if not state.is_file() or summary.get('state_sha256')!=file_sha256(state): return None
    return summary


def probe_frozen_disk(grid, cavity, screens, gain_column, *, charges=(-1,0,1),waist_m=.0004):
    """Weak, seeded, reflective double-pass probe of a FROZEN pumped disk.

    No curved output coupler, free-running eigenmode selection or extra gain
    saturation is imposed. This is the existing collapsed-disk approximation;
    the seed does not update the pump cycle. Mean populations are not claimed to
    represent an independently pumped amplifier with no intracavity extraction.
    """
    from .resonator import lg0_field
    from .stress_optics import apply_jones
    x,y=grid.mesh;mask=x*x+y*y<=(cavity.disk_diameter_m/2)**2
    logg=np.asarray(gain_column,float)
    if logg.shape!=grid.shape or np.any(~np.isfinite(logg)) or abs(logg).max()>50:
        raise ValueError('finite single-pass logarithmic intensity gain required')
    phase=np.exp(2j*np.pi*screens.geometry_roundtrip_opd_m/cavity.wavelength_m)
    result=[];fields={}
    pupil=min(.002, .49*grid.nx*grid.dx, .49*grid.ny*grid.dy)
    for ell in charges:
        f=np.stack([lg0_field(x,y,waist_m,int(ell)),np.zeros(grid.shape,dtype=complex)])
        f=f/np.sqrt(np.sum(abs(f)**2)*grid.dx*grid.dy)
        j=apply_jones(f.transpose(1,2,0),screens.inward_jones)
        j=apply_jones(j,screens.outward_jones)
        out=(j*np.exp(logg)[...,None]*phase[...,None]*mask[...,None]
                 *np.sqrt(cavity.disk_hr_reflectivity)).transpose(2,0,1)
        oam=oam_spectrum(out,grid.x,grid.y,radius_m=pupil)
        oam_in=oam_spectrum(f,grid.x,grid.y,radius_m=pupil)
        power=float(np.sum(abs(out)**2)*grid.dx*grid.dy)
        result.append({'charge':int(ell),'double_pass_power_gain':power,
            'vector_overlap_with_input':vector_overlap(f,out),
            'input_azimuthal_purity':oam_in['charge_fractions'][str(ell)],
            'output_azimuthal_purity':oam['charge_fractions'][str(ell)],
            'cross_polarized_fraction':float(np.sum(abs(out[1])**2)/np.sum(abs(out)**2)),
            'oam':oam,'input_oam':oam_in,
            'scope':'weak frozen cycle-averaged disk; collapsed two-pass screen; no cavity mode selection'})
        if ell in (-1,1): fields[f'probe_ell_{ell}']=out
    return result,fields


def execute_case(case, root, output_root):
    """Run the real repository solver. Exceptions are saved and re-raised."""
    from .propagation import Grid2D
    from .thermal import DiskThermalMesh
    from .resonator import ThinDiskResonator
    from .coupled_resonator import (HotCavitySettings,run_coupled_hot_cavity,PlateAssembly,
                                    initial_vector_modes)
    from .vector_cavity import PlaneExchange
    from .numerical_quality import cavity_sampling_diagnostic
    from .thermomechanics import von_mises
    if not re.fullmatch(r'[A-Za-z0-9_-]+',case['id']): raise ValueError('unsafe case id')
    root=Path(root);out=Path(output_root)/case['id'];out.mkdir(parents=True,exist_ok=True)
    begin=time.monotonic();physical=physical_spec(case,root);source_hash=None;source=None
    if case['kind']=='frozen_assembly':
        source_folder=Path(output_root)/case['source_case']
        source=verify_saved(source_folder)
        if source is None or source['status']!='COMPLETED':
            raise ValueError('frozen-source dependency is missing, corrupt or nonconverged')
        source_hash=source['state_sha256']
        if source['identity']['solver_source_sha256']!=source_digest(root):
            raise ValueError('cannot reuse a source produced by different solver code')
    identity=case_identity(case,root,source_state_hash=source_hash)
    saved=verify_saved(out,identity)
    if saved is not None: return saved
    write_json(out/'summary.json',{'status':'RUNNING','identity':identity,'case':case})
    try:
        g=case['grid'];grid=Grid2D.square(g['optical_n'],g['window_m'])
        cavity=ThinDiskResonator(**physical['cavity'])
        mesh=DiskThermalMesh.disk(g['nr'],g['nz'],g['nphi'],radius_m=cavity.disk_diameter_m/2,
                                 thickness_m=cavity.disk_thickness_m)
        density=ContinuousDopant(**case['dopant']).cell_averages(mesh)
        assembly=json.loads((root/case['assembly_config']).read_text())
        assembly=_merge(assembly,case.get('assembly_overrides',{}))
        assembly['numerics']['mechanical']=case['mechanical']
        assembly['numerics']['plate_thermal_nz']=case['plate_thermal_nz']
        fractions=None;fields=None;gain_column=None;budget=None;history=[];r=None
        if case['kind']=='coupled':
            initial=initial_vector_modes(grid,cavity,tuple(case['initial_charges']))
            if case.get('initial_mix'):
                mix=initial_vector_modes(grid,cavity,(1,))[0]
                initial[0]+=case['initial_mix']*mix
            def progress(row):
                history.append(serializable(row))
                write_json(out/'progress.json',history)
                print('STAGE7V_PROGRESS '+json.dumps({'case':case['id'],**serializable(row)}),flush=True)
            p=case['pump']
            r=run_coupled_hot_cavity(grid,mesh,cavity,p['energy_J'],assembly,
               repetition_rate_Hz=p['repetition_rate_Hz'],pump_duration_s=p['duration_s'],
               pump_waist_m=p['waist_m'],density_m3=density,initial_fields=initial,
               mode_count=case['mode_count'],settings=HotCavitySettings(**case['settings']),progress=progress)
            fields=r.fields_used;temp=r.temperature;disp=r.displacement;screens=r.screens
            q=r.assembly_heat_W_m3
            if q is None:q=np.zeros(mesh.shape)
            if r.mean_fractions is not None:
                fractions=r.mean_fractions
                from .populations import HoYAGFourLevelParams
                prm=HoYAGFourLevelParams()
                coefficient=density.reshape(mesh.nz,-1)*(prm.sigma_em_laser_m2*fractions[2]-prm.sigma_abs_laser_m2*fractions[3])
                gain_column=PlaneExchange(grid,mesh).surface_on_grid(coefficient.sum(axis=0)*cavity.disk_thickness_m/mesh.nz)
            if r.cycle_heat is not None:budget=r.cycle_heat.budget
            status='COMPLETED' if r.converged else 'NONCONVERGED';reason=r.status
        elif case['kind']=='frozen_assembly':
            with np.load(Path(output_root)/case['source_case']/'state.npz',allow_pickle=False) as old:
                oldmesh=DiskThermalMesh(old['r_edges_m'],old['z_edges_m'],int(old['nphi']))
                q=conservative_heat_remap(old['heat_W_m3'],oldmesh,mesh)
                gain_column=old['gain_column'].copy()
                if gain_column.shape!=grid.shape:raise ValueError('frozen probes currently require the source optical grid')
            temp,disp,screens=PlateAssembly(mesh,grid,assembly).solve(q)
            status='COMPLETED';reason='frozen heat, finite plate and mechanics re-solved; no laser-power feedback'
        else: raise ValueError('unknown case kind')
        scalars={'output_W':None,'heat_W':float(np.sum(q*mesh.volumes_m3)),'absorbed_W':None,
                 'disk_peak_K':float(temp.disk_temperature_K.max()),
                 'plate_peak_K':float(temp.plate_temperature_K.max()),
                 'stress_peak_MPa':float(von_mises(disp.disk_stress_Pa).max()/1e6),
                 'thermal_balance_relative':float(temp.relative_balance_error),
                 'mechanical_residual':float(disp.free_residual_relative)}
        if budget is not None:
            scalars.update(output_W=budget['output_W'],heat_W=budget['heat_W'],absorbed_W=budget['pump_absorbed_W'],
                    energy_ledger_relative=budget['local_ledger_L1_error_over_incident'])
        arrays={'heat_W_m3':q,'density_m3':density,'r_edges_m':mesh.r_edges_m,'z_edges_m':mesh.z_edges_m,
            'nphi':np.array(mesh.nphi),'x_m':grid.x,'y_m':grid.y,
            'disk_temperature_K':temp.disk_temperature_K,'plate_temperature_K':temp.plate_temperature_K,
            'mean_opd_m':screens.mean_roundtrip_opd_m,'inward_jones':screens.inward_jones,
            'outward_jones':screens.outward_jones,'geometry_opd_m':screens.geometry_roundtrip_opd_m,
            'front_uz_m':screens.front_uz_m,'rear_uz_m':screens.rear_uz_m,
            'interface_flux_W_m2':temp.interface_flux_W_m2}
        if fields is not None:arrays['fields']=fields
        if fractions is not None:arrays['mean_fractions']=fractions
        if gain_column is not None:arrays['gain_column']=gain_column
        probes=[]
        if status=='COMPLETED' and gain_column is not None:
            probes,probe_fields=probe_frozen_disk(grid,cavity,screens,gain_column,
                                  charges=tuple(case['probes']),waist_m=case['probe_waist_m'])
            arrays.update(probe_fields)
        mode_power=[]
        if r is not None and r.optical_state is not None and r.optical_state.periodic_converged:
            opt=r.optical_state;lag=opt.period_cycles
            mode_power=(opt.history[-lag:,10:].mean(axis=0)*case['pump']['repetition_rate_Hz']).tolist()
            arrays['optical_history']=opt.history
        physical['assembly_numerics_used']=assembly['numerics']
        np.savez_compressed(out/'state.npz',**arrays)
        summary={'status':status,'reason':reason,'case':case,'identity':identity,
          'scalars':scalars,'elapsed_s':time.monotonic()-begin,'history':history,'physical':physical,
          'mode_power_W':mode_power,'probes':probes,
          'mirror_sampling':cavity_sampling_diagnostic(grid,cavity),
          'edge_power_fraction':None if fields is None else edge_power_fraction(fields),
          'state_sha256':file_sha256(out/'state.npz'),'mesh_converged':False,
          'physical_calibration_verified':False,'python_version':platform.python_version(),
          'scope':'coupled hot-cavity' if case['kind']=='coupled' else 'frozen heat assembly and weak seeded probes'}
        write_json(out/'summary.json',summary)
        print('STAGE7V_CASE '+json.dumps({k:serializable(v) for k,v in summary.items()
               if k not in ('history','physical','case','probes')},allow_nan=False),flush=True)
        print('STAGE7V_PROBES '+json.dumps({'case':case['id'],'probes':serializable(probes)},allow_nan=False),flush=True)
        return summary
    except Exception as exc:
        write_json(out/'summary.json',{'status':'ERROR','case':case,'identity':identity,
             'error':repr(exc),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-begin})
        raise


def compare_cases(folder_a, folder_b, *, kind='refinement', thresholds=None):
    thresholds=thresholds or THRESHOLDS
    a,b=verify_saved(folder_a),verify_saved(folder_b)
    if a is None or b is None:return {'status':'INCOMPLETE','reason':'missing or corrupt case/state'}
    if a['status']!='COMPLETED' or b['status']!='COMPLETED':
        return {'status':'FAIL','reason':'a case did not converge'}
    if a['case']['kind']!=b['case']['kind']:
        return {'status':'FAIL','reason':'cannot conflate coupled and frozen-source comparisons'}
    if a['identity']['solver_source_sha256']!=b['identity']['solver_source_sha256']:
        return {'status':'FAIL','reason':'different numerical source versions'}
    if kind!='sensitivity' and a['identity']['physics_fingerprint']!=b['identity']['physics_fingerprint']:
        return {'status':'FAIL','reason':'physical specimen/configuration changed during refinement'}
    if a['case']['kind']=='frozen_assembly' and a['identity']['source_state_sha256']!=b['identity']['source_state_sha256']:
        return {'status':'FAIL','reason':'frozen-source comparisons require the identical source hash'}
    values={};checks={}
    for key,lim in [('output_W','power_relative'),('heat_W','heat_relative'),('absorbed_W','absorbed_relative')]:
        aa,bb=a['scalars'].get(key),b['scalars'].get(key)
        if aa is not None and bb is not None:
            diff=abs(aa-bb)/max(abs(aa),abs(bb),1e-12)
            values[key+'_relative_difference']=diff;checks[key]=diff<thresholds[lim]
    for key in ('disk_peak_K','plate_peak_K'):
        diff=abs(a['scalars'][key]-b['scalars'][key]);values[key+'_difference']=diff
        checks[key]=diff<thresholds['temperature_K']
    with np.load(Path(folder_a)/'state.npz',allow_pickle=False) as na, np.load(Path(folder_b)/'state.npz',allow_pickle=False) as nb:
        opd=piston_removed_opd_error(na['mean_opd_m'],na['x_m'],na['y_m'],nb['mean_opd_m'],nb['x_m'],nb['y_m'])
        values['opd']=opd;checks['opd']=opd['rms_difference_m']<thresholds['opd_rms_m']
        if 'fields' in na and 'fields' in nb:
            comparison=compare_fields(na['fields'],na['x_m'],na['y_m'],nb['fields'],nb['x_m'],nb['y_m'])
            values['fields']=comparison
            # Different mode counts use subspace capture; no invented mode matching.
            overlap=comparison['minimum_subspace_capture'] if kind=='mode_comparison' else comparison['minimum_matched_overlap']
            checks['field']=overlap>thresholds['field_overlap']
    if kind=='sensitivity':return {'status':'MEASURED','differences':values,'note':'physical sensitivity, not numerical convergence'}
    return {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'differences':values}


def compile_report(plan, output_root):
    out=Path(output_root);groups=[]
    for group in plan['groups']:
        pairs=[]
        ids=group['cases']
        if group['kind'] in ('sensitivity','initialization'):
            combinations=[(ids[0], b) for b in ids[1:]]
        else:combinations=list(zip(ids[:-1],ids[1:]))
        for a,b in combinations:
            pairs.append({'a':a,'b':b,**compare_cases(out/a,out/b,kind=group['kind'],thresholds=plan['thresholds'])})
        statuses=[p['status'] for p in pairs]
        status=('FAIL' if 'FAIL' in statuses else 'INCOMPLETE' if 'INCOMPLETE' in statuses or len(ids)<group['minimum_levels']
                else 'MEASURED' if group['kind']=='sensitivity' else 'PASS')
        groups.append({'id':group['id'],'status':status,'pairs':pairs})
    ready={g['id']:g['status'] for g in groups}
    missing=[g for g in plan['required_groups'] if ready.get(g)!='PASS']
    issues=[];cases=[]
    for case in plan['cases']:
        summary=verify_saved(out/case['id'])
        if summary is None:
            raw=out/case['id']/'summary.json'
            status=json.loads(raw.read_text()).get('status','INCOMPLETE') if raw.exists() else 'NOT_RUN'
            cases.append({'id':case['id'],'status':status});issues.append(case['id']+':'+status);continue
        cases.append({'id':case['id'],'status':summary['status'],'scalars':summary['scalars'],
                      'mode_power_W':summary.get('mode_power_W',[]),'elapsed_s':summary['elapsed_s']})
        if summary['status']!='COMPLETED':issues.append(case['id']+':not_converged')
        if case['kind']=='coupled':
            if not summary['mirror_sampling']['nyquist_satisfied_over_radius']:issues.append(case['id']+':whole_aperture_sampling')
            if summary['edge_power_fraction']>THRESHOLDS['edge_power_fraction']:issues.append(case['id']+':edge_power')
            if summary['scalars'].get('energy_ledger_relative',np.inf)>THRESHOLDS['energy_ledger_relative']:issues.append(case['id']+':energy_ledger')
        if summary['scalars']['thermal_balance_relative']>THRESHOLDS['thermal_balance_relative']:issues.append(case['id']+':thermal_balance')
        if summary['scalars']['mechanical_residual']>THRESHOLDS['mechanical_residual']:issues.append(case['id']+':mechanical_equilibrium')
    measurements=[]
    if ready.get('mount_sensitivity')!='MEASURED':measurements.append('mount_sensitivity')
    baseline=verify_saved(out/'base256')
    if baseline is None or baseline.get('status')!='COMPLETED' or len(baseline.get('probes',[]))<7:
        measurements.append('seeded_probes')
    return {'schema_version':1,'profile':plan['profile'],'groups':groups,'cases':cases,
      'thresholds':plan['thresholds'],'unqualified_required_groups':missing,'case_quality_issues':issues,
      'uncompleted_measurements':measurements,
      'numerical_qualification_passed':not missing and not issues and not measurements,
      'physical_calibration_verified':False,'approved_for_trusted_nn_targets':False,
      'interpretation':'Missing runs are not evidence of accuracy. Numerical qualification is separate from experimental calibration.'}
