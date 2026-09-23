"""Stage 7W adapter reuses Stage 7V evidence, mesh and qualification machinery.

Plans use a new numerical closure fingerprint. No earlier Stage 7V result is
silently reused or requalified. The original one-mode control plan is untouched.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time
import traceback
import numpy as np
from .validation_plan import SCHEMA, make_plan, physics_from_repository, validate_plan
from .validation_metrics import stable_hash, ContinuousDensity, oam_spectrum
from .validation_backend import (make_grid,make_mesh,assembly_config,initial_modes,
          source_manifest,sha256_file,probe_diagnostics,_cavity_sampling)
from .validation_campaign import write_json, build_report
from .coupled_resonator import HotCavitySettings
from .polarization_tracking import TrackingSettings
from .polarized_resonator import run_polarization_hot_cavity


def make_polarized_plan(root):
    plan=make_plan(physics_from_repository(root),kind='coupled')
    # Replace the one-mode baseline by a declared two-mode baseline, not a
    # runtime promotion. Remove its now-duplicate modes_2 case from comparisons.
    plan['cases']=[case for case in plan['cases'] if case['id']!='modes_2']
    for case in plan['cases']:
        n=case['numerics']
        n['closure']='polarization_pair/1'
        n['tracking']=asdict(TrackingSettings())
        n['mode_count']=max(2,n['mode_count'])
        n['settings']['eigen_candidates']=max(n['mode_count']+2,n['settings']['eigen_candidates'])
        case.pop('spec_hash',None)
        case['spec_hash']=stable_hash(case)
    for group in plan['groups']:
        ids=['reference' if x=='modes_2' else x for x in group['case_ids']]
        group['case_ids']=list(dict.fromkeys(ids))
        group['required_successive_pairs']=(len(group['case_ids'])-1 if group['name']=='initialization'
                                           else min(2,len(group['case_ids'])-1))
    plan['closure']='Stage 7W, explicitly paired vector modes'
    plan['limits']+=['Old one-mode controls remain a separate Stage 7V campaign.',
                     'Polarization-family guard can require a larger retained basis; this is not convergence.',
                     'All old acceptance thresholds are retained; modal power, coherency and gain gates are additional.']
    return validate_plan(plan)


def coupled_case_polarized(case,progress=None):
    from .resonator import ThinDiskResonator
    from .populations import HoYAGFourLevelParams
    from .vector_cavity import PlaneExchange
    if case['numerics'].get('closure')!='polarization_pair/1':
        raise ValueError('not an explicit Stage 7W numerical case')
    grid,mesh,cfg=make_grid(case),make_mesh(case),assembly_config(case)
    cavity=ThinDiskResonator(**case['physics']['cavity'])
    pump=case['physics']['pump']
    settings=HotCavitySettings(**case['numerics']['settings'])
    tracking=TrackingSettings(**case['numerics']['tracking'])
    density_spec=ContinuousDensity(**case['physics']['density'])
    density=density_spec.cell_average(mesh,case['numerics']['density_quadrature_order'])
    p=HoYAGFourLevelParams(N_total_m3=density_spec.mean_m3)
    result=run_polarization_hot_cavity(grid,mesh,cavity,pump['energy_J'],cfg,
        repetition_rate_Hz=pump['repetition_rate_Hz'],pump_duration_s=pump['duration_s'],
        pump_waist_m=pump['waist_m'],params=p,density_m3=density,
        initial_fields=initial_modes(grid,cavity,case['numerics']),mode_count=case['numerics']['mode_count'],
        settings=settings,tracking=tracking,progress=progress)
    arrays={'fields_used':result.fields_used,'x_m':grid.x,'y_m':grid.y,
            'r_edges_m':mesh.r_edges_m,'z_edges_m':mesh.z_edges_m,'density_m3':density}
    base={'status':'completed' if result.converged else 'not_converged','solver_status':result.status,
          'field_semantics':'incoherent_cavity_modes','history':result.history,
          'solver_metadata':result.metadata,'sampling':_cavity_sampling(grid,case['physics']),
          'physical_density_fingerprint':density_spec.fingerprint,
          'density_volume_mean_m3':float(np.sum(density*mesh.volumes_m3)/mesh.volumes_m3.sum()),
          'scope':'Stage 7W self-consistent polarization-resolved adiabatic hot cavity, finite plate and bonded mechanics.'}
    if result.cycle_heat is None or result.mean_fractions is None:
        base['metrics']={'coupled_converged':False}
        return base,arrays
    heat=result.cycle_heat;budget=heat.budget;temp=result.temperature
    er=None if result.eigenfields is None else float(np.max(result.eigenfields.residuals))
    base['metrics']={'output_W':budget['output_W'],'pump_absorbed_W':budget['pump_absorbed_W'],
        'heat_W':budget['heat_W'],'peak_disk_K':float(temp.disk_temperature_K.max()),
        'peak_plate_K':float(temp.plate_temperature_K.max()),
        'thermal_energy_error_relative':temp.relative_balance_error,
        'mechanical_residual':result.displacement.free_residual_relative,
        'optical_population_error_relative':budget['local_ledger_L1_error_over_incident'],
        'eigen_residual':er if er is not None and np.isfinite(er) else None,
        'coupled_converged':result.converged,'mode_power_W':list(budget['output_W_by_mode'])}
    arrays.update(mean_roundtrip_opd_m=result.screens.mean_roundtrip_opd_m,
        mean_fractions=result.mean_fractions,raw_heat_W_m3=heat.heat_W_m3.reshape(mesh.shape),
        assembly_heat_W_m3=result.assembly_heat_W_m3,disk_temperature_K=temp.disk_temperature_K,
        plate_temperature_K=temp.plate_temperature_K,interface_flux_W_m2=temp.interface_flux_W_m2,
        fields_predicted=result.fields_predicted)
    f=result.mean_fractions.reshape(4,*mesh.shape)
    gain=(p.sigma_em_laser_m2*f[2]-p.sigma_abs_laser_m2*f[3])*density
    column=np.sum(gain*np.diff(mesh.z_edges_m)[:,None,None],axis=0)
    log_gain=PlaneExchange(grid,mesh,order=settings.projection_order).surface_on_grid(column)
    _,base['probes']=probe_diagnostics(grid,result.screens,log_gain,case['physics'])
    base['mode_oam']=[oam_spectrum(f,grid) for f in result.fields_used]
    base['energy_budget']=budget
    return base,arrays


def execute_polarized_case(case,directory,manifest,*,resume=True):
    out=Path(directory)/case['id'];out.mkdir(parents=True,exist_ok=True)
    key=stable_hash({'schema':SCHEMA,'spec_hash':case['spec_hash'],'source_hash':manifest['source_hash']})
    path=out/'summary.json'
    if path.exists() and resume:
        prev=json.loads(path.read_text())
        if prev.get('execution_key')!=key:raise ValueError('stale Stage 7W cache; use a new output directory')
        if prev.get('status')=='completed':
            if not (out/'state.npz').is_file() or sha256_file(out/'state.npz')!=prev.get('state_sha256'):
                raise ValueError('missing or corrupt numerical state')
            return prev
    record={'schema':SCHEMA,'id':case['id'],'kind':case['kind'],'purpose':case['purpose'],
        'spec_hash':case['spec_hash'],'physics_hash':case['physics_hash'],'execution_key':key,
        'source_hash':manifest['source_hash'],'status':'running','case':case,'dataset_ready':False}
    write_json(path,record)
    # Previous failed attempts remain in their log; this file describes THIS attempt.
    (out/'iterations.jsonl').write_text('')
    begin=time.perf_counter()
    def progress(row):
        from .validation_campaign import json_safe
        line=json.dumps(json_safe(row),allow_nan=False)
        with (out/'iterations.jsonl').open('a') as stream:stream.write(line+'\n')
        print('STAGE7W_ITERATION '+line,flush=True)
    try:
        data,arrays=coupled_case_polarized(case,progress=progress)
        for name,array in arrays.items():
            if array is None:continue
            if np.asarray(array).dtype.kind in 'fc' and not np.all(np.isfinite(array)):
                raise FloatingPointError(f'nonfinite array {name}')
        np.savez_compressed(out/'state.npz',**{k:v for k,v in arrays.items() if v is not None})
        record.update(data);record['state_sha256']=sha256_file(out/'state.npz')
    except Exception as exc:
        record.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    record['wall_seconds']=time.perf_counter()-begin
    write_json(path,record)
    return record
