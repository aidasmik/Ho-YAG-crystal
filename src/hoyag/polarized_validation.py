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


def add_six_mode_candidate(plan, diagnostic):
    """Opt-in plan amendment only after a finite-bank diagnostic supports six.

    The coupled solver still applies its own family guard at every iteration.
    """
    if not diagnostic.get('modes_6_complete_family_candidate'):
        raise ValueError('six-mode retained boundary has not been supported')
    plan=deepcopy(validate_plan(plan))
    if any(c['id']=='modes_6' for c in plan['cases']):
        raise ValueError('six-mode candidate is already present')
    reference=next(c for c in plan['cases'] if c['id']=='reference')
    case=deepcopy(reference)
    case['id']='modes_6'
    case['purpose']='Six-mode complete-family candidate; requires independent coupled guard'
    case['numerics']['mode_count']=6
    case['numerics']['settings']['eigen_candidates']=max(10,case['numerics']['settings']['eigen_candidates'])
    case.pop('spec_hash',None)
    case['spec_hash']=stable_hash(case)
    plan['cases'].append(case)
    for group in plan['groups']:
        if group['name']=='retained_modes':
            index=group['case_ids'].index('modes_8')
            group['case_ids'].insert(index,'modes_6')
    plan['limits'].append('Six-mode candidate was added from a finite-bank diagnostic; coupled validity remains unproven.')
    return validate_plan(plan)


def coupled_case_polarized(case,progress=None,state_callback=None,
                          checkpoint_callback=None,resume_state=None):
    from .resonator import ThinDiskResonator
    from .populations import HoYAGFourLevelParams
    from .vector_cavity import PlaneExchange, VectorRoundTrip
    from .thermomechanics import DiskPlateMesh
    from .cooling_plate import cooling_plate_mesh
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
        settings=settings,tracking=tracking,progress=progress,
        state_callback=state_callback,checkpoint_callback=checkpoint_callback,
        resume_state=resume_state)
    arrays={'fields_used':result.fields_used,'x_m':grid.x,'y_m':grid.y,
            'r_edges_m':mesh.r_edges_m,'z_edges_m':mesh.z_edges_m,'density_m3':density}
    from .thermal_resonator import area_averaged_lg0
    arrays['pump_input_average_irradiance_W_m2']=(
        pump['energy_J']*pump['repetition_rate_Hz']*
        area_averaged_lg0(mesh,pump['waist_m']).reshape(mesh.nr,mesh.nphi))
    plate_mesh=cooling_plate_mesh(mesh,radius_m=cfg['geometry']['plate_radius_m'],
        thickness_m=cfg['geometry']['plate_thickness_m'],
        nz=cfg['numerics']['plate_thermal_nz'])
    arrays.update(plate_r_edges_m=plate_mesh.r_edges_m,
                  plate_z_edges_m=plate_mesh.z_edges_m)
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
        inward_jones=result.screens.inward_jones,
        outward_jones=result.screens.outward_jones,
        geometry_roundtrip_opd_m=result.screens.geometry_roundtrip_opd_m,
        mean_fractions=result.mean_fractions,raw_heat_W_m3=heat.heat_W_m3.reshape(mesh.shape),
        assembly_heat_W_m3=result.assembly_heat_W_m3,disk_temperature_K=temp.disk_temperature_K,
        plate_temperature_K=temp.plate_temperature_K,interface_flux_W_m2=temp.interface_flux_W_m2,
        disk_contact_temperature_K=temp.disk_contact_temperature_K,
        plate_contact_temperature_K=temp.plate_contact_temperature_K,
        fields_predicted=result.fields_predicted,
        disk_displacement_m=result.displacement.disk_u_m,
        plate_displacement_m=result.displacement.plate_u_m,
        disk_stress_Pa=result.displacement.disk_nodal_stress_Pa,
        plate_stress_Pa=result.displacement.plate_nodal_stress_Pa)
    geometry=cfg['geometry']
    mechanical=DiskPlateMesh.make(radius_m=geometry['disk_radius_m'],
        disk_thickness_m=geometry['disk_thickness_m'],
        plate_radius_m=geometry['plate_radius_m'],
        plate_thickness_m=geometry['plate_thickness_m'],
        **cfg['numerics']['mechanical'])
    arrays.update(disk_nodes_m=mechanical.disk.nodes_m, plate_nodes_m=mechanical.plate.nodes_m,
                  disk_tetrahedra=mechanical.disk.tetrahedra,
                  plate_tetrahedra=mechanical.plate.tetrahedra)
    f=result.mean_fractions.reshape(4,*mesh.shape)
    gain=(p.sigma_em_laser_m2*f[2]-p.sigma_abs_laser_m2*f[3])*density
    column=np.sum(gain*np.diff(mesh.z_edges_m)[:,None,None],axis=0)
    log_gain=PlaneExchange(grid,mesh,order=settings.projection_order).surface_on_grid(column)
    arrays['single_pass_log_gain']=log_gain
    final_operator=VectorRoundTrip(grid,cavity,result.screens,log_gain)
    arrays['output_coupler_fields']=np.asarray([
        final_operator.propagate(field)[1] for field in result.fields_used])
    _,base['probes']=probe_diagnostics(grid,result.screens,log_gain,case['physics'])
    base['mode_oam']=[oam_spectrum(f,grid) for f in result.fields_used]
    base['energy_budget']=budget
    base['cooling_metrics']={
        'bulk_crystal_heat_W':float(budget['heat_W']),
        'coating_heat_W':None,
        'crystal_to_plate_W':float(np.sum(temp.interface_flux_W_m2*mesh.face_areas_m2)),
        'assembly_to_coolant_W':float(sum(temp.outward_heat_W.values())),
        'thermal_storage_W':0.,
        'coolant_inlet_K':None,'coolant_outlet_K':None,
        'cooling_capacity_W':None,'capacity_utilization':None,
        'thermal_balance_error_W':float(temp.balance_error_W),
        'model':'steady finite plate with prescribed coolant temperature'}
    return base,arrays


def execute_polarized_case(case,directory,manifest,*,resume=True,live_snapshots=False):
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
    from .checkpoints import load_checkpoint,save_checkpoint
    if not resume and (out/'checkpoints/latest.json').exists():
        raise ValueError('choose a new output directory for a fresh attempt')
    prior_checkpoint=load_checkpoint(out/'checkpoints',case,manifest) if resume else None
    record={'schema':SCHEMA,'id':case['id'],'kind':case['kind'],'purpose':case['purpose'],
        'spec_hash':case['spec_hash'],'physics_hash':case['physics_hash'],'execution_key':key,
        'source_hash':manifest['source_hash'],'status':'running','case':case,'dataset_ready':False,
        'resumed_from_iteration':None if prior_checkpoint is None else prior_checkpoint['iteration']}
    write_json(path,record)
    # Previous failed attempts remain in their log; this file describes THIS attempt.
    if prior_checkpoint is None:(out/'iterations.jsonl').write_text('')
    begin=time.perf_counter()
    if live_snapshots:
        from .live_control import ControlState
        controller=ControlState(out)
    def progress(row):
        from .validation_campaign import json_safe
        line=json.dumps(json_safe(row),allow_nan=False)
        with (out/'iterations.jsonl').open('a') as stream:stream.write(line+'\n')
        print('STAGE7W_ITERATION '+line,flush=True)
    def publish_state(row,arrays):
        from .snapshots import ScientificSnapshot,save_snapshot
        metadata={
            'schema_version':1,'run_id':str(out.resolve()),
            'configuration_hash':case['spec_hash'],'source_hash':manifest['source_hash'],
            'state_id':f"{case['id']}-iteration-{row['iteration']:04d}",
            'application_mode':'oscillator_reference','fidelity_mode':'live_reference',
            'solver_status':'accepted_outer_iteration',
            'time_kind':'outer_iteration','t_sim_s':None,
            'outer_iteration':row['iteration'],'pulse_id':None,
            'averaging_interval':'periodic optical cycle',
            't_published_wall':time.time(),
            'mesh_ids':{'optical':case['spec_hash'],'material':case['spec_hash']},
            'coordinate_frames':{'optical':'laboratory x/y, metres',
                                 'material':'cylindrical z/phi/r'},
            'units':{'output_coupler_fields':'complex amplitude, shape only',
                     'disk_temperature_K':'K','plate_temperature_K':'K',
                     'raw_heat_W_m3':'W/m^3','disk_displacement_m':'m'},
            'optical_state_kind':'incoherent_cavity_modes',
            'modal_powers_W':row.get('output_W_by_mode'),
            'field_normalization':'shape-normalized; modal powers supplied separately',
            'wavelength_m':case['physics']['cavity']['wavelength_m'],
            'disk_conductivity_W_mK':case['physics']['assembly']['thermal']['disk']['conductivity_W_mK'],
            'energy_metrics':{'output_W':row['output_W'],'heat_W':row['heat_W']},
            'cooling_metrics':{'finite_cooling_capacity':False},
            'subsolver_timestamps':None,'sensor_flags':{'experimental_sensors':False},
            'approximation_flags':['adiabatic_cycle_averaged_modes','incoherent_modal_mixture',
                                   'outer_iteration_not_physical_time','finite_candidate_bank'],
            'convergence_diagnostics':row.get('convergence_gates'),
            'available_arrays':sorted(arrays),
        }
        save_snapshot(ScientificSnapshot(metadata,arrays),out/'snapshots')
        controller.after_iteration()
    def checkpoint_state(info,arrays):
        save_checkpoint(out/'checkpoints',case,manifest,info,arrays)
    try:
        data,arrays=coupled_case_polarized(case,progress=progress,
            state_callback=publish_state if live_snapshots else None,
            checkpoint_callback=checkpoint_state,resume_state=prior_checkpoint)
        for name,array in arrays.items():
            if array is None:continue
            if np.asarray(array).dtype.kind in 'fc' and not np.all(np.isfinite(array)):
                raise FloatingPointError(f'nonfinite array {name}')
        np.savez_compressed(out/'state.npz',**{k:v for k,v in arrays.items() if v is not None})
        record.update(data);record['state_sha256']=sha256_file(out/'state.npz')
    except Exception as exc:
        from .live_control import UserCancelled
        if isinstance(exc,UserCancelled):
            record.update(status='cancelled',error=str(exc))
        else:
            record.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    record['wall_seconds']=time.perf_counter()-begin
    write_json(path,record)
    return record
