"""Run Stage 7 and save both converged and explicitly nonconverged results.

python examples/stage7_hot_cavity.py --quick
python examples/stage7_hot_cavity.py --pump-uJ 1000 --require-converged

Quick mode is an integration demonstration, not a mesh-converged prediction.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from dataclasses import asdict
import numpy as np
from hoyag.propagation import Grid2D
from hoyag.thermal import DiskThermalMesh
from hoyag.resonator import ThinDiskResonator
from hoyag.coupled_resonator import run_coupled_hot_cavity, HotCavitySettings

ROOT=Path(__file__).resolve().parents[1]


def clean_json(obj):
    if isinstance(obj,dict):return {k:clean_json(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [clean_json(v) for v in obj]
    if isinstance(obj,np.ndarray):return clean_json(obj.tolist())
    if isinstance(obj,(np.floating,float)):return float(obj) if np.isfinite(obj) else None
    if isinstance(obj,np.integer):return int(obj)
    if isinstance(obj,np.bool_):return bool(obj)
    return obj


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/stage7_hot_cavity.json')
    parser.add_argument('--pump-uJ',type=float)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--max-outer',type=int)
    parser.add_argument('--modes',type=int)
    parser.add_argument('--require-converged',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'results/stage7/generated')
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    cavity_cfg=json.loads((ROOT/cfg['cavity_configuration']).read_text())
    assembly_cfg=json.loads((ROOT/cfg['assembly_configuration']).read_text())
    cavity=ThinDiskResonator(**cavity_cfg['cavity'])
    mesh_cfg=cfg['mesh'].copy();options=cfg['feedback'].copy()
    if args.quick:
        mesh_cfg.update(optical_n=128,nr=16,nz=2,nphi=4)
        assembly_cfg['numerics']['mechanical'].update(nr=4,outer_rings=2,ntheta=16,nz_disk=2,nz_plate=2)
        assembly_cfg['numerics']['plate_thermal_nz']=6
        options.update(max_outer_iterations=12,field_tolerance=3e-3,
                       temperature_tolerance_K=.05,displacement_tolerance_m=5e-10,
                       optical_max_cycles=340,eigen_candidates=3)
    if args.max_outer is not None:options['max_outer_iterations']=args.max_outer
    settings=HotCavitySettings(**options)
    grid=Grid2D.square(mesh_cfg['optical_n'],mesh_cfg['optical_window_m'])
    mesh=DiskThermalMesh.disk(mesh_cfg['nr'],mesh_cfg['nz'],mesh_cfg['nphi'],
                       radius_m=cavity.disk_diameter_m/2,thickness_m=cavity.disk_thickness_m)
    pump=cfg['pump'];energy=pump['energy_J'] if args.pump_uJ is None else args.pump_uJ*1e-6
    def progress(row):print('STAGE7_ITERATION '+json.dumps(clean_json(row),allow_nan=False),flush=True)
    result=run_coupled_hot_cavity(grid,mesh,cavity,energy,assembly_cfg,
                repetition_rate_Hz=pump['repetition_rate_Hz'],pump_duration_s=pump['duration_s'],
                pump_waist_m=pump['waist_m'],mode_count=args.modes or cfg['mode_count'],
                settings=settings,progress=progress)
    args.output.mkdir(parents=True,exist_ok=True)
    summary={'converged':result.converged,'status':result.status,'quick_demonstration':args.quick,
             'outer_iterations':len(result.history),'mesh':mesh_cfg,'metadata':result.metadata,
             'assembly_configuration':assembly_cfg,'history':result.history,
             'energy_budget':None if result.cycle_heat is None else result.cycle_heat.budget}
    if result.history:summary['last_iteration']=result.history[-1]
    (args.output/'summary.json').write_text(json.dumps(clean_json(summary),indent=2,allow_nan=False)+'\n')
    if result.history:
        with (args.output/'history.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(result.history[0]));writer.writeheader();writer.writerows(result.history)
    arrays={'fields_used':result.fields_used,'x_m':grid.x,'y_m':grid.y,
            'r_edges_m':mesh.r_edges_m,'z_edges_m':mesh.z_edges_m,
            'roundtrip_losses_used':result.roundtrip_losses_used}
    if result.fields_predicted is not None:arrays['fields_predicted']=result.fields_predicted
    if result.mean_fractions is not None:arrays['mean_fractions']=result.mean_fractions
    if result.assembly_heat_W_m3 is not None:arrays['assembly_heat_W_m3']=result.assembly_heat_W_m3
    if result.cycle_heat is not None:arrays['raw_heat_W_m3']=result.cycle_heat.heat_W_m3.reshape(mesh.shape)
    if result.temperature is not None:
        arrays['disk_temperature_K']=result.temperature.disk_temperature_K
        arrays['plate_temperature_K']=result.temperature.plate_temperature_K
        arrays['interface_flux_W_m2']=result.temperature.interface_flux_W_m2
    if result.screens is not None:
        arrays['mean_roundtrip_opd_m']=result.screens.mean_roundtrip_opd_m
        arrays['inward_jones']=result.screens.inward_jones
        arrays['outward_jones']=result.screens.outward_jones
    if result.optical_state is not None:
        arrays['optical_history']=result.optical_state.history
        arrays['optical_waveform']=result.optical_state.waveform
    if result.eigenfields is not None:
        arrays['candidate_eigenvalues']=result.eigenfields.candidate_eigenvalues
        arrays['eigen_residuals']=result.eigenfields.residuals
    np.savez_compressed(args.output/'state.npz',**arrays)
    print('STAGE7_SUMMARY '+json.dumps(clean_json({k:v for k,v in summary.items()
                       if k not in ('assembly_configuration','history')}),allow_nan=False),flush=True)
    if args.require_converged and not result.converged:
        raise SystemExit('Stage 7 did not converge; partial results were saved with status=false')

if __name__=='__main__':main()
