"""Run signed heat -> 3-D disk temperature -> thermal phase -> modal feedback.

Example: python examples/stage5_demo.py --pump-uJ 1000
All mount and mean-fluorescence assumptions are recorded in the output metadata.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from hoyag.propagation import Grid2D
from hoyag.resonator import ThinDiskResonator
from hoyag.thermal import DiskHeatSolver,ThermalBoundary
from hoyag.thermal_resonator import thermal_feedback

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/stage5_thermal.json')
    parser.add_argument('--cavity-config',type=Path)
    parser.add_argument('--pump-uJ',type=float,default=1000.)
    parser.add_argument('--nxy',type=int)
    parser.add_argument('--nz',type=int)
    parser.add_argument('--max-outer',type=int)
    parser.add_argument('--output',type=Path,default=ROOT/'results/stage5/generated')
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    cavity_cfg=json.loads((args.cavity_config or ROOT/cfg['cavity_config']).read_text())
    cavity=ThinDiskResonator(**cavity_cfg['cavity'])
    num=cfg['numerics'];bulk=cfg['bulk'];nxy=args.nxy or num['transverse_points'];nz=args.nz or num['thermal_z_cells']
    grid=Grid2D.square(nxy,cavity.disk_diameter_m)
    solver=DiskHeatSolver(grid,nz=nz,thickness_m=cavity.disk_thickness_m,
        diameter_m=cavity.disk_diameter_m,conductivity_W_mK=bulk['conductivity_W_mK'],
        density_kg_m3=bulk['density_kg_m3'],heat_capacity_J_kgK=bulk['heat_capacity_J_kgK'],
        boundary=ThermalBoundary(**cfg['boundary']))
    args.output.mkdir(parents=True,exist_ok=True)
    def progress(row):
        print(json.dumps(row),flush=True)
        with (args.output/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    start=time.perf_counter()
    result=thermal_feedback(cavity,solver,args.pump_uJ*1e-6,
        repetition_rate_Hz=cavity_cfg['pump']['repetition_rate_Hz'],
        pump_waist_m=cavity_cfg['pump']['waist_radius_m'],
        radial_points=num['radial_quadrature_points'],z_slices=num['optical_z_slices'],
        max_outer=args.max_outer or num['max_outer'],mixing=num['mixing'],
        tolerance=num['relative_outer_tolerance'],dn_dT_K1=bulk['dn_dT_K1'],progress=progress)
    hist=result['history'];temp=result['temperature'];fit=result['lens_fit']
    summary={'stage':5,'converged':result['converged'],'reason':result['reason'],
        'runtime_s':time.perf_counter()-start,'cavity':cavity.summary(),
        'pump_energy_J':args.pump_uJ*1e-6,'configuration':cfg,
        'actual_thermal_grid':[nz,nxy,nxy],'actual_radial_points':num['radial_quadrature_points'],
        'cold_output_W':hist[0]['output_W'],'hot_output_W':hist[-1]['output_W'],
        'heat_budget_W':result['heat']['budget_W'],
        'max_temperature_K':float(temp.temperature_K.max()),
        'max_temperature_rise_K':float(temp.temperature_K.max()-solver.boundary.sink_temperature_K),
        'single_pass_lens_power_m1':fit.mean_power_m1,
        'single_pass_focal_length_m':1/fit.mean_power_m1 if fit.mean_power_m1!=0 else None,
        'nonquadratic_opd_rms_m':fit.residual_rms_m,
        'thermal_flux_residual_W':temp.residual_W,'optical_energy_closure_relative':result['heat']['closure_relative'],
        'source_mapping':result['mapping'],'outer_iterations':len(hist),
        'source_energy_provenance':result['energies_provenance']}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    with (args.output/'outer_iterations.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(hist[0]));w.writeheader();w.writerows(hist)
    np.savez_compressed(args.output/'fields.npz',temperature_K=temp.temperature_K,
        source_W_m3=result['source_W_m3'],opd_m=result['single_pass_opd_m'],
        fitted_opd_m=fit.fitted_opd_m,mask=solver.mask,x_m=grid.x,y_m=grid.y,z_m=solver.z_m,
        radial_heat_W_m3=result['heat']['source_W_m3'],radius_m=result['radius_m'],
        area_m2=result['model'].area,optical_dz_m=result['model'].dz,
        fractions=result['heat']['final_fractions'],log_photons=result['heat']['final_log_photons'])
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
