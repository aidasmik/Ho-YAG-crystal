"""Stage 5 demo. Uses the shared four-manifold resonator and thermal mesh.
Run: python examples/stage5_thermal_demo.py --nr 48 --nz 8 --iterations 10
All coating/cooling specifications and heat spectroscopy are model assumptions.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from hoyag.heat import HeatSpectroscopy
from hoyag.resonator import ThinDiskResonator
from hoyag.thermal import DiskThermalMesh,DiskCooling,ThermalBoundary
from hoyag.thermal_resonator import run_thermal_resonator
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/stage5_thermal.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results/stage5/generated')
    parser.add_argument('--iterations',type=int)
    parser.add_argument('--nr',type=int);parser.add_argument('--nz',type=int)
    parser.add_argument('--pump-uJ',type=float)
    parser.add_argument('--resume',type=Path,help='warm start from state.npz on the same mesh')
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    c=ThinDiskResonator(**cfg['cavity']);n=cfg['numerics'];th=cfg['thermal'];pump=cfg['pump']
    nr=args.nr if args.nr is not None else n['nr'];nz=args.nz if args.nz is not None else n['nz']
    mesh=DiskThermalMesh.disk(nr,nz,radius_m=c.disk_diameter_m/2,thickness_m=c.disk_thickness_m,radial_exponent=n['radial_exponent'])
    cooling=DiskCooling(rear=ThermalBoundary(th['rear_bath_temperature_K'],th['rear_contact_conductance_W_m2K']),
        front=ThermalBoundary(th['reference_temperature_K'],th['front_conductance_W_m2K']),
        rim=ThermalBoundary(th['reference_temperature_K'],th['rim_conductance_W_m2K']))
    energy=pump['energy_J'] if args.pump_uJ is None else args.pump_uJ*1e-6
    initial={}
    if args.resume:
        with np.load(args.resume) as d: initial={'initial_fractions':d['final_fractions'],'initial_log_photons':d['final_log_photons']}
    result=run_thermal_resonator(mesh,c,energy,pump['repetition_rate_Hz'],cooling=cooling,
        conductivity_W_mK=th['conductivity_W_mK'],dn_dT_K1=th['dn_dT_K1'],
        mass_density_kg_m3=th['mass_density_kg_m3'],heat_capacity_J_kgK=th['heat_capacity_J_kgK'],
        reference_temperature_K=th['reference_temperature_K'],pump_waist_m=pump['waist_radius_m'],
        pump_duration_fwhm_s=pump['duration_fwhm_s'],
        max_feedback_iterations=args.iterations if args.iterations is not None else n['max_feedback_iterations'],
        mode_tolerance=n['mode_tolerance'],heat_tolerance=n['heat_tolerance'],relaxation=n['relaxation'],
        optical_max_cycles=n['optical_max_cycles'],progress=lambda row:print(json.dumps(row),flush=True),**initial)
    args.output.mkdir(parents=True,exist_ok=True);heat=result.heat;temp=result.thermal
    summary={
        'status':result.status,'feedback_converged':result.feedback_converged,
        'optical_periodic_converged':result.optical.periodic_converged,
        'optical_period_cycles':result.optical.period_cycles,
        'pump_energy_J':energy,'nr':nr,'nz':nz,'nphi':1,
        'peak_cell_temperature_K':float(temp.temperature_K.max()),
        'minimum_cell_temperature_K':float(temp.temperature_K.min()),
        'single_pass_opd_peak_to_valley_m':float(np.ptp(result.single_pass_opd_m)),
        'lens':asdict(result.lens),'mode_radius_used_m':result.mode_radius_used_m,
        'mode_radius_predicted_m':result.mode_radius_predicted_m,
        'heat_budget':heat.budget,'boundary_heat_W':temp.boundary_heat_W,
        'thermal_balance_relative_error':temp.relative_balance_error,
        'heat_spectroscopy':asdict(HeatSpectroscopy()),'configuration':cfg,
        'iteration_history':result.iterations,
    }
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez_compressed(args.output/'state.npz',temperature_K=temp.temperature_K,
        heat_W_m3=heat.heat_W_m3.reshape(mesh.shape),pump_absorbed_W_m3=heat.pump_absorbed_W_m3.reshape(mesh.shape),
        stimulated_signal_W_m3=heat.stimulated_signal_W_m3.reshape(mesh.shape),
        fluorescence_W_m3=heat.fluorescence_W_m3.reshape(mesh.shape),
        storage_change_W_m3=heat.storage_change_W_m3.reshape(mesh.shape),
        single_pass_opd_m=result.single_pass_opd_m,
        final_fractions=heat.final_fractions,final_log_photons=heat.final_log_photons,
        r_edges_m=mesh.r_edges_m,z_edges_m=mesh.z_edges_m,r_m=mesh.r_m,z_m=mesh.z_m,
        optical_history=result.optical.history,optical_waveform=result.optical.waveform)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('configuration','heat_spectroscopy','iteration_history')},indent=2),flush=True)

if __name__=='__main__': main()
