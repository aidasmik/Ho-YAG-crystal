"""Thermal-only refinement of an existing cycle-resolved heat source."""
from pathlib import Path
import json
import numpy as np
from hoyag.propagation import Grid2D
from hoyag.thermal import DiskHeatSolver,ThermalBoundary
from hoyag.thermal_optics import radial_source_to_volume,thermal_opd,fit_thermal_lens

def refine(folder):
    folder=Path(folder);data=np.load(folder/'fields.npz');s=json.loads((folder/'summary.json').read_text())
    rows=[];w=s['cavity']['waist_m'];bulk=s['configuration']['bulk']
    for nxy,nz in [(64,8),(96,12),(128,16),(192,24)]:
        solver=DiskHeatSolver(Grid2D.square(nxy,s['cavity']['disk_diameter_m']),nz=nz,
            thickness_m=s['cavity']['disk_thickness_m'],diameter_m=s['cavity']['disk_diameter_m'],
            conductivity_W_mK=bulk['conductivity_W_mK'],
            density_kg_m3=bulk['density_kg_m3'],heat_capacity_J_kgK=bulk['heat_capacity_J_kgK'],
            boundary=ThermalBoundary(**s['configuration']['boundary']))
        q,mapping=radial_source_to_volume(data['radius_m'],data['radial_heat_W_m3'],data['area_m2'],float(data['optical_dz_m']),solver)
        out=solver.steady(q);opd=thermal_opd(out.temperature_K,solver,reference_K=solver.boundary.sink_temperature_K,dn_dT_K1=bulk['dn_dT_K1'])
        x,y=solver.grid.mesh
        fit=fit_thermal_lens(opd,solver.grid,fit_radius_m=2*w,weights=np.exp(-2*(x*x+y*y)/w**2))
        rows.append({'grid':[nz,nxy,nxy],'max_rise_K':float(out.temperature_K.max()-solver.boundary.sink_temperature_K),
            'single_pass_power_m1':fit.mean_power_m1,'heat_W':out.deposited_power_W,'flux_residual_W':out.residual_W})
        print(rows[-1],flush=True)
    (folder/'thermal_refinement.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();refine(a.folder)
