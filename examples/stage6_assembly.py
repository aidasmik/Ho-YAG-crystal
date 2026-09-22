"""Finite plate + bond + 3D thermoelasticity + vector optical screens.

From repository root:
  python examples/stage6_assembly.py --stage5-state results/stage5/generated/state.npz
  python examples/stage6_assembly.py --illustrative-heat-W 0.634

The first consumes the actual Stage 5 heat map; the second explicitly constructs
a synthetic Gaussian source of the chosen TOTAL heat, not a pump-power estimate.
This is a frozen-source Stage 6 calculation, not a new lasing-output prediction.
"""
from __future__ import annotations
import argparse,json,hashlib
from dataclasses import asdict
from pathlib import Path
import numpy as np
from hoyag.thermal import DiskThermalMesh,ThermalBoundary
from hoyag.cooling_plate import DiskPlateHeatSolver,cooling_plate_mesh,sample_temperature,ThermalMaterial
from hoyag.thermomechanics import DiskPlateMesh,ElasticMaterial,BondedInterface,solve_disk_plate,von_mises,stress_tensor
from hoyag.stress_optics import build_hot_disk_screens,CubicElastoOptic,crystal_axes_111
ROOT=Path(__file__).resolve().parents[1]


def load_stage5_heat(path):
    with np.load(path,allow_pickle=False) as a:
        if not all(k in a for k in ('heat_W_m3','r_edges_m','z_edges_m')):
            raise ValueError('expected Stage 5 state.npz with heat_W_m3, r_edges_m, z_edges_m')
        q=a['heat_W_m3'].copy()
        if q.ndim!=3:raise ValueError('Stage 5 heat must have order (z,r,phi)')
        mesh=DiskThermalMesh(a['r_edges_m'],a['z_edges_m'],q.shape[2])
        if q.shape!=mesh.shape:raise ValueError('heat and edge dimensions disagree')
    return mesh,q


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=ROOT/'config/stage6_assembly.json')
    sources=ap.add_mutually_exclusive_group(required=True)
    sources.add_argument('--stage5-state',type=Path)
    sources.add_argument('--illustrative-heat-W',type=float)
    ap.add_argument('--output',type=Path,default=ROOT/'results/stage6/generated')
    ap.add_argument('--nr-mechanical',type=int)
    ap.add_argument('--ntheta-mechanical',type=int)
    ap.add_argument('--nz-mechanical',type=int)
    ap.add_argument('--contact-h',type=float)
    ap.add_argument('--support',choices=['clamped','roller','free'])
    args=ap.parse_args();cfg=json.loads(args.config.read_text())
    g=cfg['geometry'];th=cfg['thermal'];mech=cfg['mechanical'];op=cfg['optics'];num=cfg['numerics']
    if args.stage5_state:
        disk,q=load_stage5_heat(args.stage5_state)
        source={'kind':'Stage 5 archived heat field','path':str(args.stage5_state),'sha256':hashlib.sha256(args.stage5_state.read_bytes()).hexdigest()}
    else:
        if args.illustrative_heat_W<0 or not np.isfinite(args.illustrative_heat_W):raise ValueError('heat must be nonnegative')
        disk=DiskThermalMesh.disk(num['thermal_nr'],num['thermal_nz'],nphi=num['thermal_nphi'],radius_m=g['disk_radius_m'],thickness_m=g['disk_thickness_m'])
        profile=np.exp(-2*(disk.r_m/.0005)**2)[None,:,None]*np.ones(disk.shape)
        q=profile*args.illustrative_heat_W/np.sum(profile*disk.volumes_m3)
        source={'kind':'synthetic Gaussian heat demonstration','total_heat_W':args.illustrative_heat_W,'radius_m':.0005}
    if not np.isclose(disk.r_edges_m[-1],g['disk_radius_m'],rtol=0,atol=1e-12) or not np.isclose(disk.z_edges_m[-1],g['disk_thickness_m'],rtol=0,atol=1e-12):
        raise ValueError('source and assembly crystal dimensions differ')
    plate=cooling_plate_mesh(disk,radius_m=g['plate_radius_m'],thickness_m=g['plate_thickness_m'],nz=num['plate_thermal_nz'])
    contact=th['interface_conductance_W_m2K'] if args.contact_h is None else args.contact_h
    heat=DiskPlateHeatSolver(disk,plate,contact_conductance_W_m2K=contact,
        coolant=ThermalBoundary(th['coolant_temperature_K'],th['coolant_conductance_W_m2K']),
        disk_material=ThermalMaterial(**th['disk']),plate_material=ThermalMaterial(**th['plate']))
    temp=heat.steady(q)
    nm=num['mechanical'].copy()
    for arg,key in [(args.nr_mechanical,'nr'),(args.ntheta_mechanical,'ntheta'),(args.nz_mechanical,'nz_disk')]:
        if arg is not None:nm[key]=arg
    if args.nz_mechanical is not None:nm['nz_plate']=args.nz_mechanical
    fm=DiskPlateMesh.make(radius_m=g['disk_radius_m'],disk_thickness_m=g['disk_thickness_m'],
        plate_radius_m=g['plate_radius_m'],plate_thickness_m=g['plate_thickness_m'],**nm)
    td=sample_temperature(disk,temp.disk_temperature_K,fm.disk.centers_m)
    tp=sample_temperature(plate,temp.plate_temperature_K,fm.plate.centers_m,z_offset_m=g['disk_thickness_m'])
    dm=ElasticMaterial(**mech['disk']);pm=ElasticMaterial(**mech['plate'])
    support=args.support or mech['plate_support']
    disp=solve_disk_plate(fm,td,tp,disk_material=dm,plate_material=pm,
        interface=BondedInterface(**mech['bond']),support=support,front_pressure_Pa=mech['front_pressure_Pa'])
    n=num['optical_n'];window=num['optical_window_m'];dx=window/n
    x=(np.arange(n)-(n-1)/2)*dx;xx,yy=np.meshgrid(x,x,indexing='xy');xy=np.stack((xx,yy),axis=-1)
    screens=build_hot_disk_screens(fm,disp,disk,temp.disk_temperature_K,xy,
        index=op['index'],wavelength_m=op['wavelength_m'],dn_dT_K1=op['dn_dT_K1'],reference_temperature_K=op['reference_temperature_K'],
        material=dm,coefficients=CubicElastoOptic(**op['cubic_elasto_optic']),crystal_axes=crystal_axes_111(op['crystal_azimuth_rad']))
    radius=np.hypot(xx,yy);roi=radius<=op['report_radius_m']
    weight=np.exp(-2*radius**2/op['report_weight_radius_m']**2)*roi;weight/=weight.sum()
    mean=screens.mean_roundtrip_opd_m;avg=np.sum(weight*mean)
    jrt=screens.outward_jones@screens.inward_jones
    crossed=np.abs(jrt[...,1,0])**2
    summary={'source':source,'scope':'frozen-source linear bonded assembly, no laser-power re-solve',
       'geometry':g,'contact_conductance_W_m2K':contact,'coolant_conductance_W_m2K':th['coolant_conductance_W_m2K'],'mechanical_support':support,
       'disk_max_temperature_K':float(temp.disk_temperature_K.max()),'plate_max_temperature_K':float(temp.plate_temperature_K.max()),
       'input_heat_W':temp.input_heat_W,'interface_heat_W':float(np.sum(temp.interface_flux_W_m2*disk.face_areas_m2)),
       'thermal_balance_error_W':temp.balance_error_W,'mechanical_free_residual':disp.free_residual_relative,
       'disk_peak_von_mises_Pa':float(von_mises(disp.disk_stress_Pa).max()),
       'disk_peak_principal_tensile_Pa':float(np.linalg.eigvalsh(stress_tensor(disp.disk_stress_Pa)).max()),
       'plate_peak_von_mises_Pa':float(von_mises(disp.plate_stress_Pa).max()),
       'interface_peak_normal_traction_abs_Pa':float(np.max(abs(disp.interface_traction_on_disk_Pa[:,2]))),
       'interface_peak_shear_traction_Pa':float(np.linalg.norm(disp.interface_traction_on_disk_Pa[:,:2],axis=1).max()),
       'interface_peak_displacement_jump_m':float(np.linalg.norm(disp.interface_jump_m,axis=1).max()),
       'roi_radius_m':op['report_radius_m'],'optical_weight_radius_m':op['report_weight_radius_m'],
       'front_surface_pv_m':float(np.ptp(screens.front_uz_m[roi])),
       'rear_crystal_surface_pv_m':float(np.ptp(screens.rear_uz_m[roi])),
       'geometric_roundtrip_opd_pv_m':float(np.ptp(screens.geometry_roundtrip_opd_m[roi])),
       'thermal_roundtrip_opd_pv_m':float(2*np.ptp(screens.thermal_single_pass_opd_m[roi])),
       'photoelastic_mean_roundtrip_opd_pv_m':float(2*np.ptp(screens.photoelastic_mean_single_pass_opd_m[roi])),
       'total_mean_roundtrip_opd_pv_m':float(np.ptp(mean[roi])),
       'weighted_piston_removed_mean_wavefront_rms_m':float(np.sqrt(np.sum(weight*(mean-avg)**2))),
       'weighted_cross_polarized_fraction_reflected_disk':float(np.sum(weight*crossed)),
       'mechanical_nodes':len(fm.disk.nodes_m)+len(fm.plate.nodes_m),
       'mechanical_tetrahedra':len(fm.disk.tetrahedra)+len(fm.plate.tetrahedra),
       'configuration':cfg,'mechanical_mesh_used':nm}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez_compressed(args.output/'state.npz',
       heat_W_m3=q,disk_temperature_K=temp.disk_temperature_K,plate_temperature_K=temp.plate_temperature_K,
       disk_r_edges_m=disk.r_edges_m,disk_z_edges_m=disk.z_edges_m,plate_r_edges_m=plate.r_edges_m,plate_z_edges_m=plate.z_edges_m,
       interface_flux_W_m2=temp.interface_flux_W_m2,disk_contact_temperature_K=temp.disk_contact_temperature_K,plate_contact_temperature_K=temp.plate_contact_temperature_K,
       disk_nodes_m=fm.disk.nodes_m,disk_tetrahedra=fm.disk.tetrahedra,plate_nodes_m=fm.plate.nodes_m,plate_tetrahedra=fm.plate.tetrahedra,
       disk_u_m=disp.disk_u_m,plate_u_m=disp.plate_u_m,disk_stress_Pa=disp.disk_stress_Pa,plate_stress_Pa=disp.plate_stress_Pa,
       interface_jump_m=disp.interface_jump_m,interface_traction_on_disk_Pa=disp.interface_traction_on_disk_Pa,
       x_m=x,front_uz_m=screens.front_uz_m,rear_uz_m=screens.rear_uz_m,
       thermal_single_pass_opd_m=screens.thermal_single_pass_opd_m,
       geometry_roundtrip_opd_m=screens.geometry_roundtrip_opd_m,
       photoelastic_mean_single_pass_opd_m=screens.photoelastic_mean_single_pass_opd_m,
       mean_roundtrip_opd_m=mean,inward_jones=screens.inward_jones,outward_jones=screens.outward_jones)
    print(json.dumps({k:v for k,v in summary.items() if k!='configuration'},indent=2))

if __name__=='__main__':main()
