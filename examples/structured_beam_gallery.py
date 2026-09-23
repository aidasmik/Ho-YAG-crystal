"""Draw weak structured input/output probes through a nonuniform Ho:YAG disk."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hoyag.snapshots import snapshot_from_case
from hoyag.structured_beam_gallery import GallerySettings,simulate_gallery
from hoyag.validation_backend import sha256_file,source_manifest


def _relative_phase(field):
    intensity=abs(field)**2
    reference=np.angle(field.flat[np.argmax(intensity)])
    phase=np.angle(field*np.exp(-1j*reference))
    return np.ma.masked_where(intensity<.01*intensity.max(),phase)


def draw_beams(result, path):
    grid=result['grid']; extent=[grid.x[0]*1e3,grid.x[-1]*1e3,
                                  grid.y[0]*1e3,grid.y[-1]*1e3]
    rows=list(result['outcomes'].items())
    fig,axes=plt.subplots(len(rows),4,figsize=(15,13),constrained_layout=True)
    phase_cmap=plt.get_cmap('twilight').copy();phase_cmap.set_bad('#262935')
    for i,(name,case) in enumerate(rows):
        inp,out=case['input_field'],case['output_field']
        ii,oi=abs(inp)**2,abs(out)**2
        vmax=max(float(ii.max()),float(oi.max()))
        for col,(data,title) in enumerate(((ii,'Input irradiance'),(oi,'Output irradiance'))):
            image=axes[i,col].imshow(data,origin='lower',extent=extent,cmap='inferno',
                norm=PowerNorm(gamma=.55,vmin=0,vmax=vmax),interpolation='nearest')
            axes[i,col].set_title(f'{title}\n{case["input_power_W" if col==0 else "output_power_W"]:.4f} W')
            fig.colorbar(image,ax=axes[i,col],shrink=.72,label='W/m²')
        for col,(field,title) in enumerate(((inp,'Input phase'),(out,'Output phase')),2):
            image=axes[i,col].imshow(_relative_phase(field),origin='lower',extent=extent,
                 cmap=phase_cmap,vmin=-np.pi,vmax=np.pi,interpolation='nearest')
            axes[i,col].set_title(title+' (rad; global phase removed)')
            fig.colorbar(image,ax=axes[i,col],shrink=.72,ticks=[-np.pi,0,np.pi])
        axes[i,0].set_ylabel(f'{name}\ny (mm)\nDisk gain {case["disk_power_gain"]:.4f}×')
        for ax in axes[i]:
            ax.set_xlabel('x (mm)');ax.set_xlim(-2,2);ax.set_ylim(-2,2)
    fig.suptitle('Structured seed → one Ho:YAG traversal → '
                 f'{result["settings"].post_disk_distance_m:g} m free space\n'
                 'Frozen Stage 7W inversion; imposed nonuniform Ho concentration; weak-signal model',fontsize=14)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=150)
    plt.close(fig)


def draw_density(result,path):
    grid=result['grid']; density=result['density'].values_m3/1e26
    z_edges=np.asarray(result['z_edges_m'])
    extent=[grid.x[0]*1e3,grid.x[-1]*1e3,grid.y[0]*1e3,grid.y[-1]*1e3]
    data=np.ma.masked_where(density<=0,density)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#e6e8ec')
    low=float(data.min());high=float(data.max())
    for ax,iz,title in ((axes[0],0,'Entrance slice'),
                        (axes[1],len(density)//2,'Middle slice')):
        image=ax.imshow(data[iz],origin='lower',extent=extent,cmap=cmap,vmin=low,vmax=high)
        ax.set_title(title);ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)')
        ax.set_aspect('equal')
    y_index=int(np.argmin(abs(grid.y-result['settings'].bump_y_m)))
    image=axes[2].imshow(data[:,y_index,:],origin='lower',aspect='auto',
        extent=[grid.x[0]*1e3,grid.x[-1]*1e3,z_edges[0]*1e3,z_edges[-1]*1e3],
        cmap=cmap,vmin=low,vmax=high)
    axes[2].set_title(f'x–z cut at y={grid.y[y_index]*1e3:.2f} mm')
    axes[2].set_xlabel('x (mm)');axes[2].set_ylabel('depth z (mm)')
    fig.colorbar(image,ax=axes,label='Ho density (10²⁶ ions/m³)',shrink=.82)
    fig.suptitle('Declared, nonuniform Ho concentration; zero outside the 10-mm disk')
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-directory',type=Path,
                   default=ROOT/'results/local_stage7w/reference')
    p.add_argument('--output-directory',type=Path,
                   default=ROOT/'results/structured_beams')
    p.add_argument('--axial-end-to-end-fraction',type=float,default=.18)
    p.add_argument('--off-axis-peak-fraction',type=float,default=.30)
    p.add_argument('--bump-x-mm',type=float,default=.25)
    p.add_argument('--bump-y-mm',type=float,default=-.18)
    p.add_argument('--bump-width-mm',type=float,default=.65)
    p.add_argument('--post-disk-distance-m',type=float,default=.25)
    args=p.parse_args()
    snapshot=snapshot_from_case(args.case_directory)
    settings=GallerySettings(axial_end_to_end_fraction=args.axial_end_to_end_fraction,
        off_axis_peak_fraction=args.off_axis_peak_fraction,
        bump_x_m=args.bump_x_mm*1e-3,bump_y_m=args.bump_y_mm*1e-3,
        bump_width_m=args.bump_width_mm*1e-3,
        post_disk_distance_m=args.post_disk_distance_m)
    result=simulate_gallery(snapshot,settings)
    result['z_edges_m']=snapshot.arrays['z_edges_m']
    output=args.output_directory
    output.mkdir(parents=True,exist_ok=True)
    draw_beams(result,output/'input_output_beams.png')
    draw_density(result,output/'ho_density.png')
    arrays={'x_m':result['grid'].x,'y_m':result['grid'].y,
            'z_edges_m':result['z_edges_m'],
            'ho_density_m3':result['density'].values_m3}
    for index,(name,case) in enumerate(result['outcomes'].items()):
        arrays[f'mode_{index}_input_field']=case['input_field']
        arrays[f'mode_{index}_output_field']=case['output_field']
    np.savez_compressed(output/'fields.npz',**arrays)
    reference_summary=json.loads((args.case_directory/'summary.json').read_text())
    reference_path=args.case_directory.resolve()
    try:
        reference_label=str(reference_path.relative_to(ROOT))
    except ValueError:
        reference_label=str(reference_path)
    summary={'model':result['model'],
             'application_mode':'seeded_multipass_amplifier',
             'fidelity_mode':'weak_diagnostic_probe',
             'source_hash':source_manifest(ROOT)['source_hash'],
             'reference_state_id':result['reference_state_id'],
             'reference_state_sha256':reference_summary['state_sha256'],
             'reference_case':reference_label,
             'fields_sha256':sha256_file(output/'fields.npz'),
             'wavelength_m':result['wavelength_m'],
             'settings':asdict(result['settings']),
             'density_mean_active_m3':float(result['density'].values_m3[result['density'].values_m3>0].mean()),
             'density_min_active_m3':float(result['density'].values_m3[result['density'].values_m3>0].min()),
             'density_max_active_m3':float(result['density'].values_m3.max()),
             'modes':{name:{key:value for key,value in case.items() if key not in ('input_field','output_field')}
                      for name,case in result['outcomes'].items()},
             'time_kind':'steady_state',
             'limitations':['imposed density is not pump/heat/mechanics self-consistent',
                            'weak-signal probe; no population depletion by these inputs',
                            'single disk traversal; no specified multipass hardware topology']}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'output_directory':str(output.resolve()),
                      'density_range_m3':[summary['density_min_active_m3'],summary['density_max_active_m3']],
                      'power_gain_by_mode':{name:case['disk_power_gain'] for name,case in result['outcomes'].items()}}))


if __name__=='__main__':main()
