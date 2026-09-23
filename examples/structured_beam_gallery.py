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
from hoyag.structured_beam_gallery import GallerySettings,PHASE_MASKS,SOLVER_MODES,simulate_gallery
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
    y_index=int(np.argmin(abs(grid.y)))
    fig,axes=plt.subplots(len(rows),6,figsize=(22,14),constrained_layout=True)
    axes=np.asarray(axes)
    phase_cmap=plt.get_cmap('twilight').copy();phase_cmap.set_bad('#262935')
    for i,(name,case) in enumerate(rows):
        inp,out=case['input_field'],case['output_field']
        ii=case.get('input_intensity',abs(inp)**2)
        oi=case.get('output_intensity',abs(out)**2)
        vmax=max(float(ii.max()),float(oi.max()))
        image=axes[i,0].imshow(ii,origin='lower',extent=extent,cmap='inferno',
            norm=PowerNorm(gamma=.55,vmin=0,vmax=vmax),interpolation='nearest')
        axes[i,0].set_title(f'Input irradiance\n{case["input_power_W"]:.4f} W')
        fig.colorbar(image,ax=axes[i,0],shrink=.72,label='W/m²')
        image=axes[i,2].imshow(oi,origin='lower',extent=extent,cmap='inferno',
            norm=PowerNorm(gamma=.55,vmin=0,vmax=vmax),interpolation='nearest')
        axes[i,2].set_title(f'Output irradiance\n{case["output_power_W"]:.4f} W')
        fig.colorbar(image,ax=axes[i,2],shrink=.72,label='W/m²')
        axes[i,1].plot(grid.x*1e3,ii[y_index],color='tab:blue',linewidth=1.8)
        axes[i,1].set_title('Input side profile')
        axes[i,1].set_ylabel('W/m²')
        axes[i,3].plot(grid.x*1e3,oi[y_index],color='tab:orange',linewidth=1.8)
        axes[i,3].set_title('Output side profile')
        axes[i,3].set_ylabel('W/m²')
        for col,(field,title) in enumerate(((inp,'Input phase'),(out,'Output phase')),4):
            image=axes[i,col].imshow(_relative_phase(field),origin='lower',extent=extent,
                 cmap=phase_cmap,vmin=-np.pi,vmax=np.pi,interpolation='nearest')
            axes[i,col].set_title(title+' (rad)')
            fig.colorbar(image,ax=axes[i,col],shrink=.72,ticks=[-np.pi,0,np.pi])
        axes[i,0].set_ylabel(f'{name}\ny (mm)\nDisk gain {case["disk_power_gain"]:.4f}×')
        for ax in axes[i]:
            ax.set_xlabel('x (mm)')
            ax.set_xlim(-2,2)
        for col in (0,2,4,5):
            axes[i,col].set_ylim(-2,2)
        for col in (1,3):
            axes[i,col].set_ylim(bottom=0)
    mode=result['settings'].solver_mode
    description=('fixed seeded modes with periodic pump/population, heat, bonded thermoelastic, '
                 'and photoelastic closure' if mode=='full_seeded_modal' else
                 'frozen Stage 7W inversion; weak-signal probe')
    fig.suptitle(f'Ideal phase mask: {result["settings"].phase_mask_name} | '
                 'structured seed → one Ho:YAG traversal → '
                 f'{result["settings"].post_disk_distance_m:g} m free space\n{description}',fontsize=14)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=150)
    plt.close(fig)


def draw_side_profiles(result, path):
    """Plot centerline irradiance cuts for every input and propagated output."""
    grid=result['grid']
    y_index=int(np.argmin(abs(grid.y)))
    x_mm=grid.x*1e3
    rows=list(result['outcomes'].items())
    fig,axes=plt.subplots(3,2,figsize=(12,12),sharex=True,constrained_layout=True)
    axes=np.asarray(axes).ravel()
    for ax,(name,case) in zip(axes,rows):
        input_profile=case.get('input_intensity',abs(case['input_field'])**2)[y_index]
        output_profile=case.get('output_intensity',abs(case['output_field'])**2)[y_index]
        ax.plot(x_mm,input_profile,label='Input',linewidth=1.8)
        ax.plot(x_mm,output_profile,label='Output',linewidth=1.8)
        ax.set_title(name)
        ax.set_xlabel('x at y≈0 (mm)')
        ax.set_ylabel('Irradiance (W/m²)')
        ax.grid(alpha=.25)
        ax.legend(frameon=False)
        ax.set_xlim(-2,2)
    fig.suptitle(f'Centerline side profiles after {result["settings"].post_disk_distance_m:g} m free space\n'
                 'Input and output irradiance at y≈0',fontsize=14)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160)
    plt.close(fig)


def draw_beam_on_density(result, path):
    """Overlay each incoming irradiance footprint on the crystal entrance Ho map."""
    grid=result['grid']
    extent=[grid.x[0]*1e3,grid.x[-1]*1e3,grid.y[0]*1e3,grid.y[-1]*1e3]
    density=result['density'].values_m3/1e26
    entrance=np.ma.masked_where(density[0] <= 0,density[0])
    rows=list(result['outcomes'].items())
    fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    axes=np.asarray(axes).ravel()
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#e6e8ec')
    low=float(entrance.min()); high=float(entrance.max())
    for ax,(name,case) in zip(axes,rows):
        background=ax.imshow(entrance,origin='lower',extent=extent,cmap=cmap,
                            vmin=low,vmax=high,interpolation='nearest')
        irradiance=case.get('input_intensity',abs(case['input_field'])**2)
        normalized=irradiance/max(float(irradiance.max()),1e-30)
        levels=(.1,.3,.6,.9)
        ax.contour(grid.x*1e3,grid.y*1e3,normalized,levels=levels,
                   colors='white',linewidths=(.8,1.0,1.2,1.5))
        ax.set_title(name)
        ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)')
        ax.set_xlim(-2,2);ax.set_ylim(-2,2);ax.set_aspect('equal')
        ax.text(.03,.04,'white contours: 10–90% input Imax',transform=ax.transAxes,
                color='white',fontsize=7,ha='left',va='bottom',
                bbox={'facecolor':'black','alpha':.35,'pad':2,'edgecolor':'none'})
    fig.colorbar(background,ax=axes.tolist(),shrink=.86,
                 label='Ho concentration at entrance (10²⁶ ions/m³)')
    fig.suptitle('Incoming beam footprints on the Ho:YAG crystal entrance slice\n'
                 'Background: generated concentration; white contours: calculated input irradiance',
                 fontsize=14)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160)
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
    y_index=int(np.argmin(abs(grid.y)))
    image=axes[2].imshow(data[:,y_index,:],origin='lower',aspect='auto',
        extent=[grid.x[0]*1e3,grid.x[-1]*1e3,z_edges[0]*1e3,z_edges[-1]*1e3],
        cmap=cmap,vmin=low,vmax=high)
    axes[2].set_title(f'x–z cut at y={grid.y[y_index]*1e3:.2f} mm')
    axes[2].set_xlabel('x (mm)');axes[2].set_ylabel('depth z (mm)')
    fig.colorbar(image,ax=axes,label='Ho density (10²⁶ ions/m³)',shrink=.82)
    fig.suptitle(f'Seed {result["settings"].density_seed}: random Ho-rich and Ho-poor clusters; '
                 'zero outside the 10-mm disk')
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160)
    plt.close(fig)


def draw_phase_mask(result,path):
    grid=result['grid'];extent=[grid.x[0]*1e3,grid.x[-1]*1e3,
                                 grid.y[0]*1e3,grid.y[-1]*1e3]
    gaussian=result['outcomes']['Gaussian TEM00']
    before=abs(gaussian['seed_before_slm'])**2
    after=abs(gaussian['input_field'])**2
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,data,title in ((axes[0],result['phase_applied_rad'],'Applied phase (rad)'),
                          (axes[1],before,'Gaussian before SLM (W/m²)'),
                          (axes[2],after,'Gaussian after SLM (W/m²)')):
        image=ax.imshow(data,origin='lower',extent=extent,
            cmap='twilight' if ax is axes[0] else 'inferno',
            vmin=0,vmax=2*np.pi if ax is axes[0] else float(before.max()))
        ax.set_title(title);ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)')
        ax.set_xlim(-1.5,1.5);ax.set_ylim(-1.5,1.5)
        fig.colorbar(image,ax=ax,shrink=.78)
    fig.suptitle(f'{result["settings"].phase_mask_name}: ideal phase-only mask; '
                 'irradiance is unchanged immediately across the SLM')
    fig.savefig(path,dpi=150)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-directory',type=Path,
                   default=ROOT/'results/local_stage7w/reference')
    p.add_argument('--output-directory',type=Path,
                   default=ROOT/'results/structured_beams')
    p.add_argument('--density-seed',type=int,default=17)
    p.add_argument('--cluster-count',type=int,default=24)
    p.add_argument('--cluster-contrast',type=float,default=.27)
    p.add_argument('--cluster-min-radius-mm',type=float,default=.20)
    p.add_argument('--cluster-max-radius-mm',type=float,default=1.25)
    p.add_argument('--phase-mask',choices=PHASE_MASKS,default='none')
    p.add_argument('--phase-strength-rad',type=float,default=float(np.pi),
                   help='radial phase scale for defocus, astigmatic, and axicon masks')
    p.add_argument('--post-disk-distance-m',type=float,default=.25)
    p.add_argument('--solver-mode',choices=SOLVER_MODES,default='weak_probe',
                   help='weak_probe is fast; full_seeded_modal recomputes pump, saturation, heat, mechanics, and photoelasticity for the selected Ho map')
    p.add_argument('--plots-only',action='store_true',
                   help='save plots and summary without archiving complex field arrays')
    args=p.parse_args()
    snapshot=snapshot_from_case(args.case_directory)
    settings=GallerySettings(density_seed=args.density_seed,
        cluster_count=args.cluster_count,cluster_contrast=args.cluster_contrast,
        cluster_radius_min_m=args.cluster_min_radius_mm*1e-3,
        cluster_radius_max_m=args.cluster_max_radius_mm*1e-3,
        phase_mask_name=args.phase_mask,phase_strength_rad=args.phase_strength_rad,
        post_disk_distance_m=args.post_disk_distance_m,solver_mode=args.solver_mode)
    result=simulate_gallery(snapshot,settings)
    result['z_edges_m']=snapshot.arrays['z_edges_m']
    output=args.output_directory
    output.mkdir(parents=True,exist_ok=True)
    draw_beams(result,output/'input_output_beams.png')
    draw_side_profiles(result,output/'beam_side_profiles.png')
    draw_beam_on_density(result,output/'beam_on_ho_density.png')
    draw_density(result,output/'ho_density.png')
    draw_phase_mask(result,output/'phase_mask.png')
    archive=output/'fields.npz'
    if args.plots_only:
        archive.unlink(missing_ok=True)
        fields_sha=None
    else:
        arrays={'x_m':result['grid'].x,'y_m':result['grid'].y,
                'z_edges_m':result['z_edges_m'],
                'ho_density_m3':result['density'].values_m3,
                'phi_pattern_rad':result['phase_pattern_rad'],
                'phi_correction_rad':result['phase_correction_rad'],
                'phi_requested_rad':result['phase_requested_rad'],
                'phi_applied_rad':result['phase_applied_rad']}
        for index,(name,case) in enumerate(result['outcomes'].items()):
            arrays[f'mode_{index}_seed_before_slm']=case['seed_before_slm']
            arrays[f'mode_{index}_input_field']=case['input_field']
            arrays[f'mode_{index}_output_field']=case['output_field']
        np.savez_compressed(archive,**arrays)
        fields_sha=sha256_file(archive)
    reference_summary=json.loads((args.case_directory/'summary.json').read_text())
    reference_path=args.case_directory.resolve()
    try:
        reference_label=str(reference_path.relative_to(ROOT))
    except ValueError:
        reference_label=str(reference_path)
    excluded=('seed_before_slm','input_field','output_field','output_vector',
              'input_intensity','output_intensity')
    if settings.solver_mode=='full_seeded_modal':
        limitations=['fixed externally seeded transverse basis; cavity eigenfield is not updated',
                     'photoelastic coefficients use the audited host-YAG reference',
                     'single disk traversal plus the requested free-space output plane']
    else:
        limitations=['imposed density is not pump/heat/mechanics self-consistent',
                     'weak-signal probe; no population depletion by these inputs',
                     'single disk traversal; no specified multipass hardware topology']
    summary={'model':result['model'],
             'application_mode':'seeded_multipass_amplifier',
             'fidelity_mode':settings.solver_mode,
             'solver_mode':settings.solver_mode,
             'solver_diagnostics':result['solver_diagnostics'],
             'source_hash':source_manifest(ROOT)['source_hash'],
             'reference_state_id':result['reference_state_id'],
             'reference_state_sha256':reference_summary['state_sha256'],
             'reference_case':reference_label,
             'fields_sha256':fields_sha,'fields_archived':not args.plots_only,
             'wavelength_m':result['wavelength_m'],
             'settings':asdict(result['settings']),
             'density_mean_active_m3':float(result['density'].values_m3[result['density'].values_m3>0].mean()),
             'density_min_active_m3':float(result['density'].values_m3[result['density'].values_m3>0].min()),
             'density_max_active_m3':float(result['density'].values_m3.max()),
             'phase_mask':{'name':settings.phase_mask_name,
                           'strength_rad':settings.phase_strength_rad,
                           'response':'ideal phase-only; no measured hardware latency or loss',
                           'maximum_local_irradiance_relative_change':max(
                               float(np.max(abs(abs(case['input_field'])**2-abs(case['seed_before_slm'])**2)) /
                                     max(np.max(abs(case['seed_before_slm'])**2),1e-30))
                               for case in result['outcomes'].values())},
             'modes':{name:{key:value for key,value in case.items() if key not in excluded}
                      for name,case in result['outcomes'].items()},
             'time_kind':'steady_state',
             'limitations':limitations}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'output_directory':str(output.resolve()),
                      'density_range_m3':[summary['density_min_active_m3'],summary['density_max_active_m3']],
                      'power_gain_by_mode':{name:case['disk_power_gain'] for name,case in result['outcomes'].items()}}))


if __name__=='__main__':main()
