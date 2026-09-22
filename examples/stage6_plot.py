"""Plot saved Stage 6 fields; no simulation data are invented by the plotter."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def make_plots(result):
    result=Path(result);out=result/'figures';out.mkdir(exist_ok=True)
    s=json.loads((result/'summary.json').read_text());a=np.load(result/'state.npz',allow_pickle=False)
    paths=[]
    def save(name):
        plt.tight_layout()
        for ext in ('png','svg'):
            p=out/f'{name}.{ext}';plt.savefig(p,dpi=180);paths.append(p)
        plt.close()
    dr=a['disk_r_edges_m'];pr=a['plate_r_edges_m'];dz=a['disk_z_edges_m'];pz=a['plate_z_edges_m']+dz[-1]
    tc=s['configuration']['thermal']['coolant_temperature_K']
    plt.figure(figsize=(7,4.8))
    maximum=max(a['disk_temperature_K'].max(),a['plate_temperature_K'].max())-tc
    plt.pcolormesh(dr*1e3,dz*1e3,(a['disk_temperature_K'][:,:,0]-tc),vmin=0,vmax=maximum,shading='flat')
    image=plt.pcolormesh(pr*1e3,pz*1e3,(a['plate_temperature_K'][:,:,0]-tc),vmin=0,vmax=maximum,shading='flat')
    plt.axhline(dz[-1]*1e3,linestyle='--',label='Crystal/plate bond')
    plt.colorbar(image,label='Temperature rise above coolant (K)')
    plt.gca().invert_yaxis();plt.xlabel('Radius (mm)');plt.ylabel('Depth from optical front (mm)')
    plt.title('Finite crystal and copper cooling plate');plt.legend();save('assembly_temperature')
    r=.5*(dr[:-1]+dr[1:]);plt.figure(figsize=(7,4.8))
    plt.plot(r*1e3,a['disk_contact_temperature_K'][:,0]-tc,label='Crystal rear face')
    plt.plot(r*1e3,a['plate_contact_temperature_K'][:,0]-tc,label='Plate contact face')
    plt.xlim(0,2);plt.xlabel('Radius (mm)');plt.ylabel('Interface temperature rise (K)')
    plt.title('Finite thermal contact resistance');plt.legend();save('contact_temperature_jump')
    x=a['x_m'];iy=np.argmin(abs(x));use=abs(x)<=s['roi_radius_m']
    plt.figure(figsize=(7,4.8))
    for key,label in [('front_uz_m','Crystal front'),('rear_uz_m','Crystal rear HR')]:
        plt.plot(x[use]*1e3,a[key][iy,use]*1e9,label=label)
    plt.xlabel('x near the beam centre (mm)');plt.ylabel('Axial displacement (nm)')
    plt.title('Crystal surfaces — positive z points into plate');plt.legend();save('surface_displacement')
    plt.figure(figsize=(7,4.8))
    maps=[(2*a['thermal_single_pass_opd_m'],'Thermo-refractive'),(a['geometry_roundtrip_opd_m'],'Surface geometry'),
          (2*a['photoelastic_mean_single_pass_opd_m'],'Mean photoelastic'),(a['mean_roundtrip_opd_m'],'Total mean')]
    for m,label in maps:
        line=m[iy,use];plt.plot(x[use]*1e3,(line-line[0])*1e9,label=label)
    plt.xlabel('x near the beam centre (mm)');plt.ylabel('Round-trip OPD relative to left endpoint (nm)')
    plt.title('Hot-disk optical-path contributions');plt.legend();save('opd_components')
    xx,yy=np.meshgrid(x,x,indexing='xy');mask=np.hypot(xx,yy)>s['roi_radius_m']
    j=a['outward_jones']@a['inward_jones'];cross=np.ma.array(100*abs(j[...,1,0])**2,mask=mask)
    plt.figure(figsize=(6.5,5));image=plt.pcolormesh(x*1e3,x*1e3,cross,shading='nearest')
    plt.colorbar(image,label='Local crossed-polarizer fraction (%)')
    plt.xlim(-1,1);plt.ylim(-1,1);plt.gca().set_aspect('equal');plt.xlabel('x (mm)');plt.ylabel('y (mm)')
    plt.title('Double-pass photoelastic polarization conversion');save('polarization_conversion')
    a.close();return paths

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--result',type=Path,default=Path('results/stage6/generated'))
    for f in make_plots(p.parse_args().result):print(f)
