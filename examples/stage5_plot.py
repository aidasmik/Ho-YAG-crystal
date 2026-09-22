"""Plot saved Stage 5 numerical results; each file contains one chart.
Requires pip install -e '.[plots]'. No simulations are silently substituted.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from hoyag.thermal import DiskThermalMesh,DiskHeatSolver,DiskCooling,ThermalBoundary
from hoyag.thermal_optics import polar_to_cartesian,thermal_cavity_roundtrip,normalized_overlap
from hoyag.resonator import ThinDiskResonator,lg0_field,cavity_roundtrip


def make_plots(result_directory):
    root=Path(result_directory);summary=json.loads((root/'summary.json').read_text())
    with np.load(root/'state.npz') as source: data={k:source[k] for k in source.files}
    out=root/'figures';out.mkdir(parents=True,exist_ok=True);files=[]
    def save(name):
        plt.tight_layout()
        for ext in ('png','svg'):
            p=out/f'{name}.{ext}';plt.savefig(p,dpi=180);files.append(p)
        plt.close()
    mesh=DiskThermalMesh(data['r_edges_m'],data['z_edges_m'],summary['nphi'])
    thermal=summary['configuration']['thermal'];Tb=thermal['reference_temperature_K']
    q=data['heat_W_m3'];T=data['temperature_K'];opd=data['single_pass_opd_m']
    plt.figure(figsize=(7.2,4.8))
    plt.pcolormesh(mesh.r_edges_m*1e3,mesh.z_edges_m*1e3,q[:,:,0]/1e6,shading='flat')
    plt.colorbar(label='Cycle-averaged heat source (MW/m³)')
    plt.xlim(0,1.5);plt.gca().invert_yaxis()
    plt.xlabel('Radius r (mm)');plt.ylabel('Depth from front face z (mm)')
    plt.title('Ho:YAG disk: population-derived heat deposition')
    save('heat_source')
    plt.figure(figsize=(7.2,4.8))
    plt.pcolormesh(mesh.r_edges_m*1e3,mesh.z_edges_m*1e3,T[:,:,0]-Tb,shading='flat')
    plt.colorbar(label='Temperature rise above heat sink (K)')
    plt.xlim(0,1.5);plt.gca().invert_yaxis()
    plt.xlabel('Radius r (mm)');plt.ylabel('Depth from front face z (mm)')
    plt.title('10 W pump: rear-cooled 1 mm Ho:YAG disk')
    save('temperature')
    fit=summary['lens'];rr=np.linspace(0,fit['fit_radius_m'],200)
    fitted=fit['piston_m']-.5*fit['single_pass_power_m1']*rr**2
    plt.figure(figsize=(7,4.8))
    plt.plot(mesh.r_m*1e3,opd[:,0]*1e9,label='Calculated single-pass OPD')
    plt.plot(rr*1e3,fitted*1e9,'--',label='Weighted parabolic fit')
    plt.xlim(0,1.2);plt.xlabel('Radius r (mm)');plt.ylabel('Single-pass OPD (nm)')
    plt.title('Thermo-refractive optical-path distortion');plt.legend();save('radial_opd')
    n=512;size=.012;d=size/n;x=(np.arange(n)-(n-1)/2)*d
    g=SimpleNamespace(nx=n,ny=n,dx=d,dy=d,shape=(n,n),mesh=np.meshgrid(x,x,indexing='xy'),fx=np.fft.fftfreq(n,d),fy=np.fft.fftfreq(n,d))
    c=ThinDiskResonator(**summary['configuration']['cavity'])
    opxy=polar_to_cartesian(mesh,opd,g)
    plt.figure(figsize=(6.2,5))
    plt.imshow(4*np.pi*opxy/c.wavelength_m,extent=[-6,6,-6,6],origin='lower')
    plt.colorbar(label='Double-pass thermal phase (rad)')
    plt.xlim(-1.5,1.5);plt.ylim(-1.5,1.5)
    plt.xlabel('x (mm)');plt.ylabel('y (mm)');plt.title('Thermal phase per cavity round trip')
    save('roundtrip_thermal_phase')
    cooling=DiskCooling(rear=ThermalBoundary(thermal['rear_bath_temperature_K'],thermal['rear_contact_conductance_W_m2K']),
        front=ThermalBoundary(Tb,thermal['front_conductance_W_m2K']),rim=ThermalBoundary(Tb,thermal['rim_conductance_W_m2K']))
    solver=DiskHeatSolver(mesh,thermal['conductivity_W_mK'],cooling,
        density_kg_m3=thermal['mass_density_kg_m3'],heat_capacity_J_kgK=thermal['heat_capacity_J_kgK'])
    times=np.linspace(0,.5,101);t=np.full(mesh.shape,Tb);warm=[[0.,Tb]]
    for a,b in zip(times[:-1],times[1:]):
        step=solver.advance(t,q,b-a);t=step.temperature_K;warm.append([b,float(t.max())])
    warm=np.array(warm);np.savetxt(root/'warmup.csv',warm,delimiter=',',header='time_s,peak_cell_temperature_K',comments='')
    plt.figure(figsize=(7,4.8))
    plt.plot(warm[:,0]*1e3,warm[:,1]-Tb,label='Backward-Euler transient')
    plt.axhline(T.max()-Tb,linestyle='--',label='Steady solution')
    plt.xlabel('Time (ms)');plt.ylabel('Peak temperature rise (K)')
    plt.title('Thermal response to a fixed cycle-averaged heat source');plt.legend();save('thermal_warmup')
    rows=[]
    for h in (1e3,1e4,1e5,1e6,np.inf):
        case=DiskHeatSolver(mesh,thermal['conductivity_W_mK'],DiskCooling(rear=ThermalBoundary(Tb,h))).steady(q)
        rows.append([h,float(case.temperature_K.max()),case.relative_balance_error])
    rows=np.array(rows);np.savetxt(root/'contact_sensitivity.csv',rows,delimiter=',',header='rear_h_W_m2K,peak_cell_temperature_K,relative_heat_balance_error',comments='')
    plt.figure(figsize=(7,4.8))
    plt.semilogx(rows[:-1,0],rows[:-1,1]-Tb,'o-',label='Finite rear thermal contact')
    plt.axhline(rows[-1,1]-Tb,linestyle='--',label='Ideal fixed rear temperature')
    plt.xlabel('Rear contact conductance (W/m²/K)');plt.ylabel('Peak temperature rise (K)')
    plt.title('Cooling uncertainty: same deposited-heat map');plt.legend();save('cooling_sensitivity')
    vm=DiskThermalMesh.disk(3,64);Q=1e8;k=14.;h=1e5;L=.001
    numerical=DiskHeatSolver(vm,k,DiskCooling(rear=ThermalBoundary(Tb,h))).steady(Q)
    z=np.linspace(0,L,300);analytical=Tb+Q*L/h+Q*(L*L-z*z)/(2*k)
    plt.figure(figsize=(7,4.8))
    plt.plot(z*1e3,analytical-Tb,label='Analytical uniform-heating solution')
    plt.plot(vm.z_m*1e3,numerical.temperature_K[:,0,0]-Tb,'o',markevery=4,label='Finite volume')
    plt.xlabel('Depth from front face (mm)');plt.ylabel('Temperature rise (K)')
    plt.title('Thermal solver validation: finite rear contact');plt.legend();save('validation_1d')
    X,Y=g.mesh;test_mode=lg0_field(X,Y,c.waist_m,1)
    cold,_=cavity_roundtrip(test_mode,g,c)
    hot,_=thermal_cavity_roundtrip(test_mode,g,c,opxy)
    theta=np.linspace(0,2*np.pi,1024,endpoint=False);R=c.waist_m/np.sqrt(2)
    coords=[R*np.sin(theta)/d+(n-1)/2,R*np.cos(theta)/d+(n-1)/2]
    values=map_coordinates(hot.real,coords,order=1)+1j*map_coordinates(hot.imag,coords,order=1)
    charge=float(np.angle(np.roll(values,-1)*values.conj()).sum()/(2*np.pi))
    diagnostics={'interpretation':'weak LG1 diagnostic through Gaussian-heated cavity; not vortex-laser selection',
        'LG1_cold_vs_hot_roundtrip_overlap':normalized_overlap(cold,hot),'LG1_closed_loop_winding':charge,
        'cold_roundtrip_power_retention':float(np.sum(abs(cold)**2)/np.sum(abs(test_mode)**2)),
        'hot_roundtrip_power_retention':float(np.sum(abs(hot)**2)/np.sum(abs(test_mode)**2))}
    (root/'optical_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    return files

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,default=Path('results/stage5/generated'))
    a=p.parse_args()
    for f in make_plots(a.result): print(f)
