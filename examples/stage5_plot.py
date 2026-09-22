"""Plot saved Stage 5 numerical arrays; no simulation is rerun.

Run: python examples/stage5_plot.py results/stage5/generated
Plots use the cell-face extent, SI-to-display conversions, and a separate
figure for every quantity. Optical iteration number is NOT physical time.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def make_plots(folder: Path) -> list[Path]:
    folder=Path(folder)
    s=json.loads((folder/'summary.json').read_text())
    with np.load(folder/'fields.npz') as f:
        t=f['temperature_K'];q=f['source_W_m3'];opd=f['opd_m'];mask=f['mask']
        x=f['x_m'];y=f['y_m'];z=f['z_m']
    sink=s['configuration']['boundary']['sink_temperature_K']
    dx=x[1]-x[0];dy=y[1]-y[0]
    extent=[(x[0]-dx/2)*1e3,(x[-1]+dx/2)*1e3,
            (y[0]-dy/2)*1e3,(y[-1]+dy/2)*1e3]
    outputs=[];figures=folder/'figures';figures.mkdir(exist_ok=True)

    def save(name):
        plt.tight_layout()
        for ext in ('png','svg'):
            dest=figures/f'{name}.{ext}'
            plt.savefig(dest,dpi=180)
            outputs.append(dest)
        plt.close()

    plt.figure(figsize=(6.5,5.2))
    plt.imshow(np.where(mask[0],t[0]-sink,np.nan),origin='lower',extent=extent)
    plt.colorbar(label='Temperature rise (K)')
    plt.xlim(-1.6,1.6);plt.ylim(-1.6,1.6)
    plt.xlabel('x (mm)');plt.ylabel('y (mm)')
    plt.title('Ho:YAG temperature near the front face\n10 W incident pump; assumed rear cooling')
    save('front_temperature')

    mid=len(y)//2
    plt.figure(figsize=(7.5,4.8))
    plt.imshow(t[:,mid,:]-sink,origin='lower',aspect='auto',
               extent=[extent[0],extent[1],0,s['cavity']['disk_thickness_m']*1e3])
    plt.colorbar(label='Temperature rise (K)')
    plt.xlim(-2,2);plt.xlabel('x (mm)');plt.ylabel('Depth z from front face (mm)')
    plt.title('Temperature through the 1 mm disk\nRear HR / cooled face at z = 1 mm')
    save('temperature_cross_section')

    plt.figure(figsize=(6.5,5.2))
    plt.imshow(np.where(mask[0],opd*1e9,np.nan),origin='lower',extent=extent)
    plt.colorbar(label='Single-pass thermal OPD (nm)')
    plt.xlim(-1.6,1.6);plt.ylim(-1.6,1.6)
    plt.xlabel('x (mm)');plt.ylabel('y (mm)')
    plt.title('Thermo-optic path change relative to the sink\nNo bulging or photoelasticity')
    save('thermal_opd')

    plt.figure(figsize=(7.5,4.8))
    plt.plot(x*1e3,opd[mid]*1e9,label='Single disk traversal')
    plt.plot(x*1e3,2*opd[mid]*1e9,linestyle='--',label='Two traversals / round trip')
    plt.xlim(-1.6,1.6);plt.xlabel('x (mm)');plt.ylabel('Thermal OPD (nm)')
    plt.title('Single- and double-pass thermal phase path')
    plt.legend();save('thermal_opd_lineout')

    b=s['heat_budget_W']
    names=['Bulk heat','Escaping\nfluorescence','Output\ncoupler','Other cavity\noptical loss']
    values=[b['heat'],b['fluorescence_escape'],b['output_coupler'],
            b['all_cavity_optical_losses']-b['output_coupler']]
    plt.figure(figsize=(7.4,5.0))
    bars=plt.bar(names,values)
    plt.bar_label(bars,fmt='%.3f W',padding=3)
    plt.ylim(0,max(values)*1.22)
    plt.ylabel('Cycle-averaged power (W)')
    plt.title(f"Energy destinations: {b['absorbed_pump']:.3f} W absorbed pump\nStorage and numerical balance terms are retained in the data")
    save('energy_budget')

    with (folder/'outer_iterations.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    iteration=[int(r['iteration']) for r in rows]
    plt.figure(figsize=(7.5,4.8))
    plt.plot(iteration,[float(r['power_used_m1']) for r in rows],'o-',label='Lens used for optical solve')
    plt.plot(iteration,[float(r['power_target_m1']) for r in rows],'s--',label='Lens calculated from heat')
    plt.xlabel('Outer numerical iteration (not elapsed time)')
    plt.ylabel('Single-pass thermal-lens power (m⁻¹)')
    plt.title('Thermal / optical fixed-point convergence')
    plt.legend();save('feedback_convergence')

    refinement=folder/'thermal_refinement.json'
    if refinement.exists():
        r=json.loads(refinement.read_text())
        plt.figure(figsize=(7.5,4.8))
        plt.plot([a['grid'][1] for a in r],[a['max_rise_K'] for a in r],'o-')
        plt.xlabel('Transverse cells per axis (axial cells refined proportionally)')
        plt.ylabel('Peak temperature rise (K)')
        plt.title('Thermal grid refinement at the same optical heat source')
        save('thermal_refinement')
    return outputs


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    for dest in make_plots(p.parse_args().folder):print(dest)
