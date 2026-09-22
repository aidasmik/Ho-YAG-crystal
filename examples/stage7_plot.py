"""Plot only saved Stage 7 data. Nonconverged runs remain visibly labeled."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib.pyplot as plt


def make_plots(result):
    result=Path(result);s=json.loads((result/'summary.json').read_text())
    a=np.load(result/'state.npz',allow_pickle=False);out=result/'figures';out.mkdir(exist_ok=True)
    suffix='' if s['converged'] else ' (NOT CONVERGED)'
    paths=[]
    def save(name):
        plt.tight_layout()
        for ext in ('png','svg'):
            path=out/f'{name}.{ext}';plt.savefig(path,dpi=180);paths.append(path)
        plt.close()
    h=s['history']
    if h:
        plt.figure(figsize=(7,4.8));iterations=[r['iteration'] for r in h]
        for key,label in [('field_residual','Vector field'),('heat_residual','Unrelaxed heat')]:
            values=[max(r[key],1e-15) if r[key] is not None else np.nan for r in h]
            plt.semilogy(iterations,values,'o-',label=label)
        plt.xlabel('Outer iteration');plt.ylabel('Relative fixed-point residual')
        plt.title('Hot-cavity convergence'+suffix);plt.legend();save('convergence')
        plt.figure(figsize=(7,4.8));plt.plot(iterations,[r['output_W'] for r in h],'o-')
        plt.xlabel('Outer iteration');plt.ylabel('Cycle-averaged output power (W)')
        plt.title('Coupled oscillator output'+suffix);save('output_power')
    x=a['x_m']*1e3;y=a['y_m']*1e3;field=a['fields_used'][0]
    intensity=np.sum(abs(field)**2,axis=0);intensity/=intensity.max()
    plt.figure(figsize=(6.4,5));plt.pcolormesh(x,y,intensity,shading='nearest')
    plt.colorbar(label='Normalized intensity');plt.xlim(-1.2,1.2);plt.ylim(-1.2,1.2)
    plt.gca().set_aspect('equal');plt.xlabel('x at disk (mm)');plt.ylabel('y at disk (mm)')
    plt.title('Solved vector field: branch 0'+suffix);save('mode_intensity')
    if 'mean_roundtrip_opd_m' in a:
        plt.figure(figsize=(6.4,5));plt.pcolormesh(x,y,a['mean_roundtrip_opd_m']*1e9,shading='nearest')
        plt.colorbar(label='Mean round-trip OPD (nm)');plt.xlim(-1.2,1.2);plt.ylim(-1.2,1.2)
        plt.gca().set_aspect('equal');plt.xlabel('x (mm)');plt.ylabel('y (mm)')
        plt.title('Thermal + deformation + photoelastic phase'+suffix);save('hot_disk_opd')
    a.close();return paths

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,default=Path('results/stage7/generated'))
    for path in make_plots(p.parse_args().result):print(path)
