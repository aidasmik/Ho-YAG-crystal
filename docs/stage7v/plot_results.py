"""Plot only completed Stage 7V evidence; never fill missing campaign points."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def make_plots(directory):
    directory=Path(directory)
    report=json.loads((directory/'qualification.json').read_text())
    plan=json.loads((directory/'plan.json').read_text())
    figures=directory/'figures';figures.mkdir(exist_ok=True)
    scope='Fixed heat and populations' if report['kind']=='frozen' else 'Coupled hot cavity'
    saved=[]
    def save(name):
        plt.tight_layout()
        for ext in ('png','svg'):
            path=figures/f'{name}.{ext}';plt.savefig(path,dpi=180);saved.append(path)
        plt.close()
    def record(cid):
        if report['case_status'].get(cid)!='completed':return None
        path=directory/cid/'summary.json'
        if not path.exists():return None
        item=json.loads(path.read_text())
        return item if item.get('status')=='completed' else None
    plt.figure(figsize=(8,5));found=False;maximum_pair=0
    names={'optical_resolution':'Optical grid','material_resolution':'Material/thermal grid',
           'mechanical_resolution':'Mechanical FEM','joint_refinement':'Joint refinement'}
    for group in report['groups']:
        if group['name'] not in names:continue
        points=[]
        for i,pair in enumerate(group.get('pairs',[])):
            value=pair.get('metrics',{}).get('opd',{}).get('piston_removed_difference_rms_m')
            if value is not None:points.append((i+1,value*1e9));maximum_pair=max(maximum_pair,i+1)
        if points:
            plt.plot(*zip(*points),marker='o',label=names[group['name']]);found=True
    if found:
        plt.axhline(report['acceptance']['opd_difference_rms_m']*1e9,linestyle='--',label='Acceptance threshold')
        ticks=range(1,maximum_pair+1);plt.xticks(list(ticks),[f'Level {i} to {i+1}' for i in ticks])
        plt.ylabel('Piston-removed OPD difference RMS (nm)');plt.title(scope+': spatial refinement')
        plt.legend();save('refinement_opd')
    else:plt.close()
    modal=[record('reference')]+[record(f'modes_{i}') for i in (2,4,8)]
    modal=[r for r in modal if r and r.get('metrics',{}).get('output_W') is not None]
    if len(modal)>1:
        plt.figure(figsize=(7,4.7))
        plt.plot([r['case']['numerics']['mode_count'] for r in modal],
                 [r['metrics']['output_W'] for r in modal],marker='o',linestyle='none')
        plt.xlabel('Retained cavity modes');plt.ylabel('Cycle-averaged output power (W)')
        plt.title('Completed coupled mode-count cases; unrun/nonconverged points omitted')
        save('mode_count_output')
    contact=[record('sensitivity_contact_0.5'),record('reference'),record('sensitivity_contact_2')]
    if all(contact):
        plt.figure(figsize=(7.4,4.7))
        h=[r['case']['physics']['assembly']['thermal']['interface_conductance_W_m2K'] for r in contact]
        plt.semilogx(h,[r['metrics']['peak_disk_K']-273.15 for r in contact],marker='o')
        plt.xlabel('Crystal-plate conductance (W m^-2 K^-1)');plt.ylabel('Peak crystal temperature (deg C)')
        plt.title(scope+': contact sensitivity');save('contact_sensitivity')
    candidates=['joint_3','joint_2','joint_1','mechanical_3','modes_2','reference']
    chosen=next((record(cid) for cid in candidates if record(cid) is not None),None)
    if chosen:
        with np.load(directory/chosen['id']/'state.npz',allow_pickle=False) as state:
            if 'mean_roundtrip_opd_m' in state:
                x,y=state['x_m'],state['y_m'];xx,yy=np.meshgrid(x,y,indexing='xy');r=np.hypot(xx,yy)
                domain=plan['comparison_domain'];w=np.exp(-2*(r/domain['weight_radius_m'])**2)*(r<=domain['radius_m'])
                opd=state['mean_roundtrip_opd_m'];piston=np.sum(w*opd)/w.sum()
                plt.figure(figsize=(6.4,5))
                image=plt.pcolormesh(x*1e3,y*1e3,np.ma.array((opd-piston)*1e9,mask=r>domain['radius_m']),shading='nearest')
                plt.colorbar(image,label='Piston-removed mean round-trip OPD (nm)')
                radius=domain['radius_m']*1e3;plt.xlim(-radius,radius);plt.ylim(-radius,radius)
                plt.gca().set_aspect('equal');plt.xlabel('x (mm)');plt.ylabel('y (mm)')
                plt.title(scope+': '+chosen['id']);save('mean_opd_map')
        probes=chosen.get('probes',[])
        if probes:
            plt.figure(figsize=(7.4,4.7))
            plt.bar(range(len(probes)),[100*(1-p['target_oam_fraction']) for p in probes])
            plt.xticks(range(len(probes)),[str(p['charge']) for p in probes])
            plt.xlabel('Injected LG azimuthal charge');plt.ylabel('Power outside injected charge (%)')
            plt.title('Weak double-pass probes, not free-running vortex selection');save('probe_oam_mixing')
    (figures/'scope.json').write_text(json.dumps({'campaign_status':report['status'],
        'kind':report['kind'],'dataset_ready':False,'figures':[p.name for p in saved],
        'note':'Only available completed evidence is plotted; missing points are not imputed.'},indent=2)+'\n')
    return saved

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--result',type=Path,required=True)
    for path in make_plots(parser.parse_args().result):print(path)
