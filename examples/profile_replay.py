"""Bounded micro-benchmark of separate versus shared heat/population replay."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def worker(output):
    if os.environ.get('HOYAG_SUPERVISED')!='1':
        raise RuntimeError('profile worker requires the shared budget supervisor')
    import numpy as np
    import scipy
    import psutil
    from types import SimpleNamespace
    from hoyag.thermal import DiskThermalMesh
    from hoyag.resonator import ThinDiskResonator
    from hoyag.thermal_resonator import modal_laser_on_thermal_mesh,sample_cycle_heat
    from hoyag.coupled_resonator import time_averaged_populations
    from hoyag.validation_backend import source_manifest
    from hoyag.local_supervisor import atomic_json
    mesh=DiskThermalMesh.disk(4,2)
    model=modal_laser_on_thermal_mesh(mesh,ThinDiskResonator())
    optical=SimpleNamespace(periodic_converged=True,period_cycles=1,
        fractions_before_next_pump=np.zeros((3,model.nz,model.ns)),
        log_photon_number=np.zeros(model.nm))
    energy=1e-6
    old_times=[];new_times=[]
    differences=[]
    for _ in range(8):
        tick=time.perf_counter()
        old_heat=sample_cycle_heat(model,optical,energy)
        old_mean=time_averaged_populations(model,optical,energy,1e4)
        old_times.append(time.perf_counter()-tick)
        tick=time.perf_counter()
        new_heat,new_mean=sample_cycle_heat(model,optical,energy,return_mean_fractions=True)
        new_times.append(time.perf_counter()-tick)
        differences.append(float(np.max(abs(new_mean-old_mean))))
        if not np.array_equal(new_heat.heat_W_m3,old_heat.heat_W_m3):
            raise AssertionError('heat replay changed')
    try:
        from threadpoolctl import threadpool_info
        blas=threadpool_info()
    except ImportError:
        blas=None
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],
        capture_output=True,text=True,check=False) if shutil_which('nvidia-smi') else None
    record={'source_manifest':source_manifest(ROOT),'fixture':'4 radial x 2 depth cells; one optical period; synthetic marked-periodic state',
        'old_median_s':statistics.median(old_times),'new_median_s':statistics.median(new_times),
        'old_samples_s':old_times,'new_samples_s':new_times,
        'speedup':statistics.median(old_times)/statistics.median(new_times),
        'max_population_fraction_difference':max(differences),
        'heat_exactly_equal':True,
        'hardware':{'cpu':platform.processor(),'logical_cores':psutil.cpu_count(),
                    'ram_bytes':psutil.virtual_memory().total,
                    'gpu':gpu.stdout.strip() if gpu and gpu.returncode==0 else None,
                    'os':platform.platform(),'python':sys.version,
                    'numpy':np.__version__,'scipy':scipy.__version__,
                    'blas':blas,'thread_settings':{k:v for k,v in os.environ.items() if k.endswith('NUM_THREADS')}}}
    atomic_json(output,record)


def shutil_which(name):
    import shutil
    return shutil.which(name)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('run','worker'))
    p.add_argument('--output',type=Path,default=ROOT/'results/local_profile/replay.json')
    p.add_argument('--ledger',type=Path,default=ROOT/'.local_runtime/budget.json')
    args=p.parse_args()
    if args.command=='worker':
        worker(args.output)
        return
    from hoyag.local_supervisor import BudgetLedger,Limits,run_bounded
    out=args.output.resolve();out.parent.mkdir(parents=True,exist_ok=True)
    execution=run_bounded([sys.executable,str(Path(__file__).resolve()),'worker','--output',str(out)],
        cwd=ROOT,log_path=out.parent/'profile.log',
        summary_path=out.parent/'execution.json',
        ledger=BudgetLedger(args.ledger,Limits()),label='replay_micro_profile',
        configured_seconds=180,category='profile',env={**os.environ,'HOYAG_SUPERVISED':'1'})
    print(json.dumps(execution))
    if execution['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
