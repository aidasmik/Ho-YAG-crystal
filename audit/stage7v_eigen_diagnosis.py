"""Measure branch/polarization changes in actual archived Stage 7V states.

This diagnostic reconstructs a frozen hot operator; it does NOT update the laser
populations, certify a nonconverged state, or change the production eigensolver.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from hoyag.validation_backend import make_grid, make_mesh, assembly_config, initial_modes
from hoyag.coupled_resonator import PlateAssembly
from hoyag.resonator import ThinDiskResonator
from hoyag.populations import HoYAGFourLevelParams
from hoyag.vector_cavity import (VectorRoundTrip, PlaneExchange, normalize_vector,
                                solve_vector_eigenfields, aligned_distance)


def polarization(field):
    f=normalize_vector(field).reshape(2,-1)
    coherence=f@f.conj().T
    # S3 convention stated here: +2 Im(<Ex Ey*>).
    return {'power_x':float(coherence[0,0].real),'power_y':float(coherence[1,1].real),
            'stokes_S1':float((coherence[0,0]-coherence[1,1]).real),
            'stokes_S2':float(2*coherence[0,1].real),
            'stokes_S3':float(2*coherence[0,1].imag)}


def field_comparison(a,b):
    a,b=normalize_vector(a),normalize_vector(b)
    ia=np.sum(abs(a)**2,axis=0);ib=np.sum(abs(b)**2,axis=0)
    cross=b.reshape(2,-1)@a.reshape(2,-1).conj().T
    singular=np.linalg.svd(cross,compute_uv=False)
    return {'vector_overlap_squared':float(abs(np.vdot(a,b))**2),
            'phase_aligned_field_distance':aligned_distance(a,b),
            'polarization_summed_intensity_relative_L2':float(np.linalg.norm(ia-ib)/np.linalg.norm(ia)),
            'overlap_squared_after_best_constant_unitary_polarization':float(min(1.,singular.sum()**2)),
            'note':'Unitary-polarization alignment is a diagnostic ONLY; it is not used to waive the vector convergence test.'}


def member(z,suffix):
    matches=[name for name in z.namelist() if name.endswith(suffix)]
    if len(matches)!=1:raise ValueError(f'Expected one archive member {suffix}: {matches}')
    return matches[0]


def inspect(z,case_id,output):
    summary_name=member(z,f'coupled/{case_id}/summary.json')
    source_name=member(z,f'coupled/{case_id}/source_manifest.json')
    state_name=member(z,f'coupled/{case_id}/state.npz')
    summary=json.loads(z.read(summary_name));manifest=json.loads(z.read(source_name))
    root=Path(__file__).resolve().parents[1]
    numerical_files=['coupled_resonator.py','vector_cavity.py','resonator.py','populations.py',
                     'thermal.py','thermomechanics.py','cooling_plate.py','stress_optics.py']
    for filename in numerical_files:
        path='src/hoyag/'+filename
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=manifest['files'][path]:
            raise ValueError(f'Numerical core differs from the archived case: {path}')
    state_bytes=z.read(state_name)
    if hashlib.sha256(state_bytes).hexdigest()!=summary['state_sha256']:
        raise ValueError('Archived state checksum mismatch')
    with np.load(io.BytesIO(state_bytes),allow_pickle=False) as arrays:
        state={k:arrays[k].copy() for k in arrays.files}
    case=summary['case'];grid,mesh=make_grid(case),make_mesh(case)
    c=ThinDiskResonator(**case['physics']['cavity'])
    start=time.perf_counter()
    assembly=PlateAssembly(mesh,grid,assembly_config(case))
    temperature,displacement,screens=assembly.solve(state['assembly_heat_W_m3'])
    assembly_seconds=time.perf_counter()-start
    p=HoYAGFourLevelParams(N_total_m3=case['physics']['density']['mean_m3'])
    fractions=state['mean_fractions'].reshape(4,*mesh.shape)
    gain=(p.sigma_em_laser_m2*fractions[2]-p.sigma_abs_laser_m2*fractions[3])*state['density_m3']
    exchange=PlaneExchange(grid,mesh,order=case['numerics']['settings']['projection_order'])
    log_gain=exchange.surface_on_grid(np.sum(gain*np.diff(mesh.z_edges_m)[:,None,None],axis=0))
    op=VectorRoundTrip(grid,c,screens,log_gain)
    previous=state['fields_used']
    predicted=state['fields_predicted']
    # Keep four candidate eigenfields here solely for diagnosing the archived
    # fixed operator. The original case retained its declared number of modes.
    seeds=initial_modes(grid,c,dict(case['numerics'],mode_count=4,initial_guess='gaussian'))
    start=time.perf_counter()
    eig=solve_vector_eigenfields(op,seeds,candidates=6,tolerance=2e-7,maxiter=1000)
    eigen_seconds=time.perf_counter()-start
    order=np.argsort(-abs(eig.eigenvalues))
    fields=eig.fields[order];values=eig.eigenvalues[order]
    eigenrecords=[]
    for field,value,index in zip(fields,values,order):
        eigenrecords.append({'eigenvalue_real':float(value.real),'eigenvalue_imag':float(value.imag),
             'log_roundtrip_power_growth':float(2*np.log(abs(value))),
             'residual':float(eig.residuals[index]),'polarization':polarization(field),
             'overlap_squared_with_used':[float(abs(np.vdot(normalize_vector(a),field))**2) for a in previous]})
    output.mkdir(parents=True,exist_ok=True)
    data={'case':case_id,'archived_status':summary['status'],
          'source_state_sha256':summary['state_sha256'],'used_polarizations':[polarization(a) for a in previous],
          'predicted_polarizations':[polarization(a) for a in predicted],
          'used_to_predicted':[field_comparison(a,b) for a,b in zip(previous,predicted)],
          'leading_eigenpair_comparison':field_comparison(fields[0],fields[1]),
          'leading_complex_eigenvalue_separation':float(abs(values[0]-values[1])),
          'leading_log_power_growth_separation':float(2*np.log(abs(values[0])/abs(values[1]))),
          'existing_cluster_tolerance':2e-7,'eigen_converged':bool(eig.converged),
          'eigen_status':eig.status,'eigenfields':eigenrecords,
          'assembly_seconds':assembly_seconds,'eigen_seconds':eigen_seconds,
          'scope':'Frozen-operator diagnosis of actual saved states; not new lasing predictions.'}
    (output/f'{case_id}.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print('EIGEN_DIAGNOSIS '+json.dumps(data,allow_nan=False),flush=True)
    return data


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    with zipfile.ZipFile(args.archive) as z:
        for case in ('sensitivity_support_free','modes_2'):
            inspect(z,case,args.output)


if __name__=='__main__':main()
