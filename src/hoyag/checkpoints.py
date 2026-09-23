"""Exact-problem Stage 7W continuation checkpoints, never implicit warm starts."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

from .local_supervisor import atomic_json
from .validation_backend import sha256_file
from .validation_campaign import json_safe
from .validation_metrics import stable_hash


def mesh_fingerprint(case, arrays):
    h=sha256()
    for key in ('x_m','y_m','r_edges_m','z_edges_m'):
        h.update(key.encode())
        h.update(np.asarray(arrays[key]).tobytes())
    h.update(stable_hash(case['numerics']['mechanical']).encode())
    return h.hexdigest()


def save_checkpoint(directory, case, manifest, info, arrays):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    iteration=int(info['iteration'])
    stem=f'checkpoint_{iteration:04d}'
    state=directory/(stem+'.npz')
    temporary=directory/(stem+f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**{k:v for k,v in arrays.items() if v is not None})
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,state)
    record={'schema_version':1,'status':'accepted_outer_iteration',
        'application_mode':'oscillator_reference','time_kind':'outer_iteration',
        'physical_time_s':None,'iteration':iteration,'streak':int(info['streak']),
        'history':info['history'],'branch_ids':info['branch_ids'],
        'source_revision':manifest.get('git_revision'),
        'source_hash':manifest['source_hash'],
        'configuration_fingerprint':case['spec_hash'],
        'runtime_fingerprint':stable_hash({k:manifest[k] for k in ('python','numpy','scipy','platform')}),
        'mesh_fingerprint':mesh_fingerprint(case,arrays),
        'array_file':state.name,'array_sha256':sha256_file(state),
        'null_arrays':[key for key,value in arrays.items() if value is None]}
    atomic_json(directory/(stem+'.json'),json_safe(record))
    atomic_json(directory/'latest.json',{'record_file':stem+'.json','iteration':iteration})
    return directory/(stem+'.json')


def load_checkpoint(directory, case, manifest):
    directory=Path(directory)
    latest=directory/'latest.json'
    if not latest.exists():return None
    record=json.loads((directory/json.loads(latest.read_text())['record_file']).read_text())
    if record.get('schema_version')!=1:
        raise ValueError('unknown checkpoint schema')
    if record['source_hash']!=manifest['source_hash'] or record['configuration_fingerprint']!=case['spec_hash']:
        raise ValueError('incompatible checkpoint source or configuration; explicit warm start is required')
    runtime=stable_hash({k:manifest[k] for k in ('python','numpy','scipy','platform')})
    if record['runtime_fingerprint']!=runtime:
        raise ValueError('incompatible checkpoint numerical runtime')
    state=directory/record['array_file']
    if sha256_file(state)!=record['array_sha256']:
        raise ValueError('checkpoint checksum mismatch')
    with np.load(state,allow_pickle=False) as saved:
        arrays={k:saved[k] for k in saved.files}
    if mesh_fingerprint(case,arrays)!=record['mesh_fingerprint']:
        raise ValueError('checkpoint mesh fingerprint mismatch')
    for name in record['null_arrays']:
        arrays[name]=None
    arrays.update(iteration=record['iteration'],streak=record['streak'],
                  history=record['history'],branch_ids=record['branch_ids'])
    return arrays
