"""Immutable scientific state handed from solvers to replay/live renderers."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from threading import Lock
import time
from types import MappingProxyType

import numpy as np

from .validation_backend import sha256_file
from .local_supervisor import atomic_json


SCHEMA_VERSION = 1


def _freeze(value):
    if isinstance(value,dict):
        return MappingProxyType({key:_freeze(item) for key,item in value.items()})
    if isinstance(value,list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value,MappingProxyType):
        return {key:_thaw(item) for key,item in value.items()}
    if isinstance(value,tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class ScientificSnapshot:
    metadata: dict
    arrays: dict

    def __post_init__(self):
        frozen = {}
        for name, value in self.arrays.items():
            source = np.ascontiguousarray(value)
            arr = np.frombuffer(source.tobytes(),dtype=source.dtype).reshape(source.shape)
            if arr.dtype.kind in 'fc' and not np.all(np.isfinite(arr)):
                raise ValueError(f'nonfinite scientific array: {name}')
            frozen[name] = arr
        object.__setattr__(self, 'arrays', MappingProxyType(frozen))
        object.__setattr__(self, 'metadata', _freeze(json.loads(json.dumps(self.metadata))))
        required = ('schema_version','run_id','configuration_hash','source_hash','state_id',
                    'application_mode','fidelity_mode','solver_status','time_kind',
                    'optical_state_kind','coordinate_frames','units','approximation_flags')
        if any(self.metadata.get(key) is None for key in required):
            raise ValueError('snapshot metadata is incomplete')
        if self.metadata['time_kind'] not in ('steady_state','outer_iteration',
                                               'physical_transient','optical_pulse_window'):
            raise ValueError('invalid time kind')
        if self.metadata['optical_state_kind']=='incoherent_cavity_modes' and 'total_phase' in frozen:
            raise ValueError('incoherent modes cannot have a unique total phase')

    def array(self, name):
        return self.arrays.get(name)


def snapshot_from_case(directory: Path) -> ScientificSnapshot:
    """Read saved numerical arrays; missing quantities remain explicitly absent."""
    directory = Path(directory)
    summary = json.loads((directory/'summary.json').read_text())
    state_path = directory/'state.npz'
    if summary.get('status') != 'completed':
        raise ValueError('replay requires a completed numerical state')
    if sha256_file(state_path) != summary.get('state_sha256'):
        raise ValueError('numerical state checksum mismatch')
    with np.load(state_path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    if ('mean_fractions' in arrays and 'raw_heat_W_m3' in arrays and
            arrays['mean_fractions'].ndim == 3):
        arrays['mean_fractions'] = arrays['mean_fractions'].reshape(
            4,*arrays['raw_heat_W_m3'].shape)
    state_id = sha256((summary['state_sha256']+summary['execution_key']).encode()).hexdigest()[:16]
    metadata = {
        'schema_version': SCHEMA_VERSION, 'run_id': str(directory.resolve()),
        'configuration_hash': summary['spec_hash'], 'source_hash': summary['source_hash'],
        'state_id': state_id, 'application_mode': 'oscillator_reference',
        'fidelity_mode': 'replay', 'solver_status': summary.get('solver_status','completed'),
        'time_kind': 'steady_state', 't_sim_s': None,
        'outer_iteration': len(summary.get('history',[])), 'pulse_id': None,
        'averaging_interval': 'periodic optical cycle',
        't_published_wall': state_path.stat().st_mtime,
        'mesh_ids': {'optical': summary.get('spec_hash'), 'material': summary.get('spec_hash')},
        'coordinate_frames': {'optical':'laboratory x/y, metres',
                              'material':'cylindrical z/phi/r; z from optical front'},
        'units': {'fields_used':'arbitrary complex amplitude, unit discrete norm',
                  'output_coupler_fields':'complex amplitude, shape only',
                  'density_m3':'m^-3', 'mean_fractions':'fraction',
                  'raw_heat_W_m3':'W/m^3', 'disk_temperature_K':'K',
                  'plate_temperature_K':'K', 'disk_displacement_m':'m',
                  'plate_displacement_m':'m', 'mean_roundtrip_opd_m':'m'},
        'optical_state_kind': summary.get('field_semantics','incoherent_cavity_modes'),
        'field_normalization': 'shape-normalized; modal powers supplied separately',
        'wavelength_m': summary.get('case',{}).get('physics',{}).get('cavity',{}).get('wavelength_m'),
        'disk_conductivity_W_mK': summary.get('case',{}).get('physics',{}).get('assembly',{}).get('thermal',{}).get('disk',{}).get('conductivity_W_mK'),
        'modal_powers_W': summary.get('metrics',{}).get('mode_power_W'),
        'energy_metrics': summary.get('energy_budget'),
        'cooling_metrics': summary.get('cooling_metrics',
            {'interface_flux_available':'interface_flux_W_m2' in arrays,
             'finite_cooling_capacity':False}),
        'subsolver_timestamps': None, 'sensor_flags': {'experimental_sensors':False},
        'approximation_flags': ['adiabatic_cycle_averaged_modes',
            'incoherent_modal_mixture','replay_not_live','finite_candidate_bank'],
        'available_arrays': sorted(arrays),
    }
    return ScientificSnapshot(metadata, arrays)


class DisplayQueue:
    """Drop stale display states only; producer integration is unaffected."""
    def __init__(self, capacity=2):
        if capacity < 1:
            raise ValueError('display queue capacity must be positive')
        self._items = deque(maxlen=capacity)
        self._lock = Lock()

    def publish(self, snapshot: ScientificSnapshot):
        with self._lock:
            self._items.append(snapshot)

    def newest(self):
        with self._lock:
            return self._items[-1] if self._items else None

    def __len__(self):
        with self._lock:
            return len(self._items)


def save_snapshot(snapshot: ScientificSnapshot, directory: Path) -> Path:
    """Atomically publish array data before metadata announces an accepted state."""
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    stem=f"state_{snapshot.metadata['outer_iteration']:04d}"
    path=directory/(stem+'.npz')
    temporary=directory/(stem+f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**snapshot.arrays)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    metadata={**_thaw(snapshot.metadata),'array_file':path.name,
              'array_sha256':sha256_file(path)}
    atomic_json(directory/(stem+'.json'),metadata)
    atomic_json(directory/'latest.json',{'state_id':snapshot.metadata['state_id'],
        'record_file':stem+'.json'})
    return directory/(stem+'.json')


def load_snapshot(record_path: Path) -> ScientificSnapshot:
    record_path=Path(record_path)
    metadata=json.loads(record_path.read_text())
    array_file=record_path.parent/metadata.pop('array_file')
    digest=metadata.pop('array_sha256')
    if sha256_file(array_file)!=digest:
        raise ValueError('snapshot array checksum mismatch')
    with np.load(array_file,allow_pickle=False) as saved:
        arrays={key:saved[key] for key in saved.files}
    return ScientificSnapshot(metadata,arrays)


def newest_snapshot(directory: Path):
    latest=Path(directory)/'latest.json'
    if not latest.exists():
        return None
    record=json.loads(latest.read_text())
    return load_snapshot(latest.parent/record['record_file'])
