"""Local interactive Ho:YAG structured-beam calculation app.

Start from the repository root:

    .venv/bin/python examples/structured_beam_app.py

Then open http://127.0.0.1:8780/results/structured_beams/index.html.
The Calculate action runs the selected numerical model through the bounded
local supervisor and returns a new set of solver-generated plots.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import mimetypes
import shutil
import sys
import time
from urllib.parse import urlparse
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from hoyag.local_supervisor import BudgetLedger, Limits, run_bounded
from hoyag.structured_beam_gallery import PHASE_MASKS, SOLVER_MODES, BEAM_NAMES


def _number(payload, name, low, high, integer=False, default=None):
    value = payload.get(name,default)
    if isinstance(value, bool):
        raise ValueError(f'{name} must be numeric')
    try:
        parsed_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    if not math.isfinite(parsed_float):
        raise ValueError(f'{name} must be finite')
    if integer:
        if not parsed_float.is_integer():
            raise ValueError(f'{name} must be an integer')
        parsed = int(parsed_float)
    else:
        parsed = parsed_float
    if parsed < low or parsed > high:
        raise ValueError(f'{name} must be between {low} and {high}')
    return parsed


def validate_request(payload):
    if not isinstance(payload, dict):
        raise ValueError('request must be a JSON object')
    mask = payload.get('phase_mask', 'none')
    if mask not in PHASE_MASKS:
        raise ValueError(f'phase_mask must be one of {PHASE_MASKS}')
    solver_mode = payload.get('solver_mode', 'weak_probe')
    if solver_mode not in SOLVER_MODES:
        raise ValueError(f'solver_mode must be one of {SOLVER_MODES}')
    selected_beam=payload.get('selected_beam','Gaussian TEM00')
    if selected_beam not in BEAM_NAMES:
        raise ValueError('invalid selected_beam')
    values = {
        'density_seed': _number(payload, 'density_seed', -2_000_000_000, 2_000_000_000, True),
        'cluster_count': _number(payload, 'cluster_count', 2, 64, True),
        'cluster_contrast': _number(payload, 'cluster_contrast', 0, 1, False),
        'cluster_min_radius_mm': _number(payload, 'cluster_min_radius_mm', .05, 2.5),
        'cluster_max_radius_mm': _number(payload, 'cluster_max_radius_mm', .05, 2.5),
        'phase_strength_rad': _number(payload, 'phase_strength_rad', -50, 50),
        'post_disk_distance_m': _number(payload, 'post_disk_distance_m', 0, 2),
        'seed_energy_nj':_number(payload,'seed_energy_nj',.001,100000,default=10),
        'seed_fwhm_ps':_number(payload,'seed_fwhm_ps',.1,1000,default=10),
        'signal_traversals':_number(payload,'signal_traversals',1,20,True,default=10),
        'relay_distance_m':_number(payload,'relay_distance_m',0,2,default=0),
    }
    if values['cluster_min_radius_mm'] > values['cluster_max_radius_mm']:
        raise ValueError('cluster_min_radius_mm cannot exceed cluster_max_radius_mm')
    values['phase_mask'] = mask
    values['solver_mode'] = solver_mode
    values['selected_beam']=selected_beam
    for key in ('dn_dHo_m3','dn_dExcited_m3'):
        raw=payload.get(key)
        values[key]=None if raw in (None,'') else _number(payload,key,-1e-24,1e-24)
    provenance=payload.get('index_provenance','')
    if not isinstance(provenance,str) or len(provenance)>500:
        raise ValueError('index provenance must be text under 500 characters')
    if any(values[key] is not None for key in ('dn_dHo_m3','dn_dExcited_m3')) and not provenance.strip():
        raise ValueError('measured index coefficients require source or measurement provenance')
    values['index_provenance']=provenance
    return values


def build_gallery_command(values, output_directory):
    relative = output_directory.resolve().relative_to(ROOT)
    command=[
        sys.executable, 'examples/structured_beam_gallery.py',
        '--output-directory', str(relative),
        '--density-seed', str(values['density_seed']),
        '--cluster-count', str(values['cluster_count']),
        '--cluster-contrast', str(values['cluster_contrast']),
        '--cluster-min-radius-mm', str(values['cluster_min_radius_mm']),
        '--cluster-max-radius-mm', str(values['cluster_max_radius_mm']),
        '--phase-mask', values['phase_mask'],
        '--phase-strength-rad', str(values['phase_strength_rad']),
        '--post-disk-distance-m', str(values['post_disk_distance_m']),
        '--solver-mode', values['solver_mode'],
        '--selected-beam',values['selected_beam'],
        '--seed-energy-nj',str(values['seed_energy_nj']),
        '--seed-fwhm-ps',str(values['seed_fwhm_ps']),
        '--signal-traversals',str(values['signal_traversals']),
        '--relay-distance-m',str(values['relay_distance_m']),
    ]
    if values['solver_mode']!='periodic_seeded_amplifier':
        command.append('--plots-only')
    if values['dn_dHo_m3'] is not None:
        command.extend(('--dn-dho-m3',str(values['dn_dHo_m3'])))
    if values['dn_dExcited_m3'] is not None:
        command.extend(('--dn-dexcited-m3',str(values['dn_dExcited_m3'])))
    if values['index_provenance']:
        command.extend(('--index-provenance',values['index_provenance']))
    return command


def budget_status():
    ledger = BudgetLedger(ROOT / '.local_runtime' / 'budget.json', Limits())
    data = ledger._read()
    used_seconds = sum(float(row['elapsed_s']) for row in data['attempts'])
    coupled = sum(row.get('category', 'coupled') == 'coupled' for row in data['attempts'])
    return {
        'remaining_seconds': max(0., ledger.limits.total_seconds-used_seconds),
        'coupled_attempts_used': coupled,
        'coupled_attempts_limit': ledger.limits.max_attempts,
        'coupled_exhausted': (used_seconds >= ledger.limits.total_seconds or
                              coupled >= ledger.limits.max_attempts),
        'active': data['active'] is not None,
    }


def start_new_budget():
    """Archive the exhausted global ledger after an explicit UI action."""
    ledger = BudgetLedger(ROOT / '.local_runtime' / 'budget.json', Limits())
    ledger._acquire()
    try:
        data = ledger._read()
        if data['active'] is not None:
            raise RuntimeError('a supervised calculation is still active')
        used_seconds = sum(float(row['elapsed_s']) for row in data['attempts'])
        coupled = sum(row.get('category', 'coupled') == 'coupled' for row in data['attempts'])
        if used_seconds < ledger.limits.total_seconds and coupled < ledger.limits.max_attempts:
            raise RuntimeError('the coupled budget is still available')
        archive = ledger.path.with_name(
            f'budget.archived.{time.strftime("%Y%m%d_%H%M%S")}.{uuid.uuid4().hex[:8]}.json')
        shutil.copy2(ledger.path, archive)
        from hoyag.local_supervisor import atomic_json
        atomic_json(ledger.path, {'schema': 1, 'total_seconds': ledger.limits.total_seconds,
                                  'max_attempts': ledger.limits.max_attempts,
                                  'attempts': [], 'active': None})
        return {'archive': str(archive.resolve()), 'status': budget_status()}
    finally:
        ledger.release()


def run_calculation(values):
    run_id = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    output = ROOT / 'results' / 'structured_beams' / 'runs' / run_id
    output.mkdir(parents=True, exist_ok=False)
    execution = output / 'execution.json'
    full_solver = values['solver_mode'] in ('full_seeded_modal','periodic_seeded_amplifier')
    try:
        result = run_bounded(
            build_gallery_command(values, output), cwd=ROOT,
            log_path=output / 'execution.log', summary_path=execution,
            ledger=BudgetLedger(ROOT / '.local_runtime' / 'budget.json', Limits()),
            label=f'interactive_gallery_{run_id}',
            configured_seconds=900 if full_solver else 180,
            category='coupled' if full_solver else 'profile')
    except Exception:
        # A reservation can fail before the worker creates any files. Do not
        # leave an empty run directory for a rejected calculation.
        if output.is_dir() and not any(output.iterdir()):
            output.rmdir()
        raise
    if result['status'] != 'completed' or result.get('exit_code') != 0:
        raise RuntimeError(json.dumps(result))
    summary = json.loads((output / 'summary.json').read_text())
    return result_for_run(output, summary, result)


def result_for_run(output, summary, execution):
    relative = output.relative_to(ROOT).as_posix()
    return {
        'run_id': output.name,
        'execution': execution,
        'summary': summary,
        'images': {
            key: f'/{relative}/{filename}'
            for key, filename in {
                'beams': 'input_output_beams.png',
                'profiles': 'beam_side_profiles.png',
                'beam_density': 'beam_on_ho_density.png',
                'density': 'ho_density.png',
                'phase': 'phase_mask.png',
            }.items()
        },
    }


def latest_completed_run():
    runs=ROOT/'results'/'structured_beams'/'runs'
    for directory in sorted(runs.iterdir(),key=lambda p:p.name,reverse=True) if runs.is_dir() else ():
        summary_path=directory/'summary.json'
        execution_path=directory/'execution.json'
        if directory.is_dir() and summary_path.is_file() and execution_path.is_file():
            execution=json.loads(execution_path.read_text())
            if execution.get('status')=='completed' and execution.get('exit_code')==0:
                return result_for_run(directory,json.loads(summary_path.read_text()),execution)
    return None


class AppHandler(BaseHTTPRequestHandler):
    server_version = 'HoYAGStructuredBeamApp/1.0'

    def _send_json(self, status, value):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        endpoint = urlparse(self.path).path
        if endpoint not in ('/calculate', '/budget/new'):
            self._send_json(404, {'error': 'unknown endpoint'})
            return
        try:
            if endpoint == '/budget/new':
                if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    raise ValueError('JSON content type required')
                length = int(self.headers.get('Content-Length', '0'))
                if length <= 0 or length > 1000:
                    raise ValueError('invalid request body')
                if json.loads(self.rfile.read(length)) != {'action': 'start_new_budget'}:
                    raise ValueError('explicit budget action required')
                self._send_json(200, start_new_budget())
                return
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0 or length > 32_000:
                raise ValueError('request body must be 1–32000 bytes')
            raw = self.rfile.read(length)
            content_type = self.headers.get('Content-Type', '')
            if content_type.startswith('application/json'):
                payload = json.loads(raw)
            else:
                from urllib.parse import parse_qs
                payload = {key: values[-1] for key, values in parse_qs(raw.decode()).items()}
            values = validate_request(payload)
            self._send_json(200, run_calculation(values))
        except Exception as exc:
            code = 'budget_exhausted' if str(exc) == 'budget_exhausted' else 'calculation_error'
            self._send_json(429 if code == 'budget_exhausted' else 400,
                            {'error': str(exc), 'code': code})

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/budget/status':
            self._send_json(200, budget_status())
            return
        if path == '/runs/latest':
            result=latest_completed_run()
            self._send_json(200 if result else 404,result or {'error':'no completed run'})
            return
        if path == '/':
            path = '/results/structured_beams/index.html'
        target = (ROOT / path.lstrip('/')).resolve()
        try:
            target.relative_to(ROOT)
        except ValueError:
            self.send_error(403)
            return
        if target.is_dir():
            target = target / 'index.html'
        if not target.is_file():
            self.send_error(404)
            return
        data = target.read_bytes()
        content_type = mimetypes.guess_type(str(target))[0] or 'application/octet-stream'
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        sys.stderr.write('[structured-beam-app] ' + format % args + '\n')


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8780)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f'Ho:YAG calculation app: http://{args.host}:{args.port}/results/structured_beams/index.html')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
