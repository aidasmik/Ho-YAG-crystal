"""Local interactive Ho:YAG structured-beam calculation app.

Start from the repository root:

    .venv/bin/python examples/structured_beam_app.py

Then open http://127.0.0.1:8780/results/structured_beams/index.html.
The Calculate action runs the existing numerical weak-probe model through the
bounded local supervisor and returns a new set of solver-generated plots.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import mimetypes
import sys
import time
from urllib.parse import urlparse
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from hoyag.local_supervisor import BudgetLedger, Limits, run_bounded
from hoyag.structured_beam_gallery import PHASE_MASKS


def _number(payload, name, low, high, integer=False):
    value = payload.get(name)
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
    values = {
        'density_seed': _number(payload, 'density_seed', -2_000_000_000, 2_000_000_000, True),
        'cluster_count': _number(payload, 'cluster_count', 2, 64, True),
        'cluster_contrast': _number(payload, 'cluster_contrast', 0, 1, False),
        'cluster_min_radius_mm': _number(payload, 'cluster_min_radius_mm', .05, 2.5),
        'cluster_max_radius_mm': _number(payload, 'cluster_max_radius_mm', .05, 2.5),
        'phase_strength_rad': _number(payload, 'phase_strength_rad', -50, 50),
        'post_disk_distance_m': _number(payload, 'post_disk_distance_m', 0, 2),
    }
    if values['cluster_min_radius_mm'] > values['cluster_max_radius_mm']:
        raise ValueError('cluster_min_radius_mm cannot exceed cluster_max_radius_mm')
    values['phase_mask'] = mask
    return values


def build_gallery_command(values, output_directory):
    relative = output_directory.resolve().relative_to(ROOT)
    return [
        sys.executable, 'examples/structured_beam_gallery.py',
        '--output-directory', str(relative), '--plots-only',
        '--density-seed', str(values['density_seed']),
        '--cluster-count', str(values['cluster_count']),
        '--cluster-contrast', str(values['cluster_contrast']),
        '--cluster-min-radius-mm', str(values['cluster_min_radius_mm']),
        '--cluster-max-radius-mm', str(values['cluster_max_radius_mm']),
        '--phase-mask', values['phase_mask'],
        '--phase-strength-rad', str(values['phase_strength_rad']),
        '--post-disk-distance-m', str(values['post_disk_distance_m']),
    ]


def run_calculation(values):
    run_id = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    output = ROOT / 'results' / 'structured_beams' / 'runs' / run_id
    output.mkdir(parents=True, exist_ok=False)
    execution = output / 'execution.json'
    result = run_bounded(
        build_gallery_command(values, output), cwd=ROOT,
        log_path=output / 'execution.log', summary_path=execution,
        ledger=BudgetLedger(ROOT / '.local_runtime' / 'budget.json', Limits()),
        label=f'interactive_gallery_{run_id}', configured_seconds=180,
        category='profile')
    if result['status'] != 'completed' or result.get('exit_code') != 0:
        raise RuntimeError(json.dumps(result))
    summary = json.loads((output / 'summary.json').read_text())
    relative = output.relative_to(ROOT).as_posix()
    return {
        'run_id': run_id,
        'execution': result,
        'summary': summary,
        'images': {
            key: f'/{relative}/{filename}'
            for key, filename in {
                'beams': 'input_output_beams.png',
                'profiles': 'beam_side_profiles.png',
                'density': 'ho_density.png',
                'phase': 'phase_mask.png',
            }.items()
        },
    }


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
        if urlparse(self.path).path != '/calculate':
            self._send_json(404, {'error': 'unknown endpoint'})
            return
        try:
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
            self._send_json(400, {'error': str(exc)})

    def do_GET(self):
        path = urlparse(self.path).path
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
