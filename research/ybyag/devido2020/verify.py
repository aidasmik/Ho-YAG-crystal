"""Verify unchanged source files and report raw-data quality; no data cleaning."""
import hashlib
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'provenance.json').read_text(encoding='utf8'))
total = invalid = above_one = 0
for entry in manifest['files']:
    path = root / entry['filename']
    content = path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == entry['sha256'], path
    rows = [[float(v) for v in line.split()] for line in content.decode().splitlines() if line.strip()]
    assert all(len(row) == 8 and all(math.isfinite(v) for v in row) for row in rows), path
    total += len(rows)
    invalid += sum(row[2] <= 0 or row[3] <= 0 for row in rows)
    above_one += sum(row[3] > 22.8 * row[2] for row in rows)
print(json.dumps({'files_verified': len(manifest['files']), 'rows': total,
                  'nonpositive_power_rows': invalid,
                  'transmission_above_one_rows': above_one}, indent=2))
