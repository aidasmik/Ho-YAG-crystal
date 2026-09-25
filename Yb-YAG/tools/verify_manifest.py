"""Verify source-file SHA-256 checksums; the manifest excludes itself/exports."""
from pathlib import Path
import hashlib

ROOT=Path(__file__).resolve().parents[1]

def verify():
    count=0
    for line in (ROOT/'MANIFEST.sha256').read_text().splitlines():
        digest,name=line.split('  ',1)
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Manifest contains an escaping path')
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f'Checksum mismatch: {name}')
        count+=1
    return count

if __name__=='__main__':
    print(f'{verify()} source-file checksums verified.')
