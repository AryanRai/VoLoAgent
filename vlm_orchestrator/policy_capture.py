"""Opt-in, lossless numerical evidence. No pickle, model calls or robot changes."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=8)
def source_revision(directory):
    return subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()


def save_capture(root, capture_id, kind, payload):
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', capture_id + kind):
        raise ValueError('Invalid capture identifier')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    arrays = {}

    def encode(value):
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError('Object arrays cannot be captured')
            key = f'array_{len(arrays)}'
            arrays[key] = value.copy()
            return {'__array__': key}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): encode(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [encode(v) for v in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f'Unsupported capture value: {type(value)}')

    tree = encode(payload)
    stem = root / f'{capture_id}-{kind}'
    with stem.with_suffix('.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    digest = hashlib.sha256(stem.with_suffix('.npz').read_bytes()).hexdigest()
    manifest = {'capture_id': capture_id, 'kind': kind, 'sha256': digest,
                'arrays_file': stem.name + '.npz', 'payload': tree}
    with stem.with_suffix('.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, allow_nan=False)
    return {'capture_id': capture_id, 'manifest': str(stem.with_suffix('.json')), 'sha256': digest}


def load_capture(manifest_path):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding='utf-8'))
    array_path = path.parent / manifest['arrays_file']
    if array_path.parent.resolve() != path.parent.resolve():
        raise ValueError('Array file must be beside manifest')
    if hashlib.sha256(array_path.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('Capture SHA-256 mismatch')
    with np.load(array_path, allow_pickle=False) as arrays:
        def decode(value):
            if isinstance(value, dict):
                if set(value) == {'__array__'}:
                    return arrays[value['__array__']].copy()
                return {k: decode(v) for k, v in value.items()}
            if isinstance(value, list):
                return [decode(v) for v in value]
            return value
        return decode(manifest['payload'])
