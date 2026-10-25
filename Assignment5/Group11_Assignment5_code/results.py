"""
results.py - JSON output for every part. Framework-free, so the torch parts
and the TensorFlow part write results the same way.

Each part writes <outdir>/<part>/results.json; numpy scalars/arrays, Paths
and tuples are converted on the way out.
"""

import json
import math
from pathlib import Path

import numpy as np


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.generic):
        return jsonable(o.item())
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, Path):
        return o.as_posix()
    return o


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=2), encoding="utf-8")
    return path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
