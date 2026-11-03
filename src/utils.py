"""
utils.py — Small shared helpers: timestamped logging, timer, device, json/csv.
"""
import csv
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def log(*args):
    """print() with a date/time prefix and immediate flush (so Slurm logs show everything right away)."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]", *args, flush=True)


@contextmanager
def timer(what):
    t0 = time.time()
    log(f"{what} ...")
    yield
    log(f"{what}: done in {time.time() - t0:.1f} s")


def get_device(prefer="cuda"):
    if prefer == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
        # TF32: much faster float32 matmuls on A100 (precision is more than sufficient here)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        dev = torch.device("cpu")
    n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0)
    if n_threads > 0:
        torch.set_num_threads(n_threads)
    return dev


def describe_device(dev):
    if dev.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        return f"GPU {torch.cuda.get_device_name(0)} ({free / 1e9:.1f} GB free of {total / 1e9:.1f} GB)"
    return f"CPU ({torch.get_num_threads()} threads)"


def log_run_header(script_name, args, dev):
    log("=" * 70)
    log(f"Script: {script_name}")
    log(f"Host: {os.uname().nodename}   Slurm job: {os.environ.get('SLURM_JOB_ID', '-')}")
    log(f"Python {sys.version.split()[0]}  torch {torch.__version__}  device: {describe_device(dev)}")
    log("Arguments: " + json.dumps(jsonable(vars(args)), ensure_ascii=False))
    log("=" * 70)


def jsonable(obj):
    """Convert Path/numpy objects to plain types (for json and for a safe torch.save)."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=2, ensure_ascii=False))


def load_json(path):
    return json.loads(Path(path).read_text())


def append_csv_row(path, row: dict):
    """Append a row to a CSV file (writing the header if the file does not exist).
    If the columns change, the new keys are appended at the end."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: jsonable(v) for k, v in row.items()}
    if path.exists():
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            old_rows = list(reader)
            fields = list(reader.fieldnames or [])
        new_fields = fields + [k for k in row if k not in fields]
        if new_fields != fields:
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=new_fields)
                w.writeheader()
                for r in old_rows:
                    w.writerow(r)
        fields = new_fields
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
    else:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerow(row)


def human_bytes(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")
