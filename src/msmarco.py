"""
msmarco.py — The raw-text side of MS MARCO dev small: download, query texts, relevance judgments (qrels).

The SPLARE vectors of the passages and of the 6,980 dev queries are provided pre-encoded; what is needed here:
  qrels.dev.small.tsv     which passages are relevant for each dev query (MRR@10, nDCG@10, recall)
  queries.dev.small.tsv   the query texts (only to print readable examples of the expansion)
Both are inside the official collectionandqueries.tar.gz (~1 GB).
"""
import gzip
import hashlib
import os
import tarfile
import time
import urllib.request
from pathlib import Path

from utils import log

BASE = "https://msmarco.z22.web.core.windows.net/msmarcoranking/"
# (url, expected md5) — md5 from the official ir_datasets list
COLLECTION_AND_QUERIES = (BASE + "collectionandqueries.tar.gz", "31644046b18952c1386cd4564ba2ae69")
# files we need from inside collectionandqueries.tar.gz
DEV_MEMBERS = ["queries.dev.small.tsv", "qrels.dev.small.tsv"]


# ----------------------------------------------------------------------
#  download
# ----------------------------------------------------------------------
def md5sum(path, block=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download(url, dest, expected_md5=None, retries=3):
    dest = Path(dest)
    if dest.exists() and (expected_md5 is None or md5sum(dest) == expected_md5):
        log(f"already present: {dest.name}")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            log(f"downloading {url} (attempt {attempt})")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research script)"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length", 0))
                done, last = 0, time.time()
                while True:
                    chunk = r.read(1 << 22)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if time.time() - last > 15:
                        pct = f"{100 * done / total:.1f}%" if total else ""
                        log(f"  {done / 1e6:.0f} MB {pct}")
                        last = time.time()
            if expected_md5 and md5sum(tmp) != expected_md5:
                raise IOError("md5 does not match the expected value (corrupted file?)")
            os.replace(tmp, dest)
            log(f"ok: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
            return dest
        except Exception as e:  # noqa: BLE001 - we want to retry on any network error
            log(f"  error: {e}")
            time.sleep(5 * attempt)
    raise SystemExit(f"[ERROR] Download failed: {url}\n"
                     "Try from another node, or download it manually and use --from-tar.")


def extract_members(tar_path, out_dir, members):
    """Extract only the requested files (flattening directories) from a .tar.gz."""
    out_dir = Path(out_dir)
    wanted = set(members)
    found = set()
    with tarfile.open(tar_path, "r:*") as tar:
        for m in tar:
            name = Path(m.name).name
            if m.isfile() and name in wanted:
                log(f"extracting {name} ({m.size / 1e6:.1f} MB)")
                src = tar.extractfile(m)
                with open(out_dir / name, "wb") as f:
                    while True:
                        chunk = src.read(1 << 24)
                        if not chunk:
                            break
                        f.write(chunk)
                found.add(name)
    missing = wanted - found
    if missing:
        raise SystemExit(f"[ERROR] Missing from the tar archive: {sorted(missing)}")


# ----------------------------------------------------------------------
#  reading queries / qrels
# ----------------------------------------------------------------------
def _open_text(path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def read_queries(path):
    """File 'qid<TAB>text' -> dict {qid(int): text}."""
    out = {}
    with _open_text(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            qid, text = line.split("\t", 1)
            out[int(qid)] = text
    return out


def read_qrels(path):
    """qrels in TREC format 'qid 0 pid rel' (tabs or spaces) -> dict {qid: {pid: rel}}."""
    out = {}
    with _open_text(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _, pid, rel = parts[:4]
            out.setdefault(int(qid), {})[int(pid)] = int(rel)
    return out


def load_dev_qrels(paths):
    if not paths.dev_qrels_tsv.exists():
        raise SystemExit(f"[ERROR] Missing {paths.dev_qrels_tsv}: run  python -u src/02_download_msmarco.py")
    return read_qrels(paths.dev_qrels_tsv)


def load_dev_texts(paths):
    """{qid: text}, or None if queries.dev.small.tsv is not there (texts are only used for the examples)."""
    return read_queries(paths.dev_queries_tsv) if paths.dev_queries_tsv.exists() else None
