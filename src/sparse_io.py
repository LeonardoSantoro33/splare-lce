"""
sparse_io.py — Sparse vectors: CSR structure, reading/writing Seismic's binary format,
conversion of the Seismic ids to SAE latent indices.

SEISMIC FORMAT (documents.bin / queries.bin), all little-endian:
    u32  N                       number of vectors
    then, for each vector:
    u32  n                       number of non-zero entries
    n x u32                      component ids
    n x f32                      values (SPLARE weights)
The ids are NOT the SAE latent indices: they are positions in token_to_id_mapping.json, which maps
string -> id with the strings sorted ALPHABETICALLY. For SPLARE the strings are the SAE latent indices
("0", "1", "10", "100", ...), so the id must be remapped to int(string) in 0 .. 131,071.

CSR ("compressed sparse row") is the standard way to keep many sparse vectors
in memory: three long arrays instead of millions of dictionaries.
    indptr  (N+1,)   row i occupies positions [indptr[i], indptr[i+1])
    indices (nnz,)   column (= SAE latent) of each entry
    data    (nnz,)   value of each entry
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from utils import log


@dataclass
class CSR:
    indptr: np.ndarray            # int64, length n_rows + 1
    indices: np.ndarray           # int32, column ids
    data: np.ndarray              # float32, values
    n_cols: int                   # dimension of the space (131072)
    ids: Optional[np.ndarray] = None  # row identifiers (document pids / query qids)

    # ---------------- basic information ----------------
    @property
    def n_rows(self):
        return len(self.indptr) - 1

    @property
    def nnz(self):
        return int(self.indptr[-1])

    def row_lengths(self):
        return np.diff(self.indptr)

    def row(self, i):
        a, b = int(self.indptr[i]), int(self.indptr[i + 1])
        return self.indices[a:b], self.data[a:b]

    # ---------------- row selection ----------------
    def gather(self, rows):
        """Concatenate the requested rows. Returns (offsets, indices, data):
        row j of the result starts at offsets[j]. This is exactly the format
        expected by torch.nn.functional.embedding_bag."""
        rows = np.asarray(rows, dtype=np.int64)
        starts = self.indptr[rows]
        lens = self.indptr[rows + 1] - starts
        offsets = np.zeros(len(rows), dtype=np.int64)
        if len(rows) > 1:
            np.cumsum(lens[:-1], out=offsets[1:])
        total = int(lens.sum())
        # vectorized trick: source position = row_start + (position_in_result - row_offset)
        pos = np.repeat(starts - offsets, lens) + np.arange(total, dtype=np.int64)
        return offsets, self.indices[pos], self.data[pos]

    def subset(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        offsets, idx, val = self.gather(rows)
        indptr = np.append(offsets, len(idx)).astype(np.int64)
        ids = None if self.ids is None else np.asarray(self.ids)[rows]
        return CSR(indptr, np.ascontiguousarray(idx), np.ascontiguousarray(val), self.n_cols, ids)

    def row_ids_per_element(self):
        """For each non-zero entry, the index of its row."""
        return np.repeat(np.arange(self.n_rows, dtype=np.int64), self.row_lengths())

    def sorted_indices(self):
        """Copy with the ids sorted within each row (Seismic requires them in increasing order)."""
        rows = self.row_ids_per_element()
        order = np.lexsort((self.indices, rows))
        return CSR(self.indptr.copy(), self.indices[order], self.data[order], self.n_cols, self.ids)

    def to_dense(self):
        out = np.zeros((self.n_rows, self.n_cols), dtype=np.float32)
        out[self.row_ids_per_element(), self.indices] = self.data
        return out

    @staticmethod
    def from_dense(mat, ids=None):
        mat = np.asarray(mat, dtype=np.float32)
        rows, cols = np.nonzero(mat)
        indptr = np.zeros(mat.shape[0] + 1, dtype=np.int64)
        np.cumsum(np.bincount(rows, minlength=mat.shape[0]), out=indptr[1:])
        return CSR(indptr, cols.astype(np.int32), mat[rows, cols].astype(np.float32), mat.shape[1], ids)

    @staticmethod
    def concat(parts):
        """Stack several CSRs (same columns) on top of each other."""
        n_cols = parts[0].n_cols
        indptrs, offset = [np.zeros(1, dtype=np.int64)], 0
        for p in parts:
            indptrs.append(p.indptr[1:] + offset)
            offset += p.nnz
        ids = None
        if all(p.ids is not None for p in parts):
            ids = np.concatenate([np.asarray(p.ids) for p in parts])
        return CSR(np.concatenate(indptrs), np.concatenate([p.indices for p in parts]).astype(np.int32),
                   np.concatenate([p.data for p in parts]).astype(np.float32), n_cols, ids)

    # ---------------- statistics ----------------
    def stats(self):
        lens = self.row_lengths()
        d = self.data
        sample = d[: min(len(d), 5_000_000)]
        return {
            "n_rows": int(self.n_rows),
            "n_cols": int(self.n_cols),
            "nnz_total": int(self.nnz),
            "nnz_per_row_mean": float(lens.mean()) if len(lens) else 0.0,
            "nnz_per_row_min": int(lens.min()) if len(lens) else 0,
            "nnz_per_row_max": int(lens.max()) if len(lens) else 0,
            "value_min": float(sample.min()) if len(sample) else 0.0,
            "value_max": float(sample.max()) if len(sample) else 0.0,
            "value_mean": float(sample.mean()) if len(sample) else 0.0,
            "frac_integer_values": float(np.mean(sample == np.round(sample))) if len(sample) else 0.0,
        }

    # ---------------- saving to disk ----------------
    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "indptr.npy", np.asarray(self.indptr, dtype=np.int64))   # asarray: no copy
        np.save(folder / "indices.npy", np.asarray(self.indices, dtype=np.int32))
        np.save(folder / "data.npy", np.asarray(self.data, dtype=np.float32))
        if self.ids is not None:
            np.save(folder / "ids.npy", np.asarray(self.ids))
        (folder / "meta.json").write_text(json.dumps({"n_cols": int(self.n_cols), **self.stats()}, indent=2))

    @staticmethod
    def load(folder, mmap=False):
        folder = Path(folder)
        if not (folder / "meta.json").exists():
            raise SystemExit(f"[ERROR] Cache not found: {folder}. Did you run the script that creates it?")
        mode = "r" if mmap else None
        meta = json.loads((folder / "meta.json").read_text())
        ids = np.load(folder / "ids.npy", allow_pickle=True) if (folder / "ids.npy").exists() else None
        return CSR(np.load(folder / "indptr.npy", mmap_mode=mode),
                   np.load(folder / "indices.npy", mmap_mode=mode),
                   np.load(folder / "data.npy", mmap_mode=mode),
                   int(meta["n_cols"]), ids)


# ======================================================================
#  Seismic binary format
# ======================================================================
def read_seismic_bin(path, max_vectors=None, chunk_rows=500_000, verbose=True):
    """Read documents.bin / queries.bin into a CSR (ids still in the Seismic space).

    Pass 1: scan the file to read the lengths (this needs a loop, because each
            vector only tells its length once you reach it).
    Pass 2: from the lengths, compute where ids and values are and copy them in chunks.
    """
    path = Path(path)
    raw = np.memmap(path, dtype="<u4", mode="r")
    n_total = int(raw[0])
    n = n_total if not max_vectors else min(n_total, int(max_vectors))
    if verbose:
        log(f"{path.name}: {n_total:,} vectors in the file ({path.stat().st_size / 1e9:.2f} GB); reading {n:,}")

    lens = np.empty(n, dtype=np.int64)
    pos = 1
    step = max(1, n // 10)
    for i in range(n):
        ln = int(raw[pos])
        lens[i] = ln
        pos += 1 + 2 * ln
        if verbose and (i + 1) % step == 0:
            log(f"  pass 1 (lengths): {i + 1:,}/{n:,}")
    if n == n_total and pos != raw.size:
        raise ValueError(f"{path}: the file does not match the expected format "
                         f"(data ends at word {pos}, file has {raw.size} words)")

    header_pos = np.empty(n, dtype=np.int64)
    header_pos[0] = 1
    if n > 1:
        header_pos[1:] = 1 + np.cumsum(1 + 2 * lens[:-1])
    id_start = header_pos + 1
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lens, out=indptr[1:])
    nnz = int(indptr[-1])
    indices = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float32)

    for a in range(0, n, chunk_rows):
        b = min(n, a + chunk_rows)
        la = lens[a:b]
        offs = indptr[a:b] - indptr[a]
        tot = int(indptr[b] - indptr[a])
        pos_ids = np.repeat(id_start[a:b] - offs, la) + np.arange(tot, dtype=np.int64)
        pos_vals = pos_ids + np.repeat(la, la)
        indices[indptr[a]:indptr[b]] = np.asarray(raw[pos_ids]).astype(np.int32)
        data[indptr[a]:indptr[b]] = np.asarray(raw[pos_vals]).view("<f4")
        if verbose and (b // chunk_rows) % 4 == 0:
            log(f"  pass 2 (copy): {b:,}/{n:,}")
    del raw
    n_cols = int(indices.max()) + 1 if nnz else 0
    return CSR(indptr, indices, data, n_cols)


def write_seismic_bin(path, csr):
    """Write a CSR in Seismic's binary format (ids sorted within each row)."""
    csr = csr.sorted_indices()
    n, nnz = csr.n_rows, csr.nnz
    lens = csr.row_lengths()
    out = np.empty(1 + n + 2 * nnz, dtype="<u4")
    out[0] = n
    header_pos = np.empty(n, dtype=np.int64)
    if n:
        header_pos[0] = 1
        header_pos[1:] = 1 + np.cumsum(1 + 2 * lens[:-1])
        out[header_pos] = lens
        pos_ids = np.repeat(header_pos + 1 - csr.indptr[:-1], lens) + np.arange(nnz, dtype=np.int64)
        out[pos_ids] = csr.indices.astype("<u4")
        out[pos_ids + np.repeat(lens, lens)] = csr.data.astype("<f4").view("<u4")
    out.tofile(path)


# ======================================================================
#  Mapping: Seismic id -> SAE latent index
# ======================================================================
def load_token_mapping(path):
    """Return the list of keys: tokens[seismic_id] = key string (an SAE latent index as a string)."""
    m = json.loads(Path(path).read_text())
    if isinstance(m, list):
        return list(m)
    if not isinstance(m, dict) or not m:
        raise ValueError(f"{path}: unrecognized token mapping format")
    k0, v0 = next(iter(m.items()))
    if isinstance(v0, int):                      # token -> id  (Seismic format)
        tokens = [None] * (max(m.values()) + 1)
        for tok, i in m.items():
            tokens[i] = tok
    elif isinstance(v0, str) and str(k0).isdigit():  # id -> token
        tokens = [None] * (max(int(k) for k in m) + 1)
        for i, tok in m.items():
            tokens[int(i)] = tok
    else:
        raise ValueError(f"{path}: unrecognized token mapping format")
    missing = sum(t is None for t in tokens)
    if missing:
        log(f"WARNING: {missing} ids without a token in the mapping (they will be ignored)")
    return tokens


def build_numeric_remap(tokens):
    """remap[seismic_id] = int(key), the SAE latent index (the keys are numbers written as strings)."""
    remap = np.full(len(tokens), -1, dtype=np.int64)
    bad = []
    for i, tok in enumerate(tokens):
        if tok is None:
            continue
        try:
            remap[i] = int(tok)
        except ValueError:
            bad.append(tok)
    if bad:
        raise ValueError(f"{len(bad)} tokens of the mapping are not integers (e.g. {bad[:5]}): not an SAE mapping")
    if len(np.unique(remap[remap >= 0])) != int((remap >= 0).sum()):
        raise ValueError("two entries of the mapping point to the same SAE latent")
    return remap


def remap_columns(csr, remap, n_cols):
    """Remap the column ids with remap[...]; entries with remap = -1 are dropped."""
    new_idx = np.asarray(remap, dtype=np.int32)[csr.indices]      # int32: 4 bytes per entry
    keep = new_idx >= 0
    if keep.all():
        return CSR(csr.indptr, new_idx, csr.data, n_cols, csr.ids)
    rows = csr.row_ids_per_element()[keep]
    indptr = np.zeros(csr.n_rows + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=csr.n_rows), out=indptr[1:])
    return CSR(indptr, new_idx[keep], csr.data[keep], n_cols, csr.ids)


def load_ids(path):
    """Load doc_ids.npy / queries_ids.npy and, if possible, convert them to integers."""
    arr = np.load(path, allow_pickle=True)
    try:
        return arr.astype(np.int64)
    except (ValueError, TypeError):
        log(f"WARNING: the ids in {Path(path).name} are not numeric: keeping them as strings")
        return arr.astype(str)


def read_groundtruth(path, query_ids=None, doc_ids=None):
    """Seismic's groundtruth[_100].tsv: columns (query_index, doc_index, rank, score) WITHOUT a header.
    The indices are POSITIONS in the .bin files; with query_ids/doc_ids they are converted to the original qid/pid."""
    import pandas as pd
    df = pd.read_csv(path, sep="\t", header=None, names=["q", "d", "rank", "score"])
    if query_ids is not None:
        df["qid"] = np.asarray(query_ids)[df["q"].to_numpy()]
    if doc_ids is not None:
        df["pid"] = np.asarray(doc_ids)[df["d"].to_numpy()]
    return df


def ids_to_rows(ids):
    """Build the inverse function id -> row from an array of ids (e.g. pids)."""
    ids = np.asarray(ids)
    if ids.dtype.kind in "iu" and len(ids) and ids.min() >= 0 and ids.max() < 4 * len(ids) + 1_000_000:
        table = np.full(int(ids.max()) + 1, -1, dtype=np.int64)
        table[ids] = np.arange(len(ids), dtype=np.int64)
        return ArrayLookup(table)
    return DictLookup({v: i for i, v in enumerate(ids.tolist())})


class ArrayLookup:
    def __init__(self, table):
        self.table = table

    def __call__(self, keys):
        keys = np.asarray(keys, dtype=np.int64)
        out = np.full(keys.shape, -1, dtype=np.int64)
        ok = (keys >= 0) & (keys < len(self.table))
        out[ok] = self.table[keys[ok]]
        return out


class DictLookup:
    def __init__(self, d):
        self.d = d

    def __call__(self, keys):
        keys = np.asarray(keys)
        return np.array([self.d.get(k, -1) for k in keys.ravel().tolist()], dtype=np.int64).reshape(keys.shape)
