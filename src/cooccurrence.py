"""
cooccurrence.py — The Latent Co-occurrence Matrix M of Eq. (1), built EXACTLY over the whole corpus.

    M_ij = sum_{d in C} 1[V_d(i) > 0] * 1[V_d(j) > 0]      (i != j;  M_ii = 0)

M = B^T B, with B the binary (documents x features) matrix. Its diagonal is the document frequency df_i;
it is stored as computed and zeroed when M is used (lce.py), so df is not lost.

SIZE. Only the features active in at least one document are kept ("active" features, compact index):
  ~123.6k active SAE latents of 131,072  ->  M is 123.6k x 123.6k float32 = ~61 GB (on disk, memory-mapped).
M is far too large for the GPU, so it is built by blocks of ROWS: for each block F of R features,
    M[F, :] = sum over document chunks of  X_chunk[:, F]^T @ X_chunk
with X_chunk the dense binary (c x n_active) matrix of c documents. Matmuls of 0/1 numbers in float32
(TF32 on the A100 keeps 0 and 1 exact and accumulates in float32) give EXACT integer counts up to 2^24 = 16.7M,
more than the 8.84M documents of MS MARCO.

The job is resumable: finished row blocks are recorded in progress.json and skipped on restart.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from utils import human_bytes, load_json, log, save_json, timer

FILES = {"M": "M.npy", "active": "active.npy", "df": "df.npy", "meta": "meta.json", "progress": "progress.json"}


def doc_frequency(docs):
    """df[f] = number of documents with feature f > 0 (full feature space)."""
    df = np.zeros(docs.n_cols, dtype=np.int64)
    step = 200_000_000                                   # in chunks: bincount copies its input to int64
    for a in range(0, docs.nnz, step):
        b = min(docs.nnz, a + step)
        idx = np.asarray(docs.indices[a:b])
        val = np.asarray(docs.data[a:b])
        df += np.bincount(idx[val > 0], minlength=docs.n_cols)
    return df


def _chunk_to_device(docs, s, e, col_map, dev):
    """Rows s..e of the CSR as (row ids, compact column ids) on dev, only entries with value > 0."""
    a, b = int(docs.indptr[s]), int(docs.indptr[e])
    idx = np.asarray(docs.indices[a:b])
    val = np.asarray(docs.data[a:b])
    rows = np.repeat(np.arange(e - s, dtype=np.int64), np.diff(np.asarray(docs.indptr[s:e + 1])))
    keep = val > 0
    cols = col_map[idx[keep]]
    return torch.from_numpy(rows[keep]).to(dev), torch.from_numpy(cols.astype(np.int64)).to(dev)


def auto_blocks(n_active, dev, row_block=0, doc_chunk=0):
    """Choose block sizes that fit the free GPU memory (or small defaults on CPU)."""
    if dev.type != "cuda":
        return row_block or min(n_active, 4096), doc_chunk or 2048
    free, _ = torch.cuda.mem_get_info()
    budget = free * 0.60
    c = doc_chunk or 8192
    x_bytes = c * n_active * 4                           # X
    # out (R x n) + the column slice of X (c x R): the total FLOPs do not depend on R, a larger R only
    # rebuilds X fewer times, so R is capped (32768 rows of 123.6k SPLARE features = 16 GB)
    r = row_block or int(max(1024, (budget - x_bytes) // (n_active * 4 + c * 4)))
    r = min(r, n_active, row_block or 32768)
    return int(r), int(c)


@torch.no_grad()
def build_cooccurrence(docs, out_dir, dev, row_block=0, doc_chunk=0, rows=None, overwrite=False):
    """Exact M (Eq. 1, diagonal = df) written to out_dir/M.npy as an (n_active x n_active) float32 memmap.
    rows: optional subset of documents (tests). Returns the meta dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows is not None:
        docs = docs.subset(np.sort(np.asarray(rows)))
    n_docs = docs.n_rows

    with timer("document frequency of every feature"):
        df = doc_frequency(docs)
    active = np.flatnonzero(df > 0).astype(np.int64)
    n = len(active)
    col_map = np.full(docs.n_cols, -1, dtype=np.int64)
    col_map[active] = np.arange(n)
    need = n * n * 4
    log(f"active features: {n:,} of {docs.n_cols:,}; M will take {human_bytes(need)} on disk")

    meta_path, prog_path, m_path = out_dir / FILES["meta"], out_dir / FILES["progress"], out_dir / FILES["M"]
    meta = {"n_docs": int(n_docs), "n_cols": int(docs.n_cols), "n_active": int(n), "nnz_docs": int(docs.nnz)}
    resume = m_path.exists() and meta_path.exists() and not overwrite
    if resume:
        old = load_json(meta_path)
        if any(old.get(k) != v for k, v in meta.items()):
            raise SystemExit(f"[ERROR] {out_dir} holds an M of different data ({old}): use --overwrite")
        M = np.load(m_path, mmap_mode="r+")
        prog = load_json(prog_path) if prog_path.exists() else {"done": [], "stats": {}}
        done, stats = set(prog["done"]), dict(prog.get("stats", {}))
        log(f"resuming: {len(done)} row blocks already done")
    else:
        free_disk = shutil.disk_usage(out_dir).free
        if free_disk < need * 1.02:
            raise SystemExit(f"[ERROR] not enough disk in {out_dir}: need {human_bytes(need)}, "
                             f"free {human_bytes(free_disk)}")
        M = np.lib.format.open_memmap(m_path, mode="w+", dtype=np.float32, shape=(n, n))
        np.save(out_dir / FILES["active"], active)
        np.save(out_dir / FILES["df"], df)
        save_json({**meta, "complete": False}, meta_path)
        done, stats = set(), {}
        save_json({"done": [], "stats": {}}, prog_path)

    R, c = auto_blocks(n, dev, row_block, doc_chunk)
    blocks = [(r0, min(n, r0 + R)) for r0 in range(0, n, R)]
    log(f"row blocks: {len(blocks)} of {R:,} features; document chunks of {c:,} "
        f"({(n_docs + c - 1) // c:,} per block)")
    X = torch.zeros(c, n, dtype=torch.float32, device=dev)
    for bi, (r0, r1) in enumerate(blocks):
        if bi in done:
            continue
        out = torch.zeros(r1 - r0, n, dtype=torch.float32, device=dev)
        with timer(f"row block {bi + 1}/{len(blocks)} (features {r0:,}-{r1:,})"):
            step = max(1, ((n_docs + c - 1) // c) // 10)
            for ci, s in enumerate(range(0, n_docs, c)):
                e = min(n_docs, s + c)
                rr, cc = _chunk_to_device(docs, s, e, col_map, dev)
                Xc = X[: e - s]
                Xc.zero_()
                Xc[rr, cc] = 1.0
                out.addmm_(Xc[:, r0:r1].T, Xc)                    # (R x c) @ (c x n): exact counts
                if ci % step == 0 and bi == 0:
                    log(f"    documents {e:,}/{n_docs:,}")
        diag = out[torch.arange(r1 - r0, device=dev), torch.arange(r0, r1, device=dev)]
        dref = torch.from_numpy(df[active[r0:r1]].astype(np.float32)).to(dev)
        if not torch.equal(diag, dref):
            raise RuntimeError("diagonal of M differs from the document frequencies: counts are not exact")
        # off-diagonal statistics of this block, while it is on the device (in pieces: no big temporaries)
        out[torch.arange(r1 - r0, device=dev), torch.arange(r0, r1, device=dev)] = 0
        nz, mx = 0, 0.0
        for s0 in range(0, r1 - r0, 2048):
            part = out[s0:s0 + 2048]
            nz += int(torch.count_nonzero(part))
            mx = max(mx, float(part.max()))
        out[torch.arange(r1 - r0, device=dev), torch.arange(r0, r1, device=dev)] = dref
        M[r0:r1] = out.cpu().numpy()
        M.flush()
        done.add(bi)
        stats[str(bi)] = [nz, mx]
        save_json({"done": sorted(done), "stats": stats}, prog_path)
        del out
    del X

    nnz_offdiag = int(sum(v[0] for v in stats.values()))
    max_count = float(max((v[1] for v in stats.values()), default=0.0))
    meta.update({"complete": True, "nnz_offdiag": int(nnz_offdiag),
                 "density_offdiag": nnz_offdiag / max(1, n * (n - 1)), "max_count_offdiag": max_count,
                 "bytes_dense": int(need), "bytes_sparse_csr_estimate": int(nnz_offdiag * 8 + (n + 1) * 8)})
    save_json(meta, meta_path)
    log(f"M done: {n:,} x {n:,}, off-diagonal non-zeros {nnz_offdiag:,} "
        f"(density {meta['density_offdiag']:.4f}), largest count {max_count:,.0f}")
    return meta


class Cooccurrence:
    """Read access to a built M: rows of given features, document frequencies, active features."""

    def __init__(self, folder):
        folder = Path(folder)
        if not (folder / FILES["meta"]).exists():
            raise SystemExit(f"[ERROR] {folder} has no co-occurrence matrix: run 05_build_cooccurrence.py first")
        self.meta = json.loads((folder / FILES["meta"]).read_text())
        if not self.meta.get("complete"):
            raise SystemExit(f"[ERROR] the co-occurrence matrix in {folder} is incomplete: rerun 05 (it resumes)")
        self.folder = folder
        self.M = np.load(folder / FILES["M"], mmap_mode="r")
        self.active = np.load(folder / FILES["active"]).astype(np.int64)
        self.df_full = np.load(folder / FILES["df"]).astype(np.int64)
        self.n_cols = int(self.meta["n_cols"])
        self.n_docs = int(self.meta["n_docs"])
        self.n_active = len(self.active)
        self.col_map = np.full(self.n_cols, -1, dtype=np.int64)
        self.col_map[self.active] = np.arange(self.n_active)
        self.df_active = self.df_full[self.active]

    def rows(self, compact_ids):
        """M[compact_ids, :] as a float32 numpy array (len x n_active), read from the memmap."""
        compact_ids = np.asarray(compact_ids, dtype=np.int64)
        order = np.argsort(compact_ids, kind="stable")          # sorted reads are much faster on disk
        out = np.empty((len(compact_ids), self.n_active), dtype=np.float32)
        out[order] = self.M[compact_ids[order]]
        return out

    def neighbours(self, feature, n=10):
        """Most co-occurring features of a feature (full ids, counts), for inspection."""
        c = self.col_map[int(feature)]
        if c < 0:
            return []
        row = np.asarray(self.M[c]).copy()
        row[c] = 0
        top = np.argsort(-row)[:n]
        return [(int(self.active[j]), float(row[j])) for j in top if row[j] > 0]
