"""
retrieval.py — EXACT (brute-force) search on GPU.

Production systems use an inverted index (e.g. Seismic) with approximate search; we do exact
search instead: on an A100 it is fast enough, and an exact result is the upper bound of any
approximate index (no approximation error to discuss in the paper).

Score = <q, d> with q and d sparse SPLARE vectors (131,072 SAE latents).

TRICK: torch.nn.functional.embedding_bag(doc_indices, TABLE, doc_weights) computes for each document
sum_t weight_t * TABLE[t]. With TABLE = Q^T (features x queries) you get all document-query scores
of a batch of queries in one pass over the documents. The cost does not depend on how many non-zeros
the queries have, so expanded (denser) LCE queries cost the same as the original ones here.
"""
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from utils import log


class TorchCSR:
    """A CSR matrix (rows = documents or queries) copied to the GPU (or CPU)."""

    def __init__(self, csr, device, value_dtype=torch.float32):
        self.n_rows, self.n_cols = csr.n_rows, csr.n_cols
        self.device = device
        self.indptr_np = np.asarray(csr.indptr, dtype=np.int64)
        with warnings.catch_warnings():          # read-only arrays (mmap): we copy them to the device anyway
            warnings.simplefilter("ignore", UserWarning)
            self.indptr = torch.from_numpy(self.indptr_np).to(device)
            self.indices = torch.from_numpy(np.ascontiguousarray(csr.indices, dtype=np.int32)).to(device)
            self.data = torch.from_numpy(np.ascontiguousarray(csr.data, dtype=np.float32)).to(device, value_dtype)

    def bags(self, start, end):
        """(indices, offsets, weights) of rows [start, end) in embedding_bag format."""
        a, b = int(self.indptr_np[start]), int(self.indptr_np[end])
        offsets = (self.indptr[start:end] - a).to(torch.int32)
        return self.indices[a:b], offsets, self.data[a:b]

    def free(self):
        del self.indptr, self.indices, self.data
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def rows_times_table(tcsr, start, end, table):
    """(CSR rows start..end) @ table   ->   (end-start, table.shape[1])"""
    idx, offs, w = tcsr.bags(start, end)
    if idx.numel() == 0:
        return torch.zeros(end - start, table.shape[1], device=table.device, dtype=table.dtype)
    return F.embedding_bag(idx, table, offs, mode="sum", per_sample_weights=w.to(table.dtype))


def csr_rows_to_dense(csr, rows, device, dtype=torch.float32):
    """Rows of a (numpy) CSR -> dense matrix (len(rows), n_cols) on device."""
    offsets, idx, val = csr.gather(rows)
    lens = np.diff(np.append(offsets, len(idx)))
    r = np.repeat(np.arange(len(rows), dtype=np.int64), lens)
    out = torch.zeros(len(rows), csr.n_cols, device=device, dtype=dtype)
    if len(idx):
        out[torch.from_numpy(r).to(device), torch.from_numpy(idx.astype(np.int64)).to(device)] = \
            torch.from_numpy(val.astype(np.float32)).to(device, dtype)
    return out


def _merge_topk(best_s, best_i, s, i, k, dim):
    if best_s is None:
        return s, i
    cat_s = torch.cat([best_s, s], dim=dim)
    cat_i = torch.cat([best_i, i], dim=dim)
    kk = min(k, cat_s.shape[dim])
    s2, pos = torch.topk(cat_s, kk, dim=dim)
    return s2, torch.gather(cat_i, dim, pos)


@torch.no_grad()
def search_with_query_table(doc_tcsr, table, k, doc_chunk=1_000_000):
    """table: (n_cols, nq) = queries as (dense) COLUMNS. Returns (scores, rows) of shape (nq, k)."""
    best_s = best_i = None
    for ds in range(0, doc_tcsr.n_rows, doc_chunk):
        de = min(doc_tcsr.n_rows, ds + doc_chunk)
        S = rows_times_table(doc_tcsr, ds, de, table).T.contiguous()   # (nq, n_doc_chunk)
        kk = min(k, de - ds)
        s, i = torch.topk(S, kk, dim=1)
        best_s, best_i = _merge_topk(best_s, best_i, s, i + ds, k, dim=1)
        del S
    return best_s, best_i


@torch.no_grad()
def sparse_search(doc_tcsr, query_csr, k, q_batch=256, doc_chunk=1_000_000, query_rows=None,
                  verbose=True):
    """Exact search of sparse queries (numpy CSR) over sparse documents (TorchCSR).
    Returns scores (nq, k) float32 and document rows (nq, k) int64 (numpy)."""
    rows_all = np.arange(query_csr.n_rows) if query_rows is None else np.asarray(query_rows)
    nq = len(rows_all)
    k = min(k, doc_tcsr.n_rows)
    out_s = np.empty((nq, k), dtype=np.float32)
    out_i = np.empty((nq, k), dtype=np.int64)
    step = max(1, (nq // q_batch) // 10)
    for b, qs in enumerate(range(0, nq, q_batch)):
        qe = min(nq, qs + q_batch)
        table = csr_rows_to_dense(query_csr, rows_all[qs:qe], doc_tcsr.device).T.contiguous()
        s, i = search_with_query_table(doc_tcsr, table, k, doc_chunk)
        out_s[qs:qe] = s.float().cpu().numpy()
        out_i[qs:qe] = i.cpu().numpy()
        if verbose and b % step == 0:
            log(f"  sparse search: {qe:,}/{nq:,} queries")
    return out_s, out_i


def gpu_fits(n_bytes, device, margin=0.85):
    if device.type != "cuda":
        return True
    free, _ = torch.cuda.mem_get_info()
    return n_bytes < free * margin
