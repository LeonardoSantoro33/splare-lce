"""
04_baseline.py — SPLARE without expansion: exact search over the 8.84M MS MARCO passages.

This is the reference of every LCE result: same documents, same queries, alpha = 0.
Query sets (default: all three):
    val    the 1,000 validation queries        (the reference of the grid search of 06)
    test   the 5,980 test queries              (the reference of the final result of 07)
    dev    all 6,980 dev queries               (only to compare with the published SPLARE numbers; LCE is
                                                never reported on it, because dev contains the val queries)

Automatic check: overlap of our top-k with the provided groundtruth_100.tsv / groundtruth.tsv (exact top-100 /
top-10 computed by whoever encoded the data). It must be ~1.00 (small differences only from exact score ties);
otherwise data, ids or code are not aligned: stop and investigate.

    sbatch -J base jobs/gpu.sbatch 04_baseline.py
"""
import argparse

import numpy as np

from config import QUERY_SETS, Paths
from data import load_docs, query_set
from metrics import overlap_at_k
from pipeline import docs_to_device, report_and_save
from retrieval import sparse_search
from sparse_io import load_ids, read_groundtruth
from utils import get_device, log, log_run_header, timer


def groundtruth_rows(path, qids_eval, provided_qids):
    """Provided top-k lists aligned with our query rows (through the qids; doc index = document row)."""
    gt = read_groundtruth(path, query_ids=provided_qids).sort_values(["q", "rank"])
    k_gt = int(gt.groupby("q").size().max())
    pos = {int(q): i for i, q in enumerate(np.asarray(qids_eval).tolist())}
    rows = np.full((len(pos), k_gt), -1, dtype=np.int64)
    for qid, grp in gt.groupby("qid"):
        i = pos.get(int(qid))
        if i is not None:
            rows[i, :len(grp)] = grp["d"].to_numpy()
    return rows, k_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-sets", nargs="+", default=list(QUERY_SETS), choices=list(QUERY_SETS))
    ap.add_argument("--k", type=int, default=1000, help="number of documents to retrieve per query")
    ap.add_argument("--q-batch", type=int, default=256)
    ap.add_argument("--doc-chunk", type=int, default=1_000_000)
    args = ap.parse_args()
    paths = Paths()
    paths.ensure_work_dir()
    dev = get_device()
    log_run_header("04_baseline.py", args, dev)

    docs = load_docs(paths)
    doc_pids = np.asarray(docs.ids)
    tdocs = docs_to_device(docs, dev)
    provided_qids = load_ids(paths.provided_file("query_ids"))

    for set_name in args.query_sets:
        qs = query_set(paths, set_name)
        q = qs.q
        with timer(f"exact SPLARE search on '{set_name}' ({q.n_rows:,} queries)"):
            scores, rows = sparse_search(tdocs, q, args.k, q_batch=args.q_batch, doc_chunk=args.doc_chunk)
        extra = {"avg_query_nnz": float(q.nnz / max(1, q.n_rows))}
        for key in ("groundtruth", "groundtruth_10"):
            p = paths.provided_file(key)
            if not p.exists():
                continue
            gt_rows, k_gt = groundtruth_rows(p, q.ids, provided_qids)
            kk = min(k_gt, rows.shape[1])
            ov = overlap_at_k(rows, gt_rows, kk)
            log(f"Overlap with the provided {p.name} (top-{kk}): {ov:.4f}  (expected ~1.00)")
            extra[f"overlap_{p.stem}"] = ov
        report_and_save(paths, "splare", set_name, np.asarray(q.ids), rows, scores, doc_pids, qs.qrels, extra=extra)
    log("Done.")


if __name__ == "__main__":
    main()
