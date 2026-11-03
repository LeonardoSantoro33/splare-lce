"""
pipeline.py — Functions shared by the scripts (documents on the device, evaluation and saving of a run).
"""
import numpy as np
import torch

from metrics import evaluate_run, format_metrics, write_trec_run
from retrieval import TorchCSR, gpu_fits
from utils import append_csv_row, log, now_tag, save_json, timer


def docs_to_device(docs, dev):
    need = docs.nnz * 8 + docs.n_rows * 8
    if not gpu_fits(need, dev):
        raise SystemExit(f"[ERROR] The documents ({need / 1e9:.1f} GB) do not fit in the free GPU memory.")
    with timer(f"copying {docs.n_rows:,} documents to {dev.type}"):
        return TorchCSR(docs, dev)


def free_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def report_and_save(paths, method, set_name, qids, doc_rows, scores, doc_pids, qrels, extra=None, save_run=True):
    """Convert rows -> pids, compute the metrics, save the TREC run + json (with per-query metrics, used by
    08_significance.py) + a row in results/summary.csv."""
    pids = np.where(doc_rows >= 0, np.asarray(doc_pids)[np.clip(doc_rows, 0, None)], -1)
    metrics, per_query = evaluate_run(qids, pids, qrels)
    log(f"[{method} | {set_name}] {format_metrics(metrics)}")
    if "warning" in metrics:
        log("WARNING: " + metrics["warning"])
    tag = now_tag()
    if save_run:
        paths.runs.mkdir(parents=True, exist_ok=True)
        write_trec_run(paths.runs / f"{method}_{set_name}.trec", qids, pids, scores, tag=method)
    out = {"method": method, "query_set": set_name, "time": tag, **metrics, **(extra or {})}
    save_json({"summary": out, "per_query": per_query}, paths.result_json(method, set_name))
    append_csv_row(paths.summary_csv, out)
    return metrics, per_query, pids
