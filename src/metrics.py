"""
metrics.py — Evaluation metrics (same definitions as trec_eval / MS MARCO).

  MRR@10  (RR@10)  : 1/rank of the first relevant passage in the top 10 (0 if there is none).
                     This is the official MS MARCO dev metric.
  nDCG@10          : gain = relevance grade (1 for every judged passage of MS MARCO dev), discount
                     log2(rank+1), normalized by the ideal ranking.
  R@k (Recall@k)   : fraction of the relevant passages that appear in the top k.

MS MARCO dev small has ~1.07 relevant passages per query, all with grade 1: the metrics are binary.
"""
import numpy as np


def evaluate_run(qids, ranked_pids, qrels, rel_threshold=1, cutoffs_recall=(10, 50, 100, 1000),
                 ndcg_cut=10, mrr_cut=10):
    """
    qids        : array (nq,) with the qid of each row of ranked_pids
    ranked_pids : array (nq, K) of pids sorted by decreasing score (-1 = empty)
    qrels       : {qid: {pid: rel}}
    Returns (dict of mean metrics, dict of per-query metrics).
    The mean is over queries present both in the run and in the qrels (with at least one relevant passage).
    """
    qids = np.asarray(qids)
    ranked_pids = np.asarray(ranked_pids)
    K = ranked_pids.shape[1]
    per_query = {"qid": [], f"MRR@{mrr_cut}": [], f"nDCG@{ndcg_cut}": []}
    rec_cuts = [c for c in cutoffs_recall if c <= K]
    for c in rec_cuts:
        per_query[f"R@{c}"] = []
    disc = 1.0 / np.log2(np.arange(2, K + 2))

    n_missing = 0
    for qid, row in zip(qids.tolist(), ranked_pids):
        judg = qrels.get(qid)
        if judg is None:
            n_missing += 1
            continue
        rel_bin = {p for p, r in judg.items() if r >= rel_threshold}
        if not rel_bin:
            continue
        gains = np.array([judg.get(int(p), 0) for p in row], dtype=np.float64)
        gains[row < 0] = 0.0
        is_rel = np.array([int(p) in rel_bin for p in row], dtype=bool)

        hits = np.flatnonzero(is_rel[:mrr_cut])
        mrr = 1.0 / (hits[0] + 1) if len(hits) else 0.0

        g = np.clip(gains[:ndcg_cut], 0, None)
        dcg = float((g * disc[:len(g)]).sum())
        ideal = np.sort(np.array([r for r in judg.values() if r > 0], dtype=np.float64))[::-1][:ndcg_cut]
        idcg = float((ideal * disc[:len(ideal)]).sum())
        ndcg = dcg / idcg if idcg > 0 else 0.0

        per_query["qid"].append(qid)
        per_query[f"MRR@{mrr_cut}"].append(mrr)
        per_query[f"nDCG@{ndcg_cut}"].append(ndcg)
        for c in rec_cuts:
            per_query[f"R@{c}"].append(is_rel[:c].sum() / len(rel_bin))

    n_eval = len(per_query["qid"])
    summary = {k: float(np.mean(v)) if n_eval else 0.0 for k, v in per_query.items() if k != "qid"}
    summary["n_queries"] = n_eval
    n_judged = sum(1 for d in qrels.values() if any(r >= rel_threshold for r in d.values()))
    summary["n_queries_in_qrels"] = n_judged
    if n_eval < n_judged:
        summary["warning"] = (f"only {n_eval} of the {n_judged} judged queries are in the run: "
                              "the mean is computed over those only")
    return summary, per_query


def overlap_at_k(ranked_a, ranked_b, k):
    """Mean fraction of shared items between the top-k of two runs (same rows = same queries)."""
    a = np.asarray(ranked_a)[:, :k]
    b = np.asarray(ranked_b)[:, :k]
    vals = [len(set(x.tolist()) & set(y.tolist())) / k for x, y in zip(a, b)]
    return float(np.mean(vals)) if vals else 0.0


def write_trec_run(path, qids, ranked_pids, scores, tag="run"):
    """Standard TREC format: 'qid Q0 pid rank score tag' (readable by trec_eval, ranx, ir_measures)."""
    with open(path, "w") as f:
        for qid, pids, sc in zip(np.asarray(qids).tolist(), ranked_pids, scores):
            r = 0
            for p, s in zip(pids.tolist(), sc.tolist()):
                if p < 0:
                    continue
                r += 1
                f.write(f"{qid} Q0 {p} {r} {s:.6f} {tag}\n")


def format_metrics(m):
    keys = [k for k in m if k.startswith(("MRR", "nDCG", "R@"))]
    return "  ".join(f"{k}={m[k]:.4f}" for k in keys) + f"  (queries evaluated: {m.get('n_queries')})"
