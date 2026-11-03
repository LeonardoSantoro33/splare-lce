"""
07_test_lce.py — The final evaluation: the configuration chosen on validation, on the 5,980 TEST queries.

Nothing is chosen here. The script reads results/lce_best.json (written by 06_validate_lce.py), expands the test
queries with that configuration and runs the exact search (top 1000) for SPLARE and SPLARE + LCE in the same
pass, so that the two runs are paired query by query.

Output (WORK_DIR/results/ and runs/):
    <config>_test.json          metrics + per-query metrics of SPLARE + LCE (input of 08_significance.py)
    splare_test.json            SPLARE alone on test (rewritten; it must equal the one of 04)
    lce_test_analysis.json      wins / ties / losses per query, gain by query length (quartiles of the number of
                                latents of the original query), how much of the ranking changes
    lce_examples_test.txt       original vs added latents of the queries that gained / lost the most
    runs/<config>_test.trec     TREC run (readable by trec_eval, ranx, ir_measures)

    sbatch --time=03:00:00 -J test jobs/gpu.sbatch src/07_test_lce.py

--config NORM:ALPHA:PRUNE evaluates a configuration given by hand instead (e.g. an ablation): its result is
marked "selection: manual" and must not be presented as the result chosen on validation.
--config also accepts :mcN and :srcS at the end, e.g. cond:0.2:add:20:src0.5.

--tag NAME evaluates the configuration chosen in a declared SECOND round of validation (results/lce_best_NAME.json,
written by 06_validate_lce.py --tag NAME --after-test). Its analysis and examples are written to
lce_test_analysis_NAME.json and lce_examples_test_NAME.txt: the files of the first round are never overwritten.
"""
import argparse

import numpy as np
import torch

from config import Paths
from cooccurrence import Cooccurrence
from data import load_docs, query_set
from lce import LCEConfig, compute_expansions, expand, explain_row, prune, search_configs
from metrics import overlap_at_k
from pipeline import docs_to_device, report_and_save
from retrieval import csr_rows_to_dense
from utils import get_device, load_json, log, log_run_header, save_json, timer


def analysis(pq_base, pq_lce, q_nnz, metric="MRR@10"):
    """Per-query comparison: wins / ties / losses and the mean gain by quartile of the query length."""
    a = np.asarray(pq_base[metric], dtype=np.float64)
    b = np.asarray(pq_lce[metric], dtype=np.float64)
    d = b - a
    out = {"metric": metric, "n": int(len(d)), "wins": int((d > 1e-12).sum()), "ties": int((np.abs(d) <= 1e-12).sum()),
           "losses": int((d < -1e-12).sum()), "mean_gain": float(d.mean())}
    edges = np.quantile(q_nnz, [0, 0.25, 0.5, 0.75, 1.0])
    bins = np.clip(np.searchsorted(edges, q_nnz, side="right") - 1, 0, 3)
    out["by_query_length"] = []
    for i in range(4):
        m = bins == i
        out["by_query_length"].append({
            "quartile": i + 1, "nnz_min": int(q_nnz[m].min()) if m.any() else None,
            "nnz_max": int(q_nnz[m].max()) if m.any() else None, "n": int(m.sum()),
            "base": float(a[m].mean()) if m.any() else None, "lce": float(b[m].mean()) if m.any() else None,
            "gain": float(d[m].mean()) if m.any() else None,
            "wins": int((d[m] > 1e-12).sum()), "losses": int((d[m] < -1e-12).sum())})
    return out


def first_rel_rank(rows, doc_pids, rel):
    hit = np.flatnonzero(np.isin(doc_pids[rows], list(rel)))
    return int(hit[0]) + 1 if len(hit) else None


def write_examples(paths, qs, cfg, E, res, doc_pids, dev, n, suffix=""):
    """The n queries with the largest gain, the n with the largest loss, n at random."""
    base = LCEConfig(None, 0.0)
    q = qs.q
    ranks = []
    for i in range(q.n_rows):
        rel = {p for p, r in qs.qrels[int(q.ids[i])].items() if r > 0}
        ranks.append((first_rel_rank(res[base]["rows"][i], doc_pids, rel),
                      first_rel_rank(res[cfg]["rows"][i], doc_pids, rel)))
    rr = lambda r: 1.0 / r if r is not None and r <= 10 else 0.0  # noqa: E731
    gain = np.array([rr(b) - rr(a) for a, b in ranks])
    rng = np.random.default_rng(0)
    groups = [("largest gains (MRR@10)", np.argsort(-gain)[:n]), ("largest losses (MRR@10)", np.argsort(gain)[:n]),
              ("random", rng.choice(q.n_rows, min(n, q.n_rows), replace=False))]
    lines = [f"# SPLARE + LCE on test | {cfg.name()}",
             "# latent:weight, strongest first. rank = position of the first relevant passage in the top 1000", ""]
    E_cfg = E[cfg.e_key]
    for title, rows in groups:
        rows = np.asarray(rows, dtype=np.int64)
        Vq = csr_rows_to_dense(q, rows, dev)
        Vext = prune(expand(Vq, torch.from_numpy(E_cfg[rows]).to(dev), cfg.alpha, cfg.alpha_mode), Vq, cfg.prune,
                     cfg.keep_original)
        lines += [f"## {title}", ""]
        for j, i in enumerate(rows.tolist()):
            qid = int(q.ids[i])
            o, a = explain_row(Vq[j], Vext[j], 12)
            lines += [f"query {qid}: {qs.texts.get(qid, '') if qs.texts else ''}",
                      f"   original ({int((Vq[j] > 0).sum())} latents): {o}",
                      f"   added    ({int(((Vext[j] > 0) & (Vq[j] <= 0)).sum())} latents): {a}",
                      f"   rank of the first relevant passage: {ranks[i][0]} -> {ranks[i][1]}", ""]
    out = paths.results / f"lce_examples_test{suffix}.txt"
    out.write_text("\n".join(lines) + "\n")
    log(f"examples saved to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="NORM:ALPHA:PRUNE[:mcN][:srcS] given by hand (ablation; default: results/lce_best.json)")
    ap.add_argument("--tag", default="",
                    help="use results/lce_best_TAG.json (second round of validation, 06 --tag TAG --after-test)")
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--q-batch", type=int, default=256)
    ap.add_argument("--doc-chunk", type=int, default=1_000_000)
    ap.add_argument("--row-block", type=int, default=2048)
    ap.add_argument("--examples", type=int, default=5)
    args = ap.parse_args()
    paths = Paths()
    paths.ensure_work_dir()
    dev = get_device()
    log_run_header("07_test_lce.py", args, dev)

    best_file = paths.best_config_for(args.tag)
    suffix = f"_{args.tag}" if args.tag else ""
    chosen = LCEConfig.from_dict(load_json(best_file)["config"]) if best_file.exists() else None
    if args.config and LCEConfig.parse(args.config) != chosen:
        cfg = LCEConfig.parse(args.config)
        selection = "manual (--config)"
        log(f"WARNING: configuration given by hand ({cfg.name()}): NOT the one chosen on validation")
    else:
        if not best_file.exists():
            raise SystemExit(f"[ERROR] {best_file} not found: run 06_validate_lce.py"
                             + (f" --tag {args.tag} --after-test" if args.tag else "") + " first")
        cfg = LCEConfig.from_dict(load_json(best_file)["config"])
        selection = "validation (06_validate_lce.py" + (f", second round, tag {args.tag})" if args.tag else ")")
        log(f"configuration chosen on validation{' (second round, tag ' + args.tag + ')' if args.tag else ''}: "
            f"{cfg.name()}")
    if cfg.is_baseline:
        raise SystemExit("[ERROR] the configuration has alpha = 0: nothing to test beyond 04_baseline.py")
    if args.tag and not args.config:
        first = paths.best_config_for("")
        if first.exists() and load_json(first)["name"] == cfg.name() and paths.result_json(cfg.name(), "test").exists():
            log(f"The second round chose the same configuration as the first round ({cfg.name()}): its test result "
                f"already exists ({paths.result_json(cfg.name(), 'test').name}). Nothing new to test; nothing "
                f"was overwritten.")
            return

    cooc = Cooccurrence(paths.cooc)
    docs = load_docs(paths)
    if docs.n_cols != cooc.n_cols or docs.n_rows != cooc.n_docs:
        raise SystemExit("[ERROR] M was built on different documents: rerun 05_build_cooccurrence.py")
    doc_pids = np.asarray(docs.ids, dtype=np.int64)
    tdocs = docs_to_device(docs, dev)

    qs = query_set(paths, "test")
    base = LCEConfig(None, 0.0)
    E = compute_expansions(qs.q, cooc, [cfg], dev, args.row_block)
    with timer(f"exact search on test ({qs.q.n_rows:,} queries): SPLARE and {cfg.name()}"):
        res = search_configs(tdocs, qs.q, E, [base, cfg], args.k, dev, args.q_batch, args.doc_chunk)
    qids = np.asarray(qs.q.ids)

    old = paths.result_json("splare", "test")
    old_mrr = load_json(old)["summary"]["MRR@10"] if old.exists() else None
    m0, pq0, _ = report_and_save(paths, "splare", "test", qids, res[base]["rows"], res[base]["scores"], doc_pids,
                                 qs.qrels, extra={"avg_query_nnz": res[base]["avg_query_nnz"]})
    if old_mrr is not None and abs(old_mrr - m0["MRR@10"]) > 1e-6:
        log(f"WARNING: SPLARE on test gives MRR@10 {m0['MRR@10']:.6f} here and {old_mrr:.6f} in 04: investigate")
    m1, pq1, _ = report_and_save(
        paths, cfg.name(), "test", qids, res[cfg]["rows"], res[cfg]["scores"], doc_pids, qs.qrels,
        extra={**cfg.to_dict(), "selection": selection, "avg_query_nnz": res[cfg]["avg_query_nnz"],
               "avg_added": res[cfg]["avg_added"], "baseline_MRR@10": m0["MRR@10"],
               "baseline_nDCG@10": m0["nDCG@10"], "baseline_R@1000": m0.get("R@1000")})

    q_nnz = np.diff(np.asarray(qs.q.indptr))
    info = {"config": cfg.name(), "selection": selection, "n_queries": int(qs.q.n_rows),
            "avg_added": res[cfg]["avg_added"],
            "overlap_top10": overlap_at_k(res[base]["rows"], res[cfg]["rows"], 10),
            "overlap_top100": overlap_at_k(res[base]["rows"], res[cfg]["rows"], min(100, args.k)),
            "MRR@10": analysis(pq0, pq1, q_nnz, "MRR@10")}
    if "R@1000" in pq0:
        info["R@1000"] = analysis(pq0, pq1, q_nnz, "R@1000")
    if selection.startswith("validation"):
        save_json(info, paths.results / f"lce_test_analysis{suffix}.json")
    a = info["MRR@10"]
    log(f"per query (MRR@10): {a['wins']} wins, {a['ties']} ties, {a['losses']} losses; "
        f"top-10 overlap with SPLARE {info['overlap_top10']:.3f}")
    for g in a["by_query_length"]:
        if g["n"]:
            log(f"   query length quartile {g['quartile']} ({g['nnz_min']}-{g['nnz_max']} latents, {g['n']} queries): "
                f"MRR@10 {g['base']:.4f} -> {g['lce']:.4f} ({g['gain']:+.4f})")
    log(f"SPLARE {m0['MRR@10']:.4f} -> SPLARE + LCE {m1['MRR@10']:.4f} MRR@10 on test ({m1['MRR@10'] - m0['MRR@10']:+.4f}). "
        f"Significance: python -u src/08_significance.py --a splare_test --b {cfg.name()}_test")
    if selection.startswith("validation"):
        write_examples(paths, qs, cfg, E, res, doc_pids, dev, args.examples, suffix)
    log("Done.")


if __name__ == "__main__":
    main()
