"""
06_validate_lce.py — ONLINE phase of LCE (Sec. 3.2-3.3), model selection on the 1,000 VALIDATION queries.

For every configuration of the grid  normalization of M x alpha x pruning  (+ SPLARE without expansion) it
expands the validation queries (Eq. 2-3) and runs the exact search over the 8.84M passages. The configuration
with the best validation MRR@10 (ties: fewer added latents) is saved to results/lce_best.json; 07_test_lce.py
evaluates that one, and only that one, on the test queries.

The test queries are never read here. If a test result already exists, the script stops (--force to rerun):
choosing again after having looked at the test set would make the test result optimistic.

Output (WORK_DIR/results/):
    lce_validation.csv      the whole grid: one row per configuration, all metrics, avg. added latents
    lce_best.json           the chosen configuration (+ its validation metrics and those of SPLARE alone)
    lce_validation_best_per_norm.csv   best configuration of each normalization (table of the paper)

Grid defaults (168 configurations + SPLARE alone, ~2-4 h on an A100):
    norms    raw cond cosine ppmi
    alpha    0.05 0.1 0.2 0.3 0.5 1          (relative: the strongest added weight = alpha x max V_q)
    pruning  add:5 add:10 add:20 add:50 add:100 rtau:0.05 rtau:0.1
    original query latents never pruned (alpha = 0 is exactly SPLARE)

    sbatch --time=06:00:00 -J val jobs/gpu.sbatch 06_validate_lce.py

SECOND ROUND (optional, declared). --src S1 S2 ... adds a dimension to the grid: the expansion is computed only from
the query latents with weight >= S x max(V_q) (0 = every latent, the first round). --tag NAME writes
lce_validation_NAME.csv, lce_best_NAME.json, ... instead of the files of the first round, which are never touched.
If the test has already been looked at, the script refuses to select again unless the run is explicitly declared
with  --tag NAME --after-test  (07_test_lce.py --tag NAME then tests the configuration chosen in this round).
Including 0 in --src re-runs configurations of the first round: their numbers must be identical (checked and logged).

    sbatch --time=03:00:00 -J val2 jobs/gpu.sbatch 06_validate_lce.py --tag src --after-test \
        --src 0 0.25 0.5 0.75 --alphas 0.05 0.1 0.2 0.3 --prunes add:10 add:20 add:50
"""
import argparse

import numpy as np
import pandas as pd

from config import Paths
from cooccurrence import Cooccurrence
from data import load_docs, query_set
from lce import NORMS, LCEConfig, build_grid, compute_expansions, search_configs
from metrics import evaluate_run
from pipeline import docs_to_device
from utils import get_device, log, log_run_header, now_tag, save_json, timer

DEFAULT_ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
DEFAULT_PRUNES = ["add:5", "add:10", "add:20", "add:50", "add:100", "rtau:0.05", "rtau:0.1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norms", nargs="+", default=list(NORMS), choices=list(NORMS))
    ap.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    ap.add_argument("--prunes", nargs="+", default=DEFAULT_PRUNES)
    ap.add_argument("--min-counts", nargs="+", type=int, default=[1],
                    help="ignore co-occurrences seen in fewer than N passages (default 1 = Eq. 1 as written)")
    ap.add_argument("--alpha-mode", choices=["relative", "absolute"], default="relative")
    ap.add_argument("--keep-original", action=argparse.BooleanOptionalAction, default=True,
                    help="never prune the original query latents (default yes)")
    ap.add_argument("--select-metric", default="MRR@10", choices=["MRR@10", "nDCG@10"])
    ap.add_argument("--k", type=int, default=100, help="depth of the search (MRR@10 needs 10; R@100 needs 100)")
    ap.add_argument("--q-batch", type=int, default=256)
    ap.add_argument("--doc-chunk", type=int, default=1_000_000)
    ap.add_argument("--row-block", type=int, default=2048, help="rows of M read at a time")
    ap.add_argument("--force", action="store_true", help="rerun even if test results already exist")
    ap.add_argument("--src", nargs="+", type=float, default=[0.0],
                    help="expand only from the query latents with weight >= S x max(V_q); 0 = every latent (default)")
    ap.add_argument("--tag", default="", help="write lce_validation_TAG.csv, lce_best_TAG.json, ... (second round)")
    ap.add_argument("--after-test", action="store_true",
                    help="declare a second round of selection after the test has been looked at (needs --tag)")
    args = ap.parse_args()
    paths = Paths()
    paths.ensure_work_dir()
    dev = get_device()
    log_run_header("06_validate_lce.py", args, dev)
    out_csv, out_best = paths.validation_csv_for(args.tag), paths.best_config_for(args.tag)
    out_norm = paths.best_per_norm_csv_for(args.tag)
    if args.after_test and not args.tag:
        raise SystemExit("[ERROR] --after-test needs --tag NAME: the files of the first round must not be overwritten")

    seen_test = sorted(p.name for p in paths.results.glob("lce_*_test.json"))
    if seen_test and not args.force:
        if not (args.after_test and args.tag):
            raise SystemExit(f"[ERROR] test results already exist ({seen_test[:3]}): selecting again after looking at "
                             f"the test set biases it. A second round must be declared: --tag NAME --after-test "
                             f"(its files are kept apart). --force only for a bug fix.")
        log(f"WARNING: SECOND ROUND, declared (--after-test): the test set was already looked at ({len(seen_test)} "
            f"result files). The new variants are chosen on validation only; report both rounds.")

    cooc = Cooccurrence(paths.cooc)
    docs = load_docs(paths)
    if docs.n_cols != cooc.n_cols or docs.n_rows != cooc.n_docs:
        raise SystemExit("[ERROR] M was built on different documents: rerun 05_build_cooccurrence.py")
    doc_pids = np.asarray(docs.ids, dtype=np.int64)
    tdocs = docs_to_device(docs, dev)

    qs = query_set(paths, "val")
    grid = build_grid(args.norms, args.alphas, args.prunes, args.min_counts, args.alpha_mode, args.keep_original,
                      srcs=args.src)
    log(f"validation: {qs.q.n_rows:,} queries, {len(grid)} configurations (the first is SPLARE without expansion)")
    E = compute_expansions(qs.q, cooc, grid, dev, args.row_block)
    with timer("exact search of every configuration"):
        res = search_configs(tdocs, qs.q, E, grid, args.k, dev, args.q_batch, args.doc_chunk)
    del E
    qids = np.asarray(qs.q.ids)
    table = []
    for c in grid:
        m, _ = evaluate_run(qids, doc_pids[res[c]["rows"]], qs.qrels)
        table.append({"config": c.name(), **c.to_dict(), "avg_query_nnz": res[c]["avg_query_nnz"],
                      "avg_added": res[c]["avg_added"],
                      **{k: v for k, v in m.items() if k.startswith(("MRR", "nDCG", "R@"))}})
    df = pd.DataFrame(table)
    df.to_csv(out_csv, index=False)

    sm = args.select_metric
    if args.tag and paths.validation_csv.exists():
        # integrity check: a configuration run again must give the numbers of the first round, exactly
        old = pd.read_csv(paths.validation_csv)
        both = df.merge(old, on="config", suffixes=("", "_first"))
        if len(both):
            diff = float((both[sm] - both[f"{sm}_first"]).abs().max())
            log(f"check against the first round: {len(both)} configurations in common, max |{sm} difference| = "
                f"{diff:.6f}" + ("" if diff < 1e-9 else "   <-- NOT identical: the first round was run with other "
                                                        "settings (e.g. --k) or the code changed"))
    base = table[0]
    best = max(table[1:], key=lambda r: (round(r[sm], 6), -r["avg_added"]))
    log(f"{'configuration':<36} {'MRR@10':>7} {'nDCG@10':>8} {'R@100':>7} {'added':>6}")
    for r in sorted(table, key=lambda r: -r[sm])[:15]:
        log(f"{r['config']:<36} {r['MRR@10']:7.4f} {r['nDCG@10']:8.4f} {r.get('R@100', float('nan')):7.4f} "
            f"{r['avg_added']:6.1f}")
    per_norm = (df[df["norm"].notna()].sort_values([sm, "avg_added"], ascending=[False, True])
                .groupby("norm", sort=False).head(1))
    per_norm = pd.concat([df.iloc[[0]], per_norm])
    per_norm.to_csv(out_norm, index=False)
    log(f"SPLARE alone: {sm} {base[sm]:.4f}.  Best LCE: {best['config']}  {sm} {best[sm]:.4f} "
        f"({best[sm] - base[sm]:+.4f}).  Best of each normalization:")
    for r in per_norm.to_dict("records")[1:]:
        log(f"    {r['norm']:<7} {r['config']:<36} {sm} {r[sm]:.4f} ({r[sm] - base[sm]:+.4f})")
    lce_rows = df[df["norm"].notna()]
    if lce_rows["src"].nunique() > 1:
        per_src = lce_rows.sort_values([sm, "avg_added"], ascending=[False, True]).groupby("src", sort=True).head(1)
        log("Best configuration for each value of src (0 = every query latent):")
        for r in per_src.sort_values("src").to_dict("records"):
            log(f"    src {r['src']:<5g} {r['config']:<40} {sm} {r[sm]:.4f} ({r[sm] - base[sm]:+.4f})")
    if best[sm] <= base[sm]:
        log("NOTE: no configuration beats SPLARE on validation; the 'best' one is still saved and tested by 07.")

    best_cfg = LCEConfig.from_dict(best)
    info = {"selected_on": "val", "select_metric": sm, "n_queries": int(qs.q.n_rows), "time": now_tag(),
            "config": best_cfg.to_dict(), "name": best_cfg.name(), "best": best, "baseline": base,
            "grid_size": len(grid), "k": args.k, "tag": args.tag,
            "round": "second (declared, after the test was looked at)" if (args.tag and seen_test) else
                     ("second (declared)" if args.tag else "first")}
    save_json(info, out_best)
    log(f"grid saved to {out_csv}")
    log(f"chosen configuration saved to {out_best}: {best_cfg.name()}")
    log("Done.")


if __name__ == "__main__":
    main()
