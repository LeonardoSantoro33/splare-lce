#!/usr/bin/env python
"""
08_significance.py — Paired significance test between two runs already evaluated (per-query metrics in results/).

Default: SPLARE vs SPLARE + LCE (the configuration of results/lce_best.json) on the 5,980 test queries.
    python -u src/08_significance.py
    python -u src/08_significance.py --a splare_test --b lce_cond_a0.2_add20_test      # any two result files
No GPU, a few seconds; names = json file names in WORK_DIR/results (with or without .json).

For each metric: mean A, mean B, difference B - A, 95% bootstrap CI of the difference (10,000 resamples),
paired t-test p-value (normal approximation) and paired randomization (sign-flip) test p-value, two-sided.
A is the reference, B the system under test. A difference is called significant only if p(perm) < threshold
AND the 95% CI excludes 0. The primary metric is MRR@10 (the official MS MARCO metric, the one used for the
choice on validation), tested at 0.05; with --n-tests m the threshold becomes the Bonferroni one, 0.05 / m.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from config import Paths
from utils import append_csv_row, now_tag, save_json


def load(results_dir, name):
    # names contain dots (e.g. "a0.2"), so do not rely on Path.suffix
    if not name.endswith(".json"):
        name = f"{name}.json"
    p = Path(name)
    if not p.is_absolute() and not p.exists():
        p = results_dir / p.name
    if not p.exists():
        raise SystemExit(f"[ERROR] {p} not found")
    data = json.loads(p.read_text())
    return data["per_query"], p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="splare_test", help="reference results json (default splare_test)")
    ap.add_argument("--b", default=None, help="system results json (default: <chosen configuration>_test)")
    ap.add_argument("--tag", default="", help="default B = the configuration of results/lce_best_TAG.json "
                                              "(second round of validation)")
    ap.add_argument("--metrics", nargs="+", default=["MRR@10", "nDCG@10", "R@10", "R@100", "R@1000"])
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--n-tests", type=int, default=1, help="Bonferroni: number of tests on the same question")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = Paths()
    results_dir = paths.results
    if args.b is None:
        best_file = paths.best_config_for(args.tag)
        if not best_file.exists():
            raise SystemExit(f"[ERROR] {best_file} not found: pass --b or run 06_validate_lce.py")
        args.b = json.loads(best_file.read_text())["name"] + "_test"
    pa, fa = load(results_dir, args.a)
    pb, fb = load(results_dir, args.b)
    ia = {str(q): i for i, q in enumerate(pa["qid"])}
    ib = {str(q): i for i, q in enumerate(pb["qid"])}
    common = sorted(set(ia) & set(ib))
    thr = args.alpha / max(1, args.n_tests)
    print(f"A = {fa.name}  ({len(ia)} queries)")
    print(f"B = {fb.name}  ({len(ib)} queries)")
    print(f"paired queries: {len(common)};  threshold p < {thr:.4f}"
          + (f" (Bonferroni over {args.n_tests} tests)" if args.n_tests > 1 else ""))
    if len(common) != len(ia) or len(common) != len(ib):
        print("WARNING: the two runs do not contain the same queries")

    rng = np.random.default_rng(args.seed)
    name_a, name_b = fa.name[:-5], fb.name[:-5]
    row = {"a": name_a, "b": name_b, "n_queries": len(common), "iters": args.iters, "n_tests": args.n_tests,
           "threshold": thr, "time": now_tag()}
    per_metric = {}
    print(f"\n{'metric':>9} {'A':>8} {'B':>8} {'B-A':>8}   {'95% CI':>19}  {'p(t)':>8} {'p(perm)':>8}  sig")
    for m in args.metrics:
        if m not in pa or m not in pb:
            print(f"{m:>9}  (not in the results: skipped)")
            continue
        a = np.array([pa[m][ia[q]] for q in common], dtype=np.float64)
        b = np.array([pb[m][ib[q]] for q in common], dtype=np.float64)
        d = b - a
        n = len(d)
        mean = d.mean()
        sd = d.std(ddof=1) if n > 1 else 0.0
        p_t = math.erfc(abs(mean / (sd / math.sqrt(n))) / math.sqrt(2)) if sd > 0 else 1.0
        boot = np.empty(args.iters)
        for i in range(args.iters):
            boot[i] = d[rng.integers(0, n, n)].mean()
        lo, hi = np.percentile(boot, [2.5, 97.5])
        hits, done = 0, 0
        while done < args.iters:          # in chunks, to keep memory small on the login node
            c = min(500, args.iters - done)
            signs = rng.choice([-1.0, 1.0], size=(c, n))
            hits += int(np.sum(np.abs((signs * d).mean(axis=1)) >= abs(mean)))
            done += c
        p_perm = (hits + 1) / (args.iters + 1)
        sig = bool(p_perm < thr and (lo > 0 or hi < 0))
        print(f"{m:>9} {a.mean():8.4f} {b.mean():8.4f} {mean:+8.4f}   [{lo:+.4f}, {hi:+.4f}]  {p_t:8.4f} "
              f"{p_perm:8.4f}  {'yes' if sig else 'no'}")
        res = {"A": a.mean(), "B": b.mean(), "diff": mean, "ci95_lo": lo, "ci95_hi": hi,
               "p_ttest": p_t, "p_perm": p_perm, "significant": sig}
        per_metric[m] = res
        row.update({f"{m}_{k}": v for k, v in res.items()})
    print(f"\nsignificant = p(perm) < {thr:.4f} and the 95% CI of B - A does not contain 0 (two-sided).")

    out_json = results_dir / "significance" / f"{name_b}__vs__{name_a}.json"
    save_json({**{k: row[k] for k in ("a", "b", "n_queries", "iters", "n_tests", "threshold", "time")},
               "metrics": per_metric}, out_json)
    append_csv_row(results_dir / "significance.csv", row)
    print(f"saved: {out_json}")


if __name__ == "__main__":
    main()
