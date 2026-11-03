"""
lce_plots.py — Figures and tables of the LCE experiments, read from the result files (no number typed by hand).

    python plots/lce_plots.py                      # reads $WORK_DIR/results; figures from the src validation (lce_validation_src.csv)
    python plots/lce_plots.py --round 1            # tables of the first round only (figures always use the src validation)
    python plots/lce_plots.py --results /path/to/results

Writes to plots/figures/ (next to this script), as .png and .pdf:
  fig1_validation              validation MRR@10 of EVERY configuration, one column per normalization of M
                               (the src validation: src 0, 0.25, 0.5, 0.75); the best configuration of
                               each normalization is marked and labelled; dashed line = SPLARE without expansion
  fig2..fig5 (_test_<norm>)    one per normalization (fig2 raw, fig3 cond, fig4 cosine, fig5 ppmi): the best configuration of that
                               normalization on validation, evaluated on test: difference from SPLARE for MRR@10,
                               nDCG@10, R@10, R@100, R@1000, with the 95% bootstrap CI and p(perm). The four figures
                               share the same y axis, so they can be compared side by side.
and tables.md: the tables of docs/RESULTS.md (validation per normalization, test, significance), filled in.

The test result of a configuration is results/<config>_test.json (07_test_lce.py, or 07 --config for the
configurations that were not the chosen one); its significance is read from results/significance/ (08) or, if that
file is missing, computed here the same way as 08 (10,000 bootstrap resamples and sign flips, seed 0).
A figure whose results are missing is skipped with a message that says which command produces them.
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "figures"
NORMS = ["raw", "cond", "cosine", "ppmi"]
NORM_NAME = {"raw": "raw counts (Eq. 1)", "cond": "P(j | i)", "cosine": "cosine", "ppmi": "PPMI"}
# validated categorical palette, fixed order (blue, orange, aqua, yellow): one color per normalization, always the
# same; a different marker per normalization as well, and every mark is labelled (aqua and yellow are below 3:1
# contrast on white)
COLORS = {"raw": "#2a78d6", "cond": "#eb6834", "cosine": "#1baf7a", "ppmi": "#eda100"}
MARKERS = {"raw": "o", "cond": "s", "cosine": "^", "ppmi": "D"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METRICS = ["MRR@10", "nDCG@10", "R@10", "R@100", "R@1000"]


def style(ax, title, xlabel, ylabel, subtitle=None):
    pad = 6 + (12 * (subtitle.count("\n") + 1) if subtitle else 0)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left", pad=pad)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=MUTED, fontsize=8, va="bottom")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"written: {OUT_DIR / name}.png/.pdf")


def load_json(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


# ----------------------------------------------------------------------
#  the validation grid (both rounds) and the best configuration of each normalization
# ----------------------------------------------------------------------
def load_grid(results, rnd="all", src_only=False):
    # src_only=True: the grid of the src validation (lce_validation_src.csv: src 0, 0.25, 0.5, 0.75), used by the figures
    """Every configuration evaluated on validation. Round 2 (lce_validation_src.csv) re-runs some configurations of
    round 1 with identical numbers: each configuration is kept once."""
    parts = []
    names = ["lce_validation_src.csv"] if src_only else \
        ["lce_validation.csv"] + ([] if rnd == "1" else ["lce_validation_src.csv"])
    for name in names:
        p = results / name
        if p.exists():
            d = pd.read_csv(p)
            if "src" not in d.columns:
                d["src"] = 0.0
            d["src"] = d["src"].fillna(0.0)
            parts.append(d)
    if not parts:
        return None
    g = pd.concat(parts, ignore_index=True).drop_duplicates("config", keep="first").reset_index(drop=True)
    return g


def best_per_norm(g):
    """Best validation MRR@10 of each normalization; ties: fewer added latents (the rule of 06_validate_lce.py)."""
    lce = g[g["norm"].notna()].copy()
    lce["_m"] = lce["MRR@10"].round(6)
    best = lce.sort_values(["_m", "avg_added"], ascending=[False, True]).groupby("norm", sort=False).head(1)
    return {r["norm"]: r for r in best.to_dict("records")}


def describe(r):
    """'alpha 0.2, add:20' (+ ', src 0.75')"""
    s = f"α {r['alpha']:g}, {r['prune']}"
    if r.get("src") == r.get("src"):   # not NaN: the src validation always shows its src, also 0
        s += f", src {float(r['src']):g}"
    return s


def config_spec(r):
    """The --config of 07_test_lce.py for a row of the grid."""
    s = f"{r['norm']}:{r['alpha']:g}:{r['prune']}"
    if r.get("min_count", 1) and int(r.get("min_count", 1)) > 1:
        s += f":mc{int(r['min_count'])}"
    if r.get("src", 0) and float(r["src"]) > 0:
        s += f":src{float(r['src']):g}"
    return s


def fig_validation(results, rnd):
    g = load_grid(results, "all", src_only=True)
    if g is None:
        print("fig1: no lce_validation.csv: skipped (run 06_validate_lce.py)")
        return
    base = float(g[g["norm"].isna()]["MRR@10"].iloc[0])
    lce = g[g["norm"].notna()]
    best = best_per_norm(g)
    norms = [n for n in NORMS if n in set(lce["norm"])]
    top = float(lce["MRR@10"].max())
    lo = base - 0.012
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels = []
    for x, norm in enumerate(norms):
        sub = lce[lce["norm"] == norm]
        y = sub["MRR@10"].to_numpy()
        shown = y >= lo
        jit = rng.uniform(-0.2, 0.2, len(y))
        ax.scatter(x + jit[shown], y[shown], s=14, color=COLORS[norm], marker=MARKERS[norm], alpha=0.35,
                   linewidths=0, zorder=2)
        b = best[norm]
        ax.scatter([x], [b["MRR@10"]], s=90, color=COLORS[norm], marker=MARKERS[norm], edgecolors="white",
                   linewidths=2, zorder=4)
        ax.annotate(f"{b['MRR@10']:.4f}\n{describe(b)}", (x, b["MRR@10"]), xytext=(0, 9), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5, color=TEXT, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        below = int((~shown).sum())
        labels.append(f"{NORM_NAME[norm]}\n{len(y)} configurations" + (f"\n{below} below the axis" if below else ""))
    ax.axhline(base, color=MUTED, linestyle="--", linewidth=1.2, zorder=3)
    ax.text(1.01, base, f"SPLARE\n{base:.4f}", transform=blended_transform_factory(ax.transAxes, ax.transData),
            ha="left", va="center", fontsize=7.5, color=MUTED)
    ax.set_xticks(range(len(norms)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlim(-0.5, len(norms) - 0.5)
    ax.set_ylim(lo, top + (top - lo) * 0.18)
    nq = (load_json(results / "lce_best.json") or {}).get("n_queries")
    rounds = "src validation: src 0, 0.25, 0.5, 0.75"
    style(ax, "Validation: every LCE configuration, by normalization of M", "",
          f"validation MRR@10{f' ({nq:,} queries)' if nq else ''}",
          subtitle=f"{len(lce)} configurations, {rounds}\nlarge marker = best of the normalization (the one tested); "
                   f"dashed line = SPLARE without expansion")
    save(fig, "fig1_validation")


# ----------------------------------------------------------------------
#  the test result of the best configuration of each normalization
# ----------------------------------------------------------------------
def paired_stats(a, b, iters=10000, seed=0):
    """As 08_significance.py: difference, 95% bootstrap CI, two-sided sign-flip permutation p."""
    rng = np.random.default_rng(seed)
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    n, mean = len(d), d.mean()
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(iters)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    hits, done = 0, 0
    while done < iters:
        c = min(500, iters - done)
        hits += int(np.sum(np.abs((rng.choice([-1.0, 1.0], size=(c, n)) * d).mean(axis=1)) >= abs(mean)))
        done += c
    p = (hits + 1) / (iters + 1)
    return {"A": float(np.mean(a)), "B": float(np.mean(b)), "diff": float(mean), "ci95_lo": float(lo),
            "ci95_hi": float(hi), "p_perm": float(p), "significant": bool(p < 0.05 and (lo > 0 or hi < 0))}


def test_stats(results, name, base):
    """Per-metric statistics of <name>_test vs splare_test: from results/significance/ if 08 was run, else here."""
    sig = load_json(results / "significance" / f"{name}_test__vs__splare_test.json")
    if sig is not None and all(m in sig["metrics"] for m in METRICS if m in base["per_query"]):
        return {m: sig["metrics"][m] for m in METRICS if m in sig["metrics"]}, "08_significance.py"
    run = load_json(results / f"{name}_test.json")
    pa, pb = base["per_query"], run["per_query"]
    ia = {str(q): i for i, q in enumerate(pa["qid"])}
    ib = {str(q): i for i, q in enumerate(pb["qid"])}
    common = sorted(set(ia) & set(ib))
    out = {}
    for m in METRICS:
        if m in pa and m in pb:
            out[m] = paired_stats([pa[m][ia[q]] for q in common], [pb[m][ib[q]] for q in common])
    return out, "computed here (as 08_significance.py)"


def fig_test(results, rnd):
    g = load_grid(results, "all", src_only=True)
    base = load_json(results / "splare_test.json")
    if g is None or base is None:
        print("fig2: no validation grid or no splare_test.json: skipped (run 04, 06, 07)")
        return
    chosen = (load_json(results / "lce_best.json") or {}).get("name")
    best = best_per_norm(g)
    todo, stats = [], {}
    for norm in NORMS:
        if norm not in best:
            continue
        r = best[norm]
        if not (results / f"{r['config']}_test.json").exists():
            print(f"fig{2 + NORMS.index(norm)}_test_{norm}: {r['config']}_test.json not found: skipped. Run:  python -u src/07_test_lce.py "
                  f"--config {config_spec(r)}")
            continue
        stats[norm] = test_stats(results, r["config"], base)
        todo.append(norm)
    if not todo:
        return
    # the same y axis for the four figures
    span = max(max(abs(s["ci95_lo"]), abs(s["ci95_hi"])) for n in todo for s in stats[n][0].values())
    span = max(span, 1e-4) * 1.45
    nq = base["summary"].get("n_queries")
    for norm in todo:
        r, (st, source) = best[norm], stats[norm]
        ms = [m for m in METRICS if m in st]
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        xs = np.arange(len(ms))
        d = np.array([st[m]["diff"] for m in ms])
        err = np.array([[st[m]["diff"] - st[m]["ci95_lo"] for m in ms], [st[m]["ci95_hi"] - st[m]["diff"] for m in ms]])
        ax.bar(xs, d, width=0.56, color=COLORS[norm], zorder=2)
        ax.errorbar(xs, d, yerr=err, fmt="none", ecolor=TEXT, elinewidth=1.2, capsize=4, zorder=3)
        ax.axhline(0, color=MUTED, linewidth=1, zorder=1)
        for x, m in zip(xs, ms):
            s = st[m]
            y = s["ci95_hi"] if s["diff"] >= 0 else s["ci95_lo"]
            dtxt = "0.0000" if abs(s["diff"]) < 5e-5 else f"{s['diff']:+.4f}"
            ax.annotate(f"{dtxt}{' *' if s['significant'] else ''}\np = {s['p_perm']:.3f}", (x, y),
                        xytext=(0, 5 if s["diff"] >= 0 else -6), textcoords="offset points", ha="center",
                        va="bottom" if s["diff"] >= 0 else "top", fontsize=7.5, color=TEXT)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{m}\n{st[m]['A']:.4f} → {st[m]['B']:.4f}" for m in ms], fontsize=8)
        ax.set_ylim(-span, span)
        tag = " — chosen on validation" if r["config"] == chosen else ""
        style(ax, f"Test: SPLARE + LCE, {NORM_NAME[norm]} ({describe(r)}){tag}", "",
              "SPLARE + LCE − SPLARE (test)",
              subtitle=f"{nq:,} test queries; bar = difference, line = 95% bootstrap CI\n"
                       f"p = paired permutation test; * = significant (p < 0.05 and CI excluding 0)")
        save(fig, f"fig{2 + NORMS.index(norm)}_test_{norm}")
        print(f"   {norm}: {r['config']} (statistics: {source})")


# ----------------------------------------------------------------------
#  tables (markdown)
# ----------------------------------------------------------------------
def fmt(v, nd=4):
    return "—" if v is None or v != v else f"{v:.{nd}f}"


def tables(results):
    out = []
    best = load_json(results / "lce_best.json")
    g = load_grid(results, "1")
    if g is not None:
        p = results / "lce_validation_best_per_norm.csv"
        per = pd.read_csv(p) if p.exists() else None
        nv = f"{best['n_queries']:,} " if best else ""
        out += [f"## Validation ({nv}dev queries): best configuration of each normalization", "",
                "| Normalization | α | pruning | MRR@10 | nDCG@10 | R@100 | latents added |", "|---|---|---|---|---|---|---|"]
        if per is not None:
            for r in per.to_dict("records"):
                chosen = best is not None and r["config"] == best["name"]
                name = "none (SPLARE)" if r["norm"] != r["norm"] else NORM_NAME[r["norm"]].replace("|", "\\|")
                name = f"**{name} (chosen)**" if chosen else name
                a = "—" if r["norm"] != r["norm"] else f"{r['alpha']:g}"
                pr = "—" if r["norm"] != r["norm"] else r["prune"]
                out.append(f"| {name} | {a} | {pr} | {fmt(r['MRR@10'])} | {fmt(r['nDCG@10'])} | "
                           f"{fmt(r.get('R@100'))} | {fmt(r['avg_added'], 1)} |")
        out += ["", f"Grid: {len(g) - 1} configurations + SPLARE alone (`results/lce_validation.csv`).", ""]

    rows = []
    sp = load_json(results / "splare_test.json")
    if sp is not None:
        rows.append(("SPLARE", sp["summary"]))
    if best is not None:
        lc = load_json(results / f"{best['name']}_test.json")
        if lc is not None:
            rows.append((f"SPLARE + LCE (`{best['name']}`)", lc["summary"]))
    if rows:
        nt = rows[0][1].get("n_queries")
        out += [f"## Test ({nt:,} dev queries, never used for any choice)", "",
                "| System | " + " | ".join(METRICS) + " | query latents |", "|---|" + "---|" * (len(METRICS) + 1)]
        for name, s in rows:
            out.append(f"| {name} | " + " | ".join(fmt(s.get(m)) for m in METRICS)
                       + f" | {fmt(s.get('avg_query_nnz'), 1)} |")
        sig = load_json(results / "significance" / f"{best['name']}_test__vs__splare_test.json") if best else None
        if sig is not None:
            out.append("| Δ (95% CI) | " + " | ".join(
                (f"{sig['metrics'][m]['diff']:+.4f} [{sig['metrics'][m]['ci95_lo']:+.4f}, "
                 f"{sig['metrics'][m]['ci95_hi']:+.4f}]" + (" *" if sig['metrics'][m]['significant'] else ""))
                if m in sig["metrics"] else "—" for m in METRICS) + " | |")
            out += ["", f"\\* significant: paired permutation test p < {sig['threshold']:.3f} and 95% bootstrap CI "
                    f"excluding 0 ({sig['n_queries']:,} paired queries, {sig['iters']:,} resamples)."]
        out.append("")
    dv = load_json(results / "splare_dev.json")
    if dv is not None:
        s = dv["summary"]
        out += [f"## SPLARE alone on the whole dev ({s['n_queries']:,} queries, for comparison with the published "
                "numbers)", "",
                "| MRR@10 | nDCG@10 | R@1000 | overlap with the provided top-100 |", "|---|---|---|---|",
                f"| {fmt(s.get('MRR@10'))} | {fmt(s.get('nDCG@10'))} | {fmt(s.get('R@1000'))} | "
                f"{fmt(s.get('overlap_groundtruth_100'))} |", ""]
    if not out:
        print("tables: no results yet: skipped")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "tables.md").write_text("# Tables (generated by plots/lce_plots.py)\n\n" + "\n".join(out))
    print(f"written: {OUT_DIR / 'tables.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None, help="results folder (default $WORK_DIR/results)")
    ap.add_argument("--round", default="all", choices=["all", "1"],
                    help="validation grid: both rounds (default) or the first round only")
    args = ap.parse_args()
    results = Path(args.results) if args.results else Path(os.environ.get("WORK_DIR", "work")) / "results"
    if not results.exists():
        raise SystemExit(f"[ERROR] {results} not found: set WORK_DIR or pass --results")
    fig_validation(results, args.round)
    fig_test(results, args.round)
    tables(results)


if __name__ == "__main__":
    main()
