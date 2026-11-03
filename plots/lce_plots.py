"""
lce_plots.py — Figures and tables of the LCE experiments, read from the result files (no number typed by hand).

    python plots/lce_plots.py                      # reads $WORK_DIR/results
    python plots/lce_plots.py --results /path/to/results

Writes to plots/figures/ (next to this script), as .png and .pdf:
  fig1_validation_alpha        validation MRR@10 vs alpha, one line per normalization of M (best pruning at each
                               alpha); dashed line = SPLARE without expansion
  fig2_budget                  validation MRR@10 vs the number of ADDED latents (pruning add:n, best alpha),
                               one line per normalization
  fig3_test_by_query_length    test: gain in MRR@10 of SPLARE + LCE over SPLARE, by quartile of the number of
                               latents of the original query, with the per-query wins / losses
and tables.md: the tables of docs/RESULTS.md (validation per normalization, test, significance), filled in.
A figure whose results are missing is skipped with a message.
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import NullFormatter, NullLocator  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "figures"
NORMS = ["raw", "cond", "cosine", "ppmi"]
NORM_NAME = {"raw": "raw counts (Eq. 1)", "cond": "P(j | i)", "cosine": "cosine", "ppmi": "PPMI"}
# validated categorical palette, fixed order (blue, orange, aqua, yellow): one color per normalization, always
# the same; a different marker per line as well, since aqua and yellow are below 3:1 contrast on white
COLORS = {"raw": "#2a78d6", "cond": "#eb6834", "cosine": "#1baf7a", "ppmi": "#eda100"}
MARKERS = {"raw": "o", "cond": "s", "cosine": "^", "ppmi": "D"}
SERIES_1 = "#2a78d6"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METRICS = ["MRR@10", "nDCG@10", "R@10", "R@100", "R@1000"]


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
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


def log_ticks(ax, values):
    """Log x axis labelled with the actual grid values (0.05, 0.1, ...) instead of powers of ten."""
    ax.set_xscale("log")
    vals = sorted(set(float(v) for v in values))
    ax.set_xticks(vals)
    ax.set_xticklabels([f"{v:g}" for v in vals])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


def load_grid(results):
    p = results / "lce_validation.csv"
    return pd.read_csv(p) if p.exists() else None


def load_json(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def lines_by_norm(ax, g, xcol, base):
    for norm in NORMS:
        sub = g[g["norm"] == norm]
        if sub.empty:
            continue
        best = sub.groupby(xcol)["MRR@10"].max().sort_index()
        ax.plot(best.index, best.values, color=COLORS[norm], marker=MARKERS[norm], linewidth=2, markersize=6,
                label=NORM_NAME[norm])
    ax.axhline(base, color=MUTED, linestyle="--", linewidth=1.2, label="SPLARE alone")
    ax.legend(frameon=False, fontsize=8)


def fig_alpha(results):
    g = load_grid(results)
    if g is None:
        print("fig1: no lce_validation.csv: skipped")
        return
    base = g[g["norm"].isna()]["MRR@10"].iloc[0]
    lce = g[g["norm"].notna()]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    lines_by_norm(ax, lce, "alpha", base)
    log_ticks(ax, lce["alpha"])
    style(ax, "Validation: strength of the expansion", "alpha (relative)", "validation MRR@10")
    save(fig, "fig1_validation_alpha")


def fig_budget(results):
    g = load_grid(results)
    if g is None or not g["prune"].astype(str).str.startswith("add:").any():
        print("fig2: no add:n pruning in the validation grid: skipped")
        return
    base = g[g["norm"].isna()]["MRR@10"].iloc[0]
    add = g[g["prune"].astype(str).str.startswith("add:")].copy()
    add["budget"] = add["prune"].str.split(":").str[1].astype(float)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    lines_by_norm(ax, add, "budget", base)
    log_ticks(ax, add["budget"])
    style(ax, "Validation: how many latents to add", "latents added per query (add:n)", "validation MRR@10")
    save(fig, "fig2_budget")


def fig_query_length(results):
    an = load_json(results / "lce_test_analysis.json")
    if an is None:
        print("fig3: no lce_test_analysis.json (run 07_test_lce.py): skipped")
        return
    groups = [g for g in an["MRR@10"]["by_query_length"] if g["n"]]
    plural = lambda n, w: f"{n} {w}" + ("" if n == 1 else ("es" if w.endswith("s") else "s"))  # noqa: E731
    labels = [f"Q{g['quartile']}: {g['nnz_min']}\u2013{g['nnz_max']} latents\n"
              f"{plural(g['wins'], 'win')}, {plural(g['losses'], 'loss')}" for g in groups]
    gains = [g["gain"] for g in groups]
    fig, ax = plt.subplots(figsize=(1.6 * len(groups) + 1.4, 3.8))
    xs = list(range(len(groups)))
    ax.bar(xs, gains, width=0.6, color=SERIES_1)
    span = max(1e-9, max(abs(v) for v in gains))
    for x, y in zip(xs, gains):
        ax.annotate(f"{y:+.4f}", (x, y), xytext=(0, 4 if y >= 0 else -12), textcoords="offset points",
                    ha="center", fontsize=8, color=TEXT)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    lo, hi = min(0.0, min(gains)), max(0.0, max(gains))
    ax.set_ylim(lo - span * 0.25, hi + span * 0.25)
    style(ax, f"Test: gain of LCE by query length ({an['config']})",
          "quartile of the number of latents of the original query",
          "MRR@10, LCE − SPLARE")
    save(fig, "fig3_test_by_query_length")


# ----------------------------------------------------------------------
#  tables (markdown)
# ----------------------------------------------------------------------
def fmt(v, nd=4):
    return "—" if v is None or v != v else f"{v:.{nd}f}"


def tables(results):
    out = []
    best = load_json(results / "lce_best.json")
    g = load_grid(results)
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
    args = ap.parse_args()
    results = Path(args.results) if args.results else Path(os.environ.get("WORK_DIR", "work")) / "results"
    if not results.exists():
        raise SystemExit(f"[ERROR] {results} not found: set WORK_DIR or pass --results")
    fig_alpha(results)
    fig_budget(results)
    fig_query_length(results)
    tables(results)


if __name__ == "__main__":
    main()
