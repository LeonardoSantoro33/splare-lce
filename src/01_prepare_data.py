"""
01_prepare_data.py — Inspect the provided SPLARE files and build the cache in the SAE latent space.

Input ($SPLARE_DIR, read-only, Seismic format):
    documents.bin, queries.bin            the 8,841,823 passages and the 6,980 dev queries
    doc_ids.npy, queries_ids.npy          MS MARCO pid / qid of each row
    token_to_id_mapping.json              Seismic id -> SAE latent index (keys "0".."131071", sorted alphabetically)
    groundtruth_100.tsv, groundtruth.tsv  exact top-100 / top-10 of each dev query (used by 04 as a check)

What it does:
  1. lists the provided files with their sizes
  2. builds the conversion Seismic id -> SAE latent index from the mapping
  3. reads queries.bin and documents.bin, prints statistics and an example
  4. checks that the groundtruth score is the dot product of the provided vectors
  5. saves documents and dev queries as CSR .npy in WORK_DIR/cache: every other script reads these

No GPU needed. 8.84M documents x 285 non-zeros = 2.5 billion entries: ~30 minutes, request 96G of RAM.
    sbatch --mem=96G -J prep jobs/cpu.sbatch src/01_prepare_data.py
Quick test (saves nothing):
    python -u src/01_prepare_data.py --max-docs 100000 --no-save
"""
import argparse

import numpy as np

from config import N_FEATURES, PROVIDED_FILES, Paths
from sparse_io import (build_numeric_remap, ids_to_rows, load_ids, load_token_mapping, read_groundtruth,
                       read_seismic_bin, remap_columns)
from utils import get_device, human_bytes, log, log_run_header, save_json, timer


def show_vector(csr, i, n=12, title=""):
    idx, vals = csr.row(i)
    order = np.argsort(-vals)[:n]
    log(f"{title} (row {i}, {len(idx)} non-zero entries). Highest weights (SAE latent: weight):")
    log("     " + ", ".join(f"{int(idx[j])}:{vals[j]:.3f}" for j in order))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-docs", type=int, default=0, help="read only the first N documents (test)")
    ap.add_argument("--no-save", action="store_true", help="do not write the cache")
    args = ap.parse_args()
    paths = Paths()
    log_run_header("01_prepare_data.py", args, get_device("cpu"))
    if not args.no_save:
        paths.ensure_work_dir()
    folder = paths.provided

    # 1) files present
    log(f"Provided SPLARE directory: {folder}")
    if not folder.exists():
        raise SystemExit(f"[ERROR] {folder} does not exist: set  export SPLARE_DIR=/path/to/the/provided/files")
    for name in PROVIDED_FILES.values():
        p = folder / name
        log(f"  {name:<28} {'OK ' + human_bytes(p.stat().st_size) if p.exists() else 'MISSING'}")
    others = sorted(x.name for x in folder.iterdir() if x.name not in PROVIDED_FILES.values())
    if others:
        log(f"  other files in the directory (not used): {others}")

    # 2) mapping Seismic id -> SAE latent index
    tokens = load_token_mapping(paths.provided_file("mapping"))
    log(f"Entries in the mapping: {len(tokens):,}. First 10 (alphabetical order): {tokens[:10]}")
    remap = build_numeric_remap(tokens)
    n_cols = max(N_FEATURES, int(remap.max()) + 1)
    log(f"SAE latents: ids {int(remap[remap >= 0].min())}..{int(remap.max())}, space of {n_cols:,} features; "
        f"{len(tokens):,} of them appear in the provided data")
    report = {"n_tokens_mapping": len(tokens), "n_cols": n_cols}

    # 3a) queries
    with timer("reading queries.bin"):
        q_raw = read_seismic_bin(paths.provided_file("queries"))
    q_ids = load_ids(paths.provided_file("query_ids"))
    log(f"queries_ids.npy: {len(q_ids):,} ids (dtype {q_ids.dtype}), first: {q_ids[:5].tolist()}")
    if len(q_ids) != q_raw.n_rows:
        raise SystemExit("[ERROR] number of queries differs between queries.bin and queries_ids.npy")
    q = remap_columns(q_raw, remap, n_cols)
    q.ids = q_ids
    report["queries"] = q.stats()
    log(f"Query statistics: {report['queries']}")
    show_vector(q, 0, title=f"Example query qid={q.ids[0]}")

    # 3b) documents
    with timer("reading documents.bin"):
        d_raw = read_seismic_bin(paths.provided_file("documents"), max_vectors=args.max_docs or None)
    d_ids = load_ids(paths.provided_file("doc_ids"))
    log(f"doc_ids.npy: {len(d_ids):,} ids (dtype {d_ids.dtype}), first: {d_ids[:5].tolist()}")
    if d_ids.dtype.kind in "iu":
        same = np.array_equal(d_ids[: d_raw.n_rows], np.arange(d_raw.n_rows))
        log(f"pid == row number? {same}")
        report["pid_equals_row"] = bool(same)

    # 4) groundtruth: is its score the dot product of the provided vectors? (checked BEFORE remapping)
    gt_path = paths.provided_file("groundtruth")
    if gt_path.exists():
        gt = read_groundtruth(gt_path, q_ids, d_ids)
        k_gt = int(gt.groupby("q").size().max())
        log(f"groundtruth: {len(gt):,} rows, {gt['q'].nunique():,} queries, top-{k_gt}. First rows:\n"
            f"{gt.head(5).to_string()}")
        report["groundtruth_rows"] = int(len(gt))
        r0 = gt.iloc[0]
        if int(r0["d"]) < d_raw.n_rows:
            qi, qv = q_raw.row(int(r0["q"]))
            di, dv = d_raw.row(int(r0["d"]))
            _, ia, ib = np.intersect1d(qi, di, return_indices=True)
            dot = float((qv[ia].astype(np.float64) * dv[ib]).sum())
            log(f"dot-product check: groundtruth={r0['score']:.4f}  recomputed={dot:.4f}")
            report["gt_score_check"] = [float(r0["score"]), dot]
    del q_raw

    d = remap_columns(d_raw, remap, n_cols)
    del d_raw
    d.ids = d_ids[: d.n_rows]
    report["documents"] = d.stats()
    log(f"Document statistics: {report['documents']}")
    show_vector(d, 0, title=f"Example document pid={d.ids[0]}")

    # 5) cache
    if args.no_save:
        log("--no-save: not writing the cache.")
    elif args.max_docs:
        log("--max-docs used: NOT saving the cache (it would be incomplete).")
    else:
        with timer("saving cache"):
            q.save(paths.dev_queries_csr)
            d.save(paths.docs_csr)
            save_json(report, paths.cache / "prepare_report.json")
        lookup = ids_to_rows(d.ids)
        assert lookup(d.ids[:3]).tolist() == [0, 1, 2]
        log(f"Cache saved to {paths.cache}")
    log("Done.")


if __name__ == "__main__":
    main()
