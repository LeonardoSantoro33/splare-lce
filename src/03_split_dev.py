"""
03_split_dev.py — Split MS MARCO dev small into VALIDATION (1,000 queries) and TEST (the other 5,980).

Only the dev queries are available in the SPLARE space, so they serve both purposes, on disjoint queries:
    val    every choice of LCE is made here (06_validate_lce.py)
    test   the chosen configuration is evaluated here, once (07_test_lce.py)

Rule (deterministic): the dev qids with a SPLARE vector and at least one relevant passage are sorted in
increasing order and permuted with numpy.random.default_rng(42); the first 1,000 are val, the rest test.

Output in WORK_DIR/splits/: val_qids.txt, test_qids.txt (one qid per line, sorted) and split.json (seed, sizes,
md5 of each list, and a few statistics showing that the two halves are alike). The other scripts check the md5:
once made, the split must never change. Running this again with the same data rewrites nothing; with
different data or parameters it stops unless --overwrite (which invalidates every result computed so far).

    python -u src/03_split_dev.py          # login node, a few seconds
"""
import argparse

import numpy as np

from config import N_VAL, SPLIT_SEED, Paths
from data import compute_split, load_dev_queries, md5_of_ids, read_ids, write_ids
from msmarco import load_dev_qrels
from utils import load_json, log, save_json


def describe(name, ids, q, qrels):
    row_of = {int(x): i for i, x in enumerate(q.ids.tolist())}
    nnz = np.array([q.indptr[row_of[x] + 1] - q.indptr[row_of[x]] for x in ids.tolist()])
    nrel = np.array([sum(1 for r in qrels[x].values() if r > 0) for x in ids.tolist()])
    out = {"n": int(len(ids)), "md5": md5_of_ids(ids), "first_qids": ids[:5].tolist(),
           "query_nnz_mean": float(nnz.mean()), "query_nnz_median": float(np.median(nnz)),
           "relevant_per_query_mean": float(nrel.mean())}
    log(f"  {name:<5} {out['n']:>6,} queries   non-zeros per query: mean {out['query_nnz_mean']:.1f}, "
        f"median {out['query_nnz_median']:.0f}   relevant passages per query: {out['relevant_per_query_mean']:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-val", type=int, default=N_VAL)
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--overwrite", action="store_true", help="replace an existing different split (avoid!)")
    args = ap.parse_args()
    paths = Paths()
    paths.ensure_work_dir()

    q = load_dev_queries(paths)
    qrels = load_dev_qrels(paths)
    val, test, usable = compute_split(q.ids, qrels, args.n_val, args.seed)
    have_vec = set(q.ids.tolist())
    no_vec = len([x for x in qrels if x not in have_vec])
    log(f"dev queries with a SPLARE vector: {q.n_rows:,};  judged in qrels.dev.small.tsv: {len(qrels):,};  "
        f"usable (both): {len(usable):,}" + (f";  judged but without a vector: {no_vec}" if no_vec else ""))
    log(f"split with numpy.random.default_rng({args.seed}): {len(val):,} validation + {len(test):,} test")
    assert len(np.intersect1d(val, test)) == 0 and len(val) + len(test) == len(usable)

    if paths.split_info.exists():
        old = load_json(paths.split_info)
        same = (paths.split_file("val").exists() and paths.split_file("test").exists()
                and np.array_equal(read_ids(paths.split_file("val")), val)
                and np.array_equal(read_ids(paths.split_file("test")), test))
        if same:
            log(f"the split in {paths.splits} already exists and is identical: nothing to do.")
            log(f"  val md5 {old['val']['md5']}   test md5 {old['test']['md5']}")
            return
        if not args.overwrite:
            raise SystemExit(f"[ERROR] {paths.splits} holds a DIFFERENT split (made {old.get('n_val')} / "
                             f"seed {old.get('seed')}). Every result depends on it: use --overwrite only if you "
                             f"really want to start over.")
        log("WARNING: --overwrite: the old split is replaced; results computed with it are no longer valid.")

    write_ids(paths.split_file("val"), val)
    write_ids(paths.split_file("test"), test)
    info = {"seed": args.seed, "n_val": int(len(val)), "n_test": int(len(test)), "n_usable": int(len(usable)),
            "rule": "sorted usable dev qids, numpy.random.default_rng(seed).permutation, first n_val = val",
            "val": describe("val", val, q, qrels), "test": describe("test", test, q, qrels)}
    save_json(info, paths.split_info)
    log(f"saved: {paths.split_file('val')}, {paths.split_file('test')}, {paths.split_info}")
    log("Done.")


if __name__ == "__main__":
    main()
