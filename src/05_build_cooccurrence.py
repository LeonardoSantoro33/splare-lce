"""
05_build_cooccurrence.py — OFFLINE phase of LCE (Sec. 3.1): the Latent Co-occurrence Matrix M (Eq. 1).

M is counted once over the 8.84M MS MARCO passages and never changes. It uses the DOCUMENTS ONLY: no query
(neither val nor test) and no relevance judgment, so it cannot leak anything from the test set.

    M_ij = number of passages in which SAE latents i and j are both active   (i != j; the diagonal keeps df_i)

Output in WORK_DIR/cooccurrence/:
    M.npy           (n_active x n_active) float32, exact integer counts, memory-mapped by 06/07
    active.npy      SAE latent of each row/column of M (latents active in at least one passage)
    df.npy          document frequency of every latent (full space, 131,072)
    meta.json       sizes and statistics (density, largest count, dense vs sparse storage)
    neighbours.txt  the most co-occurring latents of the most frequent ones (to look at M)

Cost (A100 40 GB): ~123.6k active latents -> ~61 GB on disk, about 1-2 hours.
The job resumes from the finished row blocks if it is interrupted (just submit it again).

    sbatch --time=06:00:00 -J cooc jobs/gpu.sbatch src/05_build_cooccurrence.py
"""
import argparse

import numpy as np

from config import Paths
from cooccurrence import Cooccurrence, build_cooccurrence
from data import load_docs
from utils import get_device, human_bytes, log, log_run_header


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row-block", type=int, default=0, help="rows of M per block (0 = fit the GPU)")
    ap.add_argument("--doc-chunk", type=int, default=0, help="documents per matmul (0 = 8192 on GPU)")
    ap.add_argument("--max-docs", type=int, default=0, help="TEST ONLY: random sample of documents")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true", help="rebuild from scratch")
    ap.add_argument("--neighbours", type=int, default=30, help="latents shown in neighbours.txt")
    args = ap.parse_args()
    paths = Paths()
    paths.ensure_work_dir()
    dev = get_device()
    log_run_header("05_build_cooccurrence.py", args, dev)

    docs = load_docs(paths, mmap=True)
    rows = None
    out_dir = paths.cooc
    if args.max_docs:
        rows = np.random.default_rng(args.seed).choice(docs.n_rows, args.max_docs, replace=False)
        out_dir = paths.work / f"cooccurrence_sample{args.max_docs}"
        log(f"TEST: M on a sample of {args.max_docs:,} documents, saved to {out_dir} (not used by 06/07)")
    meta = build_cooccurrence(docs, out_dir, dev, args.row_block, args.doc_chunk, rows, args.overwrite)
    log(f"dense M: {human_bytes(meta['bytes_dense'])}; as a sparse CSR it would take "
        f"~{human_bytes(meta['bytes_sparse_csr_estimate'])}")

    # a look at M: neighbours of the most frequent latents
    cooc = Cooccurrence(out_dir)
    top = cooc.active[np.argsort(-cooc.df_active)[: args.neighbours]]
    lines = [f"# the {len(top)} most frequent SAE latents and their most co-occurring ones "
             f"({cooc.n_docs:,} documents). Format: latent (df): neighbour count, ..."]
    for f in top.tolist():
        nb = cooc.neighbours(f, 8)
        lines.append(f"{f} (df {int(cooc.df_full[f]):,}): " + ", ".join(f"{j} {int(c):,}" for j, c in nb))
    (out_dir / "neighbours.txt").write_text("\n".join(lines) + "\n")
    for ln in lines[:6]:
        log(ln[:200])
    log("Done.")


if __name__ == "__main__":
    main()
