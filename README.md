# SPLARE + Latent Concept Expansion (LCE)

**Training-free query expansion in the latent space of a sparse autoencoder: a corpus co-occurrence matrix of
SPLARE features, and what it does to retrieval on MS MARCO.**

Research code for a project carried out at Sapienza University of Rome, using the computing facilities of
ISTI-CNR. Exact search over the full MS MARCO passage corpus (8.84M passages).

---

## Overview

SPLARE (Formal et al., 2026) maps a text to a sparse vector over the 131,072 latent features of a sparse
autoencoder (Llama Scope, Llama-3.1-8B, layer 26). It has no explicit expansion step. **Latent Concept
Expansion** adds one, with no training and no relevance judgment:

$$M_{ij} = \sum_{d \in C} \mathbb{1}[V_d(i) > 0]\,\mathbb{1}[V_d(j) > 0] \quad (i \neq j) \qquad \text{(Eq. 1, offline)}$$

$$V_{q,ext} = V_q + \alpha \, (A \cdot V_q) \qquad \text{(Eq. 2, online)} \qquad\qquad \hat V_{q,ext} = \text{prune}(V_{q,ext}) \qquad \text{(Eq. 3)}$$

with $A = M$ (Eq. 1 as written) or a normalization of it.

| Choice | Options (all searched on the validation queries) |
|---|---|
| **A** | raw counts (Eq. 1), $P(j \mid i) = M_{ij}/df_i$, cosine $M_{ij}/\sqrt{df_i df_j}$, PPMI |
| **α** | relative: the strongest expansion term gets α × the strongest original weight (raw counts reach millions, so an absolute α is unusable at this scale) |
| **pruning** | `add:n` (keep the original latents + the n strongest added ones), `rtau:x` (keep weights > x · max V_q, Eq. 3 scale-free), also `tau:x` (Eq. 3 literally) |

The original query latents are never pruned, so α = 0 is exactly SPLARE.

## Evaluation protocol

The SPLARE vectors available are those of the MS MARCO passages and of the **6,980 MS MARCO dev small
queries**. These queries are split once, at random (`numpy.random.default_rng(42)`):

| Set | Queries | Used for |
|---|---|---|
| validation | 1,000 | every choice: normalization of M, α, pruning (`06_validate_lce.py`) |
| test | 5,980 | the configuration chosen on validation, evaluated once (`07_test_lce.py`): **the reported result** |

The two sets are disjoint. M is counted on the passages only, so it sees no query. The scripts enforce the order
(the split cannot change once made; the test needs the configuration chosen on validation; validation refuses to
choose again after the test has been run). Metrics: MRR@10 (primary, official MS MARCO), nDCG@10, R@10, R@100,
R@1000, exact search, top 1000. SPLARE and SPLARE + LCE are compared per query with a paired bootstrap and
permutation test (`08_significance.py`).

Results: [`docs/RESULTS.md`](docs/RESULTS.md) (to be filled as the runs finish).

## Repository structure

```
.
├── src/
│   ├── 00_smoke_test.py            # unit tests + full pipeline on synthetic data (CPU, ~3 min)
│   ├── 01_prepare_data.py          # provided Seismic files -> cache (Seismic ids -> SAE latent index)
│   ├── 02_download_msmarco.py      # MS MARCO dev small: qrels and query texts
│   ├── 03_split_dev.py             # dev -> 1,000 validation + 5,980 test (seed 42)
│   ├── 04_baseline.py              # SPLARE without expansion, exact search, check vs the provided ground truth
│   ├── 05_build_cooccurrence.py    # OFFLINE: the exact co-occurrence matrix M of Eq. 1 over 8.84M passages
│   ├── 06_validate_lce.py          # ONLINE: grid on validation -> the chosen configuration
│   ├── 07_test_lce.py              # the chosen configuration on test (+ per-query analysis, examples)
│   ├── 08_significance.py          # paired per-query tests: bootstrap CI + t-test + permutation test
│   │
│   ├── config.py                   # paths (env vars SPLARE_DIR, WORK_DIR), split settings
│   ├── data.py                     # where every vector and query set comes from; the split
│   ├── cooccurrence.py  lce.py     # Eq. 1 (block-wise, exact, memory-mapped) / Eq. 2-3 and the grid search
│   ├── sparse_io.py  retrieval.py  # CSR / Seismic I/O; exact GPU search
│   ├── metrics.py  msmarco.py  pipeline.py  utils.py
│
├── jobs/                           # generic Slurm job scripts (gpu.sbatch, cpu.sbatch)
├── docs/GUIDE.md                   # step-by-step guide: every command, in order
├── docs/RESULTS.md                 # result tables (to be filled)
├── plots/lce_plots.py              # figures and tables, written to plots/figures/
├── splits/                         # the validation / test qids (copied here once made)
├── requirements.txt                # direct dependencies
└── requirements_env.txt            # exact versions of the environment used for all experiments
```

The scripts write under `$WORK_DIR` (caches, the co-occurrence matrix, runs and results). The matrix alone is
~61 GB; nothing of `$WORK_DIR` is distributed except the split.

## Quickstart

```bash
conda create -n lce python=3.11 && conda activate lce
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118   # CUDA 11.8 build used on the cluster
pip install -r requirements.txt
python -u src/00_smoke_test.py             # should end with: ALL TESTS PASSED (15)
```

The pipeline needs the pre-encoded MS MARCO vectors in Seismic format (`documents.bin`, `queries.bin`, ...),
which were provided to us and are **not public**. Paths are read from environment variables:

```bash
export SPLARE_DIR=/path/to/provided/splare/data      # read-only input
export WORK_DIR=/path/to/large/scratch/splare_lce    # caches, M, splits, results (~90 GB)
```

Every long step is a Slurm job. The complete sequence, with the exact commands, is in [`docs/GUIDE.md`](docs/GUIDE.md).

## Acknowledgements

This work builds on SPLARE (Formal, Louis, Déjean and Clinchant, ICLR 2026), SPLADE (Formal, Piwowarski and
Clinchant, SIGIR 2021) and Llama Scope, and uses the MS MARCO passage ranking dataset. Exact search and the data
layout follow the Seismic format.

## License

MIT, see [LICENSE](LICENSE).

## Author

Leonardo Santoro, Sapienza University of Rome.
