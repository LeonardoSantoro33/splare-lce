# Step-by-step guide: how every result is produced

This guide lists, in execution order, the commands that produce the results, on a Slurm cluster (A100 nodes,
partition `long`). All commands are run from the repository directory. Result tables are in
[`RESULTS.md`](RESULTS.md).

| Step | What it does | Script | Resources | Time (estimate) |
|---|---|---|---|---|
| 1 | smoke test on synthetic data | `00` | CPU | ~3 min |
| 2 | provided SPLARE vectors → cache | `01` | CPU, 96 GB RAM | ~30 min |
| 3 | MS MARCO dev qrels and query texts | `02` | login node | minutes |
| 4 | split dev: 1,000 validation + 5,980 test | `03` | login node | seconds |
| 5 | SPLARE without expansion, exact search | `04` | GPU | ~15 min |
| 6 | LCE offline: co-occurrence matrix M (Eq. 1) | `05` | GPU, 61 GB disk | ~1–2 h |
| 7 | LCE online: grid on validation → chosen configuration | `06` | GPU | ~2–4 h |
| 8 | the chosen configuration on test | `07` | GPU | ~1 h |
| 9 | paired significance tests | `08` | login node | seconds |
| 10 | figures and tables | `plots/lce_plots.py` | login node | seconds |

The times are estimates: the code has been tested end to end on CPU with synthetic data (step 1).

---

## The protocol in one paragraph

Only MS MARCO dev small is available in the SPLARE space, so its 6,980 queries are split once, at random
(`numpy.random.default_rng(42)`), into **1,000 validation** and **5,980 test** queries. Every choice of LCE (the
normalization of M, α, the pruning) is made on validation (`06`); the chosen configuration is evaluated on test
once (`07`), and the result reported in the paper is that one. The co-occurrence matrix M is counted on the
passages only (no query, no relevance judgment), so it cannot leak the test set. The scripts enforce the order:
the split cannot change once made (`03` stops, the others check an md5), `07` needs the configuration chosen by
`06`, and `06` refuses to choose again once a test result exists.

---

## 0. Before you start (once)

```bash
git clone https://github.com/LeonardoSantoro33/splare-lce.git
cd splare-lce
mkdir -p logs                     # Slurm writes the job logs here
conda activate lce                # environment: requirements.txt (exact versions: requirements_env.txt)
```

**Data directories.** Both paths come from environment variables (defaults in `src/config.py`). Put them in
`~/.bashrc` so they are always set:

```bash
export SPLARE_DIR=/path/to/provided/splare/data     # read only, never modified; not public
export WORK_DIR=/path/to/scratch/$USER/splare_lce   # ~90 GB in the end
```

`WORK_DIR` after all steps: `cache/` (CSR vectors, ~25 GB), `cooccurrence/` (M, ~61 GB), `msmarco/` (dev qrels
and texts), `splits/` (validation and test qids), `runs/` (TREC files), `results/` (one `.json` per system and
query set, with per-query metrics; `summary.csv`).

**What was provided** (SPLARE encodings of MS MARCO, converted to the Seismic format):

| File | Content |
|---|---|
| `documents.bin` | 8,841,823 passages, 284.6 non-zero latents on average (2.5 billion in total, ~20 GB) |
| `queries.bin` | the 6,980 MS MARCO dev queries, 106.9 non-zero latents on average |
| `token_to_id_mapping.json` | 123,627 entries: SAE latent index (as a string, `"60"`) → Seismic id (alphabetical position) |
| `doc_ids.npy`, `queries_ids.npy` | MS MARCO pids / qids as strings (`'0'`, `'1048585'`); pid = row |
| `groundtruth.tsv`, `groundtruth_100.tsv` | top-10 / top-100 of each dev query by exact search: (query row, document row, rank from 1, score) |

We work in the space of the **SAE latent index** (0 … 131,071), not of the Seismic ids.

**Submitting jobs.**

```bash
sbatch --time=HH:MM:SS -J NAME jobs/gpu.sbatch SCRIPT.py --args
sbatch --mem=96G -J NAME jobs/cpu.sbatch SCRIPT.py --args
squeue -u $USER                        # status
tail -f logs/NAME_<jobid>.out          # follow the log (Ctrl+C leaves the job running)
scancel <jobid>
```

**Naming of results.** `results/<system>_<set>.json`, `<system>` = `splare` (no expansion) or
`lce_<norm>_a<alpha>_<pruning>` (e.g. `lce_cond_a0.2_add20`); `<set>` = `val`, `test` or `dev` (all 6,980, only
for SPLARE alone). The validation grid is `results/lce_validation.csv`, the chosen configuration
`results/lce_best.json`.

---

## 1. Smoke test

```bash
srun -p long -c 4 --mem=16G -t 00:30:00 python -u src/00_smoke_test.py
```

The last line must be `ALL TESTS PASSED (15)`. It uses synthetic data only and reproduces the worked example of
Sec. 3.4 of the paper (score 0 → 0.748).

## 2. Provided vectors → cache

```bash
sbatch --mem=96G --time=02:00:00 -J prep jobs/cpu.sbatch 01_prepare_data.py
grep "Entries in the mapping\|SAE latents\|dot-product check\|nnz_per_row_mean\|Cache saved" logs/prep_*.out
```

Check: `Entries in the mapping: 123,627`; in `dot-product check: groundtruth=X recomputed=Y`, X = Y; mean
non-zeros ~106.9 per query and ~284.6 per document.

## 3. MS MARCO dev qrels

```bash
python -u src/02_download_msmarco.py                                  # downloads ~1 GB, keeps two small files
python -u src/02_download_msmarco.py --link-from /folder/with/the/files   # or: link files you already have
```

Check: `qrels.dev.small.tsv 6,980 queries with judgments, 7,437 relevant passages (1.07 per query)`.

## 4. Validation / test split

```bash
python -u src/03_split_dev.py
```

Check: `6,980 usable`, `1,000 validation + 5,980 test`, and similar statistics for the two halves (non-zeros per
query, relevant passages per query). The qids are in `$WORK_DIR/splits/`; copy `val_qids.txt`, `test_qids.txt`
and `split.json` into the repository's `splits/` folder so that the split is published with the code.

## 5. SPLARE without expansion

```bash
sbatch --time=01:00:00 -J base jobs/gpu.sbatch 04_baseline.py
grep "Overlap\|\[splare |" logs/base_*.out
```

Check: overlap with the provided ground truth ~1.00 on every set (differences only from exact score ties). The
`dev` line (6,980 queries) is the number to compare with the published SPLARE result; `val` and `test` are the
references of steps 7 and 8.

## 6. Co-occurrence matrix M

```bash
sbatch --time=06:00:00 -J cooc jobs/gpu.sbatch 05_build_cooccurrence.py
grep "active features\|row block\|M done" logs/cooc_*.out
```

Needs ~61 GB free in `$WORK_DIR` (checked before starting). If the job stops, submit it again: it resumes from the
finished row blocks. `cooccurrence/neighbours.txt` shows the most co-occurring latents of the most frequent ones.

## 7. Validation: choose the configuration

```bash
sbatch --time=06:00:00 -J val jobs/gpu.sbatch 06_validate_lce.py
grep "SPLARE alone\|Best LCE\|    raw\|    cond\|    cosine\|    ppmi\|chosen" logs/val_*.out
```

Default grid: 4 normalizations × α ∈ {0.05, 0.1, 0.2, 0.3, 0.5, 1} × pruning ∈ {add:5, add:10, add:20, add:50,
add:100, rtau:0.05, rtau:0.1} = 168 configurations + SPLARE alone. The best validation MRR@10 (ties: fewer added
latents) goes to `results/lce_best.json`. To widen the grid, rerun with other
`--alphas` / `--prunes`: this is allowed **only before step 8**.

## 8. Test: the final number

```bash
sbatch --time=03:00:00 -J test jobs/gpu.sbatch 07_test_lce.py
grep "\[splare |\|\[lce_\|wins\|quartile\|SPLARE .* -> " logs/test_*.out
```

Runs SPLARE and SPLARE + LCE (the configuration of `lce_best.json`) on the 5,980 test queries in one pass.
`results/lce_test_analysis.json` has the per-query wins / losses and the gain by query length;
`results/lce_examples_test.txt` the original and added latents of the queries that gained and lost the most.

`--config NORM:ALPHA:PRUNE` evaluates a configuration given by hand (ablations): its result is marked
`"selection": "manual (--config)"` and is not the result chosen on validation.

## 9. Significance

```bash
python -u src/08_significance.py
```

SPLARE vs SPLARE + LCE on test: difference, 95% bootstrap CI, paired t-test and permutation test, for MRR@10
(primary), nDCG@10, R@10, R@100, R@1000. Saved to `results/significance/`.

## 10. Figures and tables

```bash
python plots/lce_plots.py
# or as a job (the job scripts add `src/` to the script name, hence `../`):
sbatch -J plots jobs/cpu.sbatch ../plots/lce_plots.py
```

Writes `plots/figures/fig1_validation` (every configuration on validation, by normalization, best of each marked)
and `fig2`–`fig5` (`fig<k>_test_<norm>`: raw, cond, cosine, ppmi) (the best configuration of each normalization on test: difference from SPLARE with 95% CI
and p(perm)), as .png and .pdf, and
`tables.md` (the tables of `RESULTS.md`, filled in from the result files).

---

## 11. Optional second round: expand only from the strong query latents

A declared follow-up, run after the first round. New option `--src S`: the expansion `A·V_q` is computed only from
the query latents with weight >= S x max(V_q); the query itself is unchanged. `S = 0` uses every latent and is
exactly the first round.

Nothing of the first round is touched: the second round writes `lce_validation_<tag>.csv`, `lce_best_<tag>.json`,
... and its names carry `_src<S>` (e.g. `lce_cond_a0.2_add20_src0.5`).

```bash
sbatch --time=03:00:00 -J val2 jobs/gpu.sbatch 06_validate_lce.py --tag src --after-test \
    --src 0 0.25 0.5 0.75 --alphas 0.05 0.1 0.2 0.3 --prunes add:10 add:20 add:50
grep "SECOND ROUND\|configurations\|check against\|Best configuration for each\|chosen" logs/val2_*.out
```

The grid is 4 values of src x 4 norms x 4 alphas x 3 prunings = 192 configurations + SPLARE. The ones with
`src = 0` are first-round configurations: their numbers must be identical (the log prints
`max |MRR@10 difference| = 0.000000`). `--after-test` declares that the test set was already looked at; the
choice is still made on validation only.

```bash
sbatch --time=03:00:00 -J test2 jobs/gpu.sbatch 07_test_lce.py --tag src
python -u src/08_significance.py --tag src
```

If the second round picks the configuration of the first round, `07` says so and tests nothing. In the paper, report
both rounds and the total number of configurations tried (168 + 192).
