# Results

All numbers come from the per-query files in `results/` and the job logs in `logs/`; `plots/lce_plots.py`
writes the tables below, filled in, to `plots/figures/tables.md`. **To be filled as the runs finish**
(`—` = not run yet).

Protocol: MS MARCO dev small (6,980 queries) split once at random (seed 42) into 1,000 validation and 5,980 test
queries. Everything is chosen on validation; the chosen configuration is evaluated on test once. With ~1.07
relevant passages per query, the standard error of a mean MRR@10 is about 0.012 on 1,000 queries and 0.005 on
5,980 (paired differences between two systems on the same queries are less noisy: that is what `08` tests).

## 1. Data

| | |
|---|---|
| SAE latents of the space | 131,072 (Llama Scope, Llama-3.1-8B, layer 26) |
| latents active in the 8.84M passages | 123,627 in the mapping (`05` prints the exact number) |
| non-zeros per passage / per dev query (mean) | 284.6 / 106.9 |
| overlap with the provided ground truth (dev, top-100 / top-10) | — / — |

## 2. The split

| | validation | test |
|---|---|---|
| queries | 1,000 | 5,980 |
| non-zeros per query (mean / median) | — | — |
| relevant passages per query | — | — |
| md5 of the qid list | — | — |

## 3. The co-occurrence matrix M (Eq. 1)

| | |
|---|---|
| size (active × active) | — |
| off-diagonal non-zeros (density) | — |
| largest count | — |
| storage dense / as sparse CSR | — / — |
| time to build (A100) | — |

## 4. Validation (1,000 queries)

Best configuration of each normalization (`lce_validation_best_per_norm.csv`; the whole grid is in
`lce_validation.csv`).

| Normalization | α | pruning | MRR@10 | nDCG@10 | R@100 | latents added |
|---|---|---|---|---|---|---|
| none (SPLARE) | — | — | — | — | — | 0 |
| raw counts (Eq. 1) | — | — | — | — | — | — |
| P(j \| i) | — | — | — | — | — | — |
| cosine | — | — | — | — | — | — |
| PPMI | — | — | — | — | — | — |

Chosen: —

## 5. Test (5,980 queries, never used for any choice)

| System | MRR@10 | nDCG@10 | R@10 | R@100 | R@1000 | query latents |
|---|---|---|---|---|---|---|
| SPLARE | — | — | — | — | — | — |
| SPLARE + LCE | — | — | — | — | — | — |
| Δ (95% CI) | — | — | — | — | — | |

Paired, two-sided: Δ = (SPLARE + LCE) − SPLARE, 95% bootstrap CI, permutation p-value (10,000 sign flips).
MRR@10 is the primary metric.

Per query (MRR@10): — wins, — ties, — losses. Gain by quartile of the number of latents of the original query:
Q1 —, Q2 —, Q3 —, Q4 —.

## 6. SPLARE alone on the whole dev (for comparison with the published numbers)

| MRR@10 | nDCG@10 | R@1000 |
|---|---|---|
| — | — | — |

This row includes the validation queries, so LCE is never reported on it.

## 7. Examples

From `results/lce_examples_test.txt`: original and added latents of the test queries that gained and lost the
most, and the rank of the first relevant passage before and after LCE.

## 8. Second round (optional, declared): expansion only from strong query latents

Run after the test of the first round was looked at; chosen on validation only (`06 --tag src --after-test`).
Both rounds are reported (168 + 192 configurations). Files: `lce_validation_src.csv`, `lce_best_src.json`.

| src | best configuration | MRR@10 (validation) | vs SPLARE |
|---|---|---|---|
| 0 (all latents, first round) | — | — | — |
| 0.25 | — | — | — |
| 0.5 | — | — | — |
| 0.75 | — | — | — |

Chosen: —  |  test MRR@10: —  |  p(perm): —
