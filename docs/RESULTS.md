# Results

All numbers come from the per-query files in `results/` and the job logs in `logs/`; `plots/lce_plots.py`
writes the tables of both rounds, filled in, to `plots/figures/tables.md`. A `—` means that the value is not
reported here; it can be read from the result files.

Protocol: MS MARCO dev small (6,980 queries) split once at random (seed 42) into 1,000 validation and 5,980 test
queries. Everything is chosen on validation; the chosen configuration is evaluated on test once. With ~1.07
relevant passages per query, the standard error of a mean MRR@10 is about 0.012 on 1,000 queries and 0.005 on
5,980 (paired differences between two systems on the same queries are less noisy: that is what `08` tests).

**Summary.** No LCE variant improves on SPLARE, on validation or on test. The configuration chosen on validation
(cosine, α = 0.2, `add:20`) is significantly *worse* on test (MRR@10 −0.0040, p = 0.0008). The configurations
evaluated as a complement are either on par with SPLARE (raw, PPMI) or slightly worse (conditional). The second
round (expansion only from the strongest query latents) does not change this.

## 1. Data

| | |
|---|---|
| SAE latents of the space | 131,072 (Llama Scope, Llama-3.1-8B, layer 26) |
| passages | 8,841,823 |
| latents active in the 8.84M passages | 123,627 in the mapping (`05` prints the exact number) |
| non-zeros per passage / per dev query (mean) | 284.6 / 106.9 |
| overlap with the provided ground truth (dev, top-100 / top-10) | — / — |

## 2. The split

| | validation | test |
|---|---|---|
| queries | 1,000 | 5,980 |
| non-zeros per query (mean / median) | 107.3 / 104 | 106.9 / 104 |
| relevant passages per query | 1.066 | 1.065 |
| md5 of the qid list | `396b933e5e0ef8be3f8ab081971c9c4a` | `76acf2884619ab7170d6b40aef09f71c` |

The lists are in `splits/`.

## 3. The co-occurrence matrix M (Eq. 1)

| | |
|---|---|
| size (active × active) | — |
| off-diagonal non-zeros (density) | — |
| largest count | — |
| storage dense / as sparse CSR | — / — |
| time to build (A100) | — |

## 4. SPLARE alone (no expansion)

| Query set | queries | MRR@10 | use |
|---|---|---|---|
| dev (val + test) | 6,980 | 0.4110 | only to compare with the published SPLARE number (0.408) |
| validation | 1,000 | 0.4047 | reference for choosing the LCE configuration |
| test | 5,980 | 0.4120 | reference to beat |

The dev row includes the validation queries, so LCE is never reported on it.

## 5. First round: validation (1,000 queries)

168 configurations (4 normalizations × 6 values of α × 7 prunings) plus SPLARE. Best configuration of each
normalization (`lce_validation_best_per_norm.csv`; the whole grid is in `lce_validation.csv`):

| Normalization | α | pruning | MRR@10 | nDCG@10 | R@100 | latents added |
|---|---|---|---|---|---|---|
| none (SPLARE) | — | — | 0.4047 | 0.4727 | 0.9255 | 0 |
| raw counts (Eq. 1) | 0.1 | `add:50` | 0.4043 (99.9%) | 0.4725 (99.9%) | 0.9260 (100.1%) | 50 |
| P(j \| i) | 0.2 | `add:20` | 0.4032 (99.6%) | 0.4715 (99.7%) | 0.9260 (100.1%) | 20 |
| cosine | 0.2 | `add:20` | **0.4055 (100.2%)** | **0.4732 (100.1%)** | 0.9230 (99.7%) | 20 |
| PPMI | 0.05 | `add:10` | 0.4040 (99.8%) | 0.4718 (99.8%) | 0.9250 (99.9%) | 10 |

Chosen on validation: **cosine, α = 0.2, `add:20`** (percentages are relative to SPLARE).

## 6. First round: test (5,980 queries, never used for any choice)

The chosen configuration is evaluated on test once. The best configuration of each of the other normalizations is
also evaluated, as a complement (marked *manual*: it was not the validation choice).

| System | MRR@10 | nDCG@10 | R@10 | R@100 | R@1000 |
|---|---|---|---|---|---|
| SPLARE | **0.4120** | **0.4802** | **0.7089** | **0.9304** | **0.9894** |
| SPLARE + LCE, cosine, α = 0.2, `add:20` (chosen) | 0.4080 (99.0%) | 0.4764 (99.2%) | 0.7054 (99.5%) | 0.9279 (99.7%) | 0.9878 (99.8%) |
| SPLARE + LCE, raw, α = 0.1, `add:50` (manual) | 0.4107 (99.7%) | 0.4790 (99.8%) | 0.7077 (99.8%) | 0.9295 (99.9%) | 0.9888 (99.9%) |
| SPLARE + LCE, P(j \| i), α = 0.2, `add:20` (manual) | 0.4087 (99.2%) | 0.4773 (99.4%) | 0.7070 (99.7%) | 0.9281 (99.8%) | 0.9884 (99.9%) |
| SPLARE + LCE, PPMI, α = 0.05, `add:10` (manual) | 0.4120 (100.0%) | 0.4803 (100.0%) | 0.7093 (100.1%) | 0.9300 (100.0%) | 0.9894 (100.0%) |

Paired, two-sided: Δ = (SPLARE + LCE) − SPLARE, 95% bootstrap CI, permutation p-value (10,000 sign flips).
MRR@10 is the primary metric. `*` = p < 0.05 and CI excluding 0, in the direction of a loss.

| System | Δ MRR@10 [95% CI] | p | Δ nDCG@10 | p | Δ R@10 | p | Δ R@100 | p | Δ R@1000 | p |
|---|---|---|---|---|---|---|---|---|---|---|
| cosine (chosen) | −0.0040 [−0.0062, −0.0018] * | 0.0008 | −0.0039 * | 0.0002 | −0.0035 * | 0.0394 | −0.0025 * | 0.0068 | −0.0016 * | 0.0024 |
| raw | −0.0013 [−0.0029, +0.0004] | 0.1313 | −0.0013 | 0.0682 | −0.0013 | 0.3109 | −0.0009 | 0.2065 | −0.0006 | 0.0612 |
| P(j \| i) | −0.0034 [−0.0054, −0.0012] * | 0.0021 | −0.0029 * | 0.0012 | −0.0019 | 0.1918 | −0.0024 * | 0.0052 | −0.0011 * | 0.0148 |
| PPMI | −0.0000 [−0.0010, +0.0009] | 0.9168 | +0.0001 | 0.8961 | +0.0003 | 0.7659 | −0.0004 | 0.4489 | +0.0000 | 1.0000 |

## 7. Second round (declared): expansion only from the strongest query latents

Run after the test of the first round was looked at; chosen on validation only (`06 --tag src --after-test`).
The expansion `A · V_q` is computed only from the query latents with weight ≥ `src` × max(V_q); the query itself is
unchanged, and `src = 0` is the first round. Grid: `src` ∈ {0, 0.25, 0.5, 0.75} × normalization (4) × α ∈
{0.05, 0.1, 0.2, 0.3} × pruning ∈ {`add:5`, `add:10`, `add:20`, `add:50`} = 256 configurations, plus SPLARE as a
check. α ≥ 0.5 and `rtau` were dropped because they were clearly worse on average in the first round. Files:
`lce_validation_src.csv`, `lce_best_src.json`.

Best configuration of each normalization on validation (1,000 queries):

| Normalization | src | α | pruning | MRR@10 | nDCG@10 | R@100 | latents added |
|---|---|---|---|---|---|---|---|
| none (SPLARE) | — | — | — | 0.4047 | 0.4727 | 0.9255 | 0 |
| raw counts (Eq. 1) | 0 | 0.1 | `add:50` | 0.4043 (99.9%) | 0.4725 (99.9%) | 0.9260 (100.1%) | 50 |
| P(j \| i) | 0.75 | 0.2 | `add:5` | 0.4041 (99.8%) | 0.4726 (100.0%) | 0.9240 (99.8%) | 5 |
| cosine | 0 | 0.2 | `add:20` | **0.4055 (100.2%)** | **0.4732 (100.1%)** | 0.9230 (99.7%) | 20 |
| PPMI | 0.75 | 0.05 | `add:50` | 0.4053 (100.1%) | 0.4730 (100.1%) | 0.9260 (100.1%) | 50 |

Chosen on validation: **cosine, α = 0.2, `add:20`, src = 0**, the same configuration as in the first round, whose
test result already exists (§6). For raw and cosine the best configuration has `src = 0`, i.e. the first round, so
no new test was run for them. The two configurations with `src = 0.75` (conditional and PPMI) were evaluated on
test as a complement (*manual*):

| System | MRR@10 | nDCG@10 | R@10 | R@100 | R@1000 |
|---|---|---|---|---|---|
| SPLARE | **0.4120** | **0.4802** | **0.7089** | **0.9304** | **0.9894** |
| SPLARE + LCE, PPMI, src = 0.75, α = 0.05, `add:50` | 0.4116 (99.9%) | 0.4797 (99.9%) | 0.7078 (99.8%) | 0.9304 (100.0%) | 0.9894 (100.0%) |
| SPLARE + LCE, P(j \| i), src = 0.75, α = 0.2, `add:5` | 0.4112 (99.8%) | 0.4796 (99.9%) | 0.7090 (100.0%) | 0.9309 (100.1%) | 0.9891 (100.0%) |

| System | Δ MRR@10 [95% CI] | p | Δ nDCG@10 | p | Δ R@10 | p | Δ R@100 | p | Δ R@1000 | p |
|---|---|---|---|---|---|---|---|---|---|---|
| PPMI, src = 0.75 | −0.0004 [−0.0016, +0.0008] | 0.5268 | −0.0006 | 0.2422 | −0.0012 | 0.2634 | +0.0000 | 1.0000 | −0.0001 | 1.0000 |
| P(j \| i), src = 0.75 | −0.0009 [−0.0025, +0.0008] | 0.2966 | −0.0007 | 0.3294 | +0.0001 | 0.9272 | +0.0005 | 0.5286 | −0.0003 | 0.3668 |

None of the differences is significant: with the source filter, LCE is on par with SPLARE, not better.

## 8. Examples

From `results/lce_examples_test.txt`: original and added latents of the test queries that gained and lost the
most, and the rank of the first relevant passage before and after LCE.
