# Research log: hill-climbing activation transport operators

Goal: improve the quality of the maps between residual-stream layers of
Gemma-2-2B that the paper (arXiv:2508.17540) calls Activation Transport
Operators (ATOs), measured on held-out data with the repo's metrics.

## Setup (fixed for every iteration)

* **Data:** `TheRootOf3/ato-activations`, 1136 SlimPajama sequences × 256
  tokens (251,642 real tokens). Only the needed layers were range-downloaded
  (`research/fetch_layers.py`), and they are stored as fp16 memmaps. The max
  non-BOS |x| is ≤371, so fp16 is exact enough.
* **Split:** the repo's 60/20/20 split by sequence index (train 681 / val 227
  / test 228 sequences, ≈151k / 50k / 50k tokens). All numbers below are on
  **test**. Hyper-parameters are picked on val, or by train-CV for the repo
  baseline.
* **Benchmark pairs:** target layers 10 and 20 (as in the paper), leaps
  k ∈ {1, 5, 9}: (9→10), (5→10), (1→10), (19→20), (15→20), (11→20).
* **Metrics** (`research/lab.py`):
  * `r2_res`: mean over dims of per-dim R². This is the repo's
    `score_residual_default`.
  * `fve`: variance-weighted R² = 1 − Σ‖y−ŷ‖² / Σ‖y−ȳ‖².
  * `feat r2`: the paper's feature-space metric. It is the R² of
    ⟨d_f, y⟩ vs ⟨d_f, ŷ⟩, with d_f the unit-norm Gemma Scope 16k decoder
    direction at the target layer, over tokens where the (JumpReLU) SAE
    fires on the true y. It covers all features with ≥10 activations on
    test (≈7–9k features, not the paper's hand-picked 5%). Reported as the
    median, the mean over R²>−1 (the paper's filter), and the fraction with
    R²>0.9.
  * Every metric is reported on **all** positions (repo protocol) and on
    **nobos** (position 0 removed).

## Observations about the data that drive the hypotheses

* Position 0 is always `<bos>` (id 2), and its residual is a **constant**
  vector: the std across sequences is 0.014 and the norm is ≈1623, against
  ≈130 for ordinary tokens. That makes 681 identical, extreme-leverage rows
  in the repo's training set, and it inflates the "all" metrics.
* Position 1 (the first real token) is also an outlier, with a mean norm of
  ≈427.
* Gemma 2 sublayers read `RMSNorm(x)` and write through a post-norm, so the
  layer update depends (almost) only on the direction x̂ = x/rms(x), not on
  ‖x‖.

## Iterations

(appended below as experiments finish)

### Iteration 0: reproduce the repo baseline (`exp0_baseline.py`)

The repo ridge is sklearn `Ridge(fit_intercept=True)`, with α picked by
5-fold contiguous CV (r2 scoring) over the repo grid {0.1…10⁴}, trained on
all train positions including BOS. Here it is reproduced exactly from
fold-wise sufficient statistics.

* At 9→10 it matches the paper: median feature R² is 0.951 on all
  positions, i.e. "small leaps transport many features with R²>0.95".
* At every k ≥ 5 pair, and at 19→20, **CV picks α = 10⁴, the edge of the
  grid**. The grid is too narrow.
* **BOS inflates the paper-style metric.** At 5→10 the median feature R²
  is 0.317 on all positions but −0.077 without BOS: features that fire on
  the huge constant BOS vector get enormous between-token variance. From here
  the primary protocol is **nobos**.
* Large leaps have positive FVE (0.43–0.65) but strongly negative median
  feature R² (down to −1.9). Feature R² is computed only on tokens where a
  feature fires. A weak linear predictor regresses toward the unconditional
  mean, so it under-predicts exactly there.

### Iteration 1: linear hypotheses (`exp1_linear.py`)

α is chosen on **val** by FVE from a relative grid c·mean-eig(Σ_xx). The
fit uses non-BOS tokens only, and BOS is predicted by lookup (it is
constant).

| mean over 6 pairs (test, nobos) | FVE | median feat R² |
|---|---|---|
| repo ridge | 0.6432 | −0.211 |
| H1: fit without BOS (shrink→0) | 0.6441 | −0.264 |
| H2: + identity prior | 0.6443 | −0.228 |
| **H3: + norm-aware [x, x/rms(x)]** | **0.6526** | **−0.118** |

* **H1 (BOS)** is neutral on FVE and *negative* on features. Choosing α by
  validation FVE gave more shrinkage (c≈0.03–0.1) than the repo's
  α=10⁴. That supports the regression-to-the-mean explanation, and exp1b
  tests it directly.
* **H2 (identity prior)** is a small, consistent gain. With 151k tokens
  for 2304 dims, the prior matters little. Under Gaussian inputs OLS equals
  the average Jacobian (Stein's lemma), so the data already pins the
  "Jacobian-lens" map down.
* **H3 (norm-aware)** improves FVE and median feature R² on **all 6
  pairs**. The gains grow with the leap: +0.002 FVE at k=1, +0.011–0.018
  at k=5–9, and +0.10–0.26 median feature R² at k=9. This matches the
  architecture: every Gemma-2 sublayer sees RMSNorm(x) and writes through a
  post-norm, so the update is ≈ a function of x̂. Note that the H3 run also
  picked smaller α (c≈1e-3).

### Iteration 2: is the gap cross-token? (`exp2_context.py`, diagnostic)

This adds source-layer context to the norm-aware map (closed-form, α on
val): x_{t−1}, and the causal mean of x over positions 1…t.

| test, nobos | 9→10 FVE | 9→10 median feat R² | 5→10 FVE | 5→10 median feat R² |
|---|---|---|---|---|
| x, x̂ | 0.8779 | 0.9267 | 0.6548 | 0.0205 |
| + x_{t−1} | **0.8800** | **0.9305** | **0.6569** | **0.0514** |
| + x_{t−1} + causal mean | 0.8795 | 0.9283 | 0.6566 | 0.0057 |

* The previous token gives a small, real gain: +0.002 FVE and +0.03 median
  feature R².
* The causal mean adds nothing. Validation picks heavy regularization
  (c=0.1) once it is included.
* **Linearly-accessible cross-token information is small.** Attention is
  content-addressed, so it can still matter nonlinearly. But the cheap,
  linear j-policy extensions are not where the remaining 12–35% of
  variance is. Same-token nonlinearity (iteration 3) is tested next.
* This direction is set aside: it changes the operator's semantics (no
  longer same-token transport) for a small gain.
