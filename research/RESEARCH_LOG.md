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
