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

### Iteration 3: transport + synthesis, a nonlinear correction in a neuron-like basis (`exp3_nonlinear.py`)

ŷ = [A B]·[x; x̂] + b (the frozen iteration-1 map) + W_dec·σ(W_enc x̂ + b_enc).
The correction is trained with Adam on the linear map's residual, on the
train split. The epoch is chosen by validation FVE. 4 epochs, CPU.

**5→10 sweep** (test, nobos):

| correction (≈ size) | FVE | median feat R² | frac R²>0.9 | s/epoch |
|---|---|---|---|---|
| none (linear, iteration 1) | 0.6548 | 0.021 | 0.178 | – |
| dense GELU h=2048 | 0.6653 | 0.174 | 0.189 | 33 |
| GeGLU h=1365 (Gemma's MLP form, matched params) | 0.6620 | 0.126 | 0.184 | ~45 |
| sign-split top-k, 1024 enc / 2048 poles, k=64 | 0.6598 | 0.137 | 0.184 | ~28 |
| **top-k ReLU h=2048, k=64** | 0.6644 | 0.178 | 0.188 | 28 |
| **top-k ReLU h=8192, k=64** | **0.6698** | **0.247** | **0.196** | 84 |

* **Small FVE gains, large feature-metric gains.** A correction worth
  +0.010–0.015 FVE moves the median feature R² by +0.15 to +0.23. That
  confirms the regression-to-the-mean diagnosis: the linear map
  under-predicts on exactly the tokens where a feature fires, and a
  nonlinear unit that "switches on" fixes this.
* **A sparse code is enough.** Top-k with 64 of 2048 units active matches
  dense GELU, and a wider dictionary with the same k keeps improving. This
  agrees with Shabalin & Piotrowski (arXiv:2610.24521), who find that
  transcoder-style sparse units ≈ thresholded neurons, and with the
  "networks are already clean" view. The synthesis between layers is
  carried by a few sparsely active, neuron-like directions.
* GeGLU (gated) and sign-split poles did *not* help at matched size and 4
  epochs. Sign-split halves the number of encoders, and the gated product
  starts slowly from small init.
* All models overfit after 2–3 epochs (151k training tokens). Data, not
  capacity, is the binding constraint, so more activations would help most.

### Iteration 1b: choose α for the feature metric? (`exp1b_featsel.py`)

The same norm-aware identity-prior map, with α picked on val by median
feature R² instead of FVE.

| mean over 6 pairs (test, nobos) | FVE | median feat R² | frac R²>0.9 |
|---|---|---|---|
| α by val FVE (iteration 1) | **0.6526** | −0.118 | 0.226 |
| α by val median feat R² | 0.6489 | **−0.091** | **0.228** |

* The feature metric always prefers the **smallest** α in the grid
  (c = 1e-5 to 1e-4), while FVE prefers c ≈ 1e-3 to 3e-3. This confirms
  the shrinkage / regression-to-the-mean mechanism of iteration 0. It is a
  genuine trade-off: −0.004 FVE for +0.027 median feature R².
* The effect is small next to the nonlinear correction (+0.23 at 5→10).
  So the repo operator keeps FVE selection by default, and the
  `rel_alphas` grid lets a user pick the feature-oriented setting.

### Iteration 4: optimize the paper's metric directly? Feature-aware loss and SAE-code metrics (`exp3 --feat_lambda`, `exp6_codeeval.py`)

The loss adds λ·Σ_{f active on y} (⟨d_f, y−ŷ⟩²/var_f) / L0 to the residual
MSE. This is the paper's per-feature R² made differentiable. Because that
metric only scores tokens where a feature *truly* fires, it cannot see a
feature being switched on where it should be off. A new guard metric
re-encodes ŷ with the target SAE and compares codes:
precision/recall/F1 of the active set and the FVE of the code
activations ("code FVE").

5→10, test, nobos:

| operator | residual FVE | median feat R² | code F1 | code prec. | code recall | code FVE | pred L0 (true ≈79) |
|---|---|---|---|---|---|---|---|
| identity | −0.104 | −1.051 | 0.326 | 0.506 | 0.241 | −0.138 | 37.9 |
| linear (iteration 1) | 0.655 | 0.021 | 0.492 | **0.834** | 0.349 | 0.664 | 33.2 |
| + GELU 2048 (MSE) | 0.665 | 0.175 | 0.517 | 0.816 | 0.379 | 0.689 | 36.9 |
| + top-k 2048 (MSE) | 0.664 | 0.179 | 0.516 | 0.814 | 0.377 | 0.688 | 36.9 |
| + top-k 8192 (MSE) | **0.670** | 0.247 | 0.528 | 0.804 | 0.393 | **0.698** | 38.8 |
| + top-k 2048, feature loss λ=1 | 0.623 | 0.492 | **0.535** | 0.745 | **0.417** | 0.691 | 44.5 |
| + top-k 2048, feature loss λ=3 | 0.563 | **0.550** | 0.512 | 0.730 | 0.394 | 0.675 | 42.9 |

* **The linear map's errors are mostly missing features, not wrong
  ones.** Precision is 0.83 and recall 0.35, with 33 features predicted
  per token against 79 true ones. Shrinkage leaves many features just
  below their JumpReLU thresholds. This is the same regression-to-the-mean
  effect seen from a third angle.
* **Every correction raises recall**, and with it F1 and code FVE.
* **λ=1 is a real improvement in feature transport.** Against its
  same-size MSE twin it has higher code F1 (0.535 vs 0.516) and code FVE
  (0.691 vs 0.688). The cost is residual MSE, presumably in directions the
  SAE does not represent.
* **λ=3 Goodharts the paper's metric.** Median feature R² rises to 0.55,
  but code F1 and code FVE fall below the MSE model. **The
  active-tokens-only feature R² should not be reported alone. Pair it with
  a false-positive-aware measure such as code F1 or code FVE.**
* The default operator stays MSE-trained (best residual FVE and code FVE).
  The feature loss at λ≈1 is an option for when the goal is to reproduce
  downstream SAE features.

### Iteration 3b: the best correction on all six pairs (top-k ReLU, 8192 units, k=64, 3 epochs)

Test, nobos. Repo → norm-aware linear → + sparse correction:

| pair | FVE | median feature R² | frac R²>0.9 |
|---|---|---|---|
| 9→10 | 0.875 → 0.878 → **0.884** | 0.916 → 0.927 → **0.942** | 0.540 → 0.568 → **0.617** |
| 5→10 | 0.648 → 0.655 → **0.670** | −0.077 → 0.021 → **0.247** | 0.171 → 0.178 → **0.196** |
| 1→10 | 0.430 → 0.448 → **0.473** | −1.896 → −1.634 → **−1.424** | 0.013 → 0.020 → **0.025** |
| 19→20 | 0.883 → 0.887 → **0.898** | 0.920 → 0.927 → **0.948** | 0.564 → 0.586 → **0.662** |
| 15→20 | 0.578 → 0.589 → **0.633** | −0.177 → −0.100 → **0.246** | 0.004 → 0.005 → **0.037** |
| 11→20 | 0.445 → 0.460 → **0.523** | −0.949 → −0.845 → **−0.298** | 0.000 → 0.000 → **0.004** |
| **mean** | **0.643 → 0.653 → 0.680** | **−0.211 → −0.118 → +0.110** | **0.215 → 0.226 → 0.257** |

* The correction helps every pair, and helps **later layers more**.
  Targeting layer 20 from 11 or 15, it adds +0.04 to +0.06 FVE and
  +0.35 to +0.55 median feature R². Targeting layer 10 from 1 or 5, it adds
  +0.015 to +0.025 FVE. This fits the paper's "evict and synthesise"
  regime in later layers: more of what layer 20 holds is computed in
  between, and a sparse same-token code can learn a good part of it.
* 1→10 stays hard. The layer-1 residual barely encodes what 9 layers of
  attention will add.

### Iteration 5: which SAE features move between layers? (`exp5_feature_transport.py`, `build_explorer_data.py`)

This uses Gemma Scope 16k residual SAEs at both ends of each pair. For
each downstream feature j that fires ≥20 times on train and test:

* **Oracle partner.** The upstream feature with the highest activation
  correlation on train (75k tokens). It is scored by its correlation on
  test.
* **Matching rules that use only weights.**
  * `dec-cos`: raw decoder cosine (the identity-map assumption of prior
    work, e.g. Balagansky et al. 2024; Laptev et al. 2025).
  * `ato-dec`: cosine after pushing upstream decoders through the
    linearised operator A + B/rms.
  * `+csls`: hubness-corrected versions of both.
  * `ato-enc`: predicted contribution to j's encoder pre-activation.

Agreement with the oracle on test:

| pair | oracle median test corr | partner corr>0.8 | dec-cos | **ato-dec** | dec-cos+CSLS | ato-dec+CSLS | ato-enc |
|---|---|---|---|---|---|---|---|
| 9→10 | 0.661 | 34.8% | 80.8% | **81.3%** | 80.5% | 80.7% | 70.0% |
| 5→10 | 0.396 | 17.8% | 61.2% | **63.2%** | 61.5% | 62.3% | 37.0% |
| 1→10 | 0.189 | 11.2% | 30.6% | **36.9%** | 30.6% | 34.9% | 14.0% |
| 19→20 | 0.781 | 47.1% | 85.3% | **85.5%** | 85.1% | 85.5% | 84.6% |
| 15→20 | 0.403 | 13.3% | 58.6% | **60.9%** | 58.6% | 59.7% | 27.6% |
| 11→20 | 0.325 | 11.1% | 50.2% | **53.3%** | 49.5% | 51.0% | 14.2% |

* **Pushing decoders through the operator is the best weights-only
  matching rule on every pair.** Its advantage over raw decoder cosine
  grows with the leap and with depth: +0.2 points at k=1, +2 at k=5,
  +3 to +6 at k=9. The residual basis rotates, and the ATO accounts for it.
* CSLS hubness correction does not help. Encoder-side scoring is poor:
  JumpReLU encoders are not aligned with decoders, and the score favours
  high-norm hubs.

**Taxonomy of target-layer features.** Classes use test R² of the linear
operator, test R² of the linear operator plus the top-k 8192 correction,
and the oracle partner's test correlation:

| pair | n | transported | assembled | synthesized | partial | unexplained |
|---|---|---|---|---|---|---|
| 9→10 | 13,254 | 34.1% | 35.8% | 0.8% | 18.2% | 11.1% |
| 5→10 | 13,254 | 14.9% | 8.5% | 2.8% | 12.9% | 60.8% |
| 1→10 | 13,254 | 5.7% | 1.5% | 0.4% | 9.5% | 82.9% |
| 19→20 | 13,629 | 44.2% | 32.2% | 1.7% | 15.6% | 6.3% |
| 15→20 | 13,629 | 2.0% | 1.7% | **17.5%** | 16.9% | 61.8% |
| 11→20 | 13,629 | 0.1% | 0.1% | **14.4%** | 3.8% | 81.7% |

* **Adjacent layers mostly carry features.** 70–76% of features are
  transported 1:1 or linearly assembled from the previous layer.
* **Two regimes at the same leap size (k=5).** Into layer 10, 15% of
  features are still carried 1:1 from layer 5 and only 3% are synthesized.
  Into layer 20, only 2% are carried from layer 15, and 17.5% are
  synthesized: same-token computation that a 64-of-8192 sparse code
  reproduces with R² ≥ 0.5. This is a feature-level, quantitative version
  of the paper's "accept new features early / evict and synthesise late"
  picture.
* **Most features at distance ≥5 are "unexplained" (61–83%).** No
  same-token map, linear or sparse-nonlinear, reaches R² 0.5. These are
  the candidates for attention-written, context-dependent features. The
  paper's richer j-policies are aimed at exactly this, and our linear
  context probe (iteration 2) showed it is not linearly accessible.
* Qualitative check with Neuronpedia labels (the `report/` explorer).
  Transported pairs are near-identical concepts: L9 "the word 'at.'" →
  L10 "the word 'at'"; L5 "interleukins and cytokines" → L10 the same.
  Assembled features combine related upstream concepts, e.g. L10
  "governors" from L5 "positions of authority within government" and
  neighbours.

### Iteration 6: SAE-code metrics on all pairs (`exp6_codeeval.py`)

Does the best operator (linear + top-k 8192 correction, MSE-trained)
reproduce the downstream SAE code? Test, nobos.

| pair | code F1, linear → +corr | precision | recall | code FVE, linear → +corr |
|---|---|---|---|---|
| 9→10 | 0.756 → **0.767** | 0.886 → 0.880 | 0.660 → 0.680 | 0.882 → **0.891** |
| 5→10 | 0.492 → **0.528** | 0.834 → 0.804 | 0.349 → 0.393 | 0.664 → **0.698** |
| 1→10 | 0.292 → **0.313** | 0.760 → 0.749 | 0.181 → 0.198 | 0.444 → **0.486** |
| 19→20 | 0.827 → **0.837** | 0.888 → 0.884 | 0.773 → 0.794 | 0.905 → **0.919** |
| 15→20 | 0.551 → **0.601** | 0.821 → 0.790 | 0.415 → 0.485 | 0.576 → **0.661** |
| 11→20 | 0.400 → **0.485** | 0.791 → 0.747 | 0.267 → 0.358 | 0.408 → **0.530** |
| **mean** | **0.553 → 0.588** | | | **0.646 → 0.698** |

* The gains are not an artifact of the active-tokens-only feature R². Code
  F1 and code FVE improve on every pair, again most in the late,
  synthesis-heavy pairs.
* For every operator, errors are dominated by **missed** features
  (recall 0.18–0.79) rather than spurious ones (precision 0.75–0.89).
  Transported vectors are "quieter" than real ones. The predicted L0 is
  19–68 against a true L0 of ≈70–80.

### Iteration 4b: the feature-loss sweet spot (5→10, test, nobos)

| correction | residual FVE | median feat R² | code F1 | precision | recall | code FVE |
|---|---|---|---|---|---|---|
| top-k 2048, MSE | 0.664 | 0.179 | 0.516 | 0.814 | 0.377 | 0.688 |
| top-k 2048, λ=0.3 | 0.652 | 0.397 | 0.537 | 0.770 | 0.412 | 0.695 |
| top-k 2048, λ=1 | 0.623 | 0.492 | 0.535 | 0.745 | 0.417 | 0.691 |
| top-k 2048, λ=3 | 0.563 | **0.550** | 0.512 | 0.730 | 0.394 | 0.675 |
| top-k 8192, MSE | **0.670** | 0.247 | 0.528 | 0.804 | 0.393 | **0.698** |
| top-k 8192, λ=1 | 0.626 | 0.491 | **0.538** | 0.739 | **0.422** | 0.694 |

* **λ≈0.3 is the best trade-off.** It doubles the median feature R² of
  the MSE model (0.18 → 0.40), and improves code F1 (+0.021) and code FVE
  (+0.007) over its MSE twin. Residual FVE costs only −0.013, and it stays
  ≈ the linear map's.
* Raising λ further keeps inflating the paper's metric. Code F1 and FVE
  peak at λ ∈ [0.3, 1], and residual FVE drops steadily.
* Width helps the MSE objective more than the feature objective.

### Iteration 7: width / sparsity of the sparse correction (15→20, MSE, 3 epochs)

| width h | k | FVE | median feat R² | frac R²>0.9 | relative cost |
|---|---|---|---|---|---|
| (linear only) | – | 0.589 | −0.100 | 0.005 | – |
| 8192 | 32 | 0.628 | 0.212 | 0.032 | 1× |
| 8192 | 64 | 0.633 | 0.246 | 0.037 | 1× |
| 8192 | 128 | 0.636 | 0.268 | 0.039 | ≈1× (sparse decoder) |
| 16384 | 64 | **0.639** | **0.284** | **0.045** | 2× (dense encoder) |

* Capacity has not saturated in either direction. k is nearly free to
  raise, because the decoder is an `embedding_bag` over k rows. Width
  doubles the dense encoder cost.
* With 151k training tokens, all of these still peak at epoch 2–3, so a
  larger activation set is the most promising next lever.
