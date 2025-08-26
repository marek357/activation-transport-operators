# Activation Transport Operators

This repository implements transport operators for mapping neural network activations between layers, with a focus on analysing how features propagate through transformer models. The work studies how high-quality sparse autoencoder (SAE) features can be "transported" from one layer to another using learned affine transformations. 

[📄 [ArXiv Paper](https://arxiv.org/abs/2508.17540)]
## Overview

Transport operators learn linear mappings that predict downstream layer activations from upstream layer activations. This enables analysis of:
- How semantic features evolve across transformer layers
- Quality of feature representations at different depths
- Causal effects of feature interventions on model behaviour


## Installation

1. Clone the repository

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Pipeline Overview

The repository implements a complete pipeline for studying activation transport:

1. **Activation Collection**: Extract activations from transformer models
2. **Feature Selection**: Identify high-quality SAE features
3. **Feature Explanation**: Get interpretable descriptions of features
4. **Transport Operator Training**: Learn linear mappings between layers
5. **Evaluation**: Assess transport quality and run transport efficiency analysis
6. **Causal Analysis**: Measure causal effects via perplexity evaluation

## Usage Guide

### 1. Collecting Activations

First, collect activations from your target model on a dataset:

```bash
python collect_activations.py --config-path=configs --config-name=activation_collection
```

**Configuration**: Edit `configs/activation_collection.yaml` to specify:
- Model to analyze (default: `gemma_2_2b`)
- Dataset for activation collection (default: `slimpajama`)
- Output directory and batch size
- Number of sequences to process

**Key parameters in `configs/activation_collection/slimpajama.yaml`:**
- `num_sequences`: Number of text sequences to process
- `max_length`: Maximum sequence length
- `batch_size`: Processing batch size
- `output_dir`: Where to save activations

**Output**: Creates a directory with zarr-compressed activation files, e.g., `activations-gemma2-2b-slimpajama-250k/`

### 2. Finding High-Quality Features

Use the automated feature discovery script to identify high-quality SAE features for any layer:

```bash
python scripts/find_feats.py --model-name google/gemma-2-2b --sae-l0 100 --top-percent 5.0
```

**What the script does:**
1. **Loads SAE models** from Gemma Scope for each layer automatically
2. **Computes multiple quality metrics** for each feature:
   - **Token coherence**: Low entropy over tokens that activate the feature (more interpretable)
   - **Vocabulary focus**: How sharply the feature targets specific tokens in vocabulary space
   - **Activation patterns**: Frequency and magnitude of feature activations
   - **Redundancy**: Correlation with other features (lower is better)
   - **Causal effect**: Impact on model predictions when ablated

3. **Combines metrics** into a composite score and ranks features
4. **Saves results** to `feature_scores_{layer}.json` files

**Key parameters:**
- `--model-name`: Transformer model to analyze (default: `google/gemma-2-2b`)
- `--sae-l0`: Target sparsity level for SAE (default: 100)
- `--top-percent`: Percentage of top features to save (default: 5%)
- `--max-tokens`: Number of tokens to analyze for statistics (default: 120k)
- `--candidate-percent`: Percentage sent to causal testing (default: 15%)

**Output format**: Each JSON file contains:
- `"high_quality_feature_ids"`: Ranked list of best feature indices
- `"scores"`: Detailed metrics for all features
- `"meta"`: Analysis parameters and statistics


The script automatically processes all layers and creates comprehensive feature rankings based on interpretability, uniqueness, and causal importance.

### 3. Getting Feature Explanations

Fetch interpretable explanations for your selected features from Neuronpedia:

```bash
python explain_features.py --config-path=configs --config-name=explain_features
```

**Configuration**: Edit `configs/explain_features.yaml` to specify:
- `sae_layer`: Which layer's features to explain
- `feature_ids_file`: Path to JSON file with feature IDs
- `cache_dir`: Directory for caching downloaded explanations
- `output_file`: Where to save the explanations

**Example configuration:**
```yaml
sae_layer: 6
feature_ids_file: "./feature_ids/feature_scores_6.json"
cache_dir: "./explanations_dict_cache"
output_file: "./feature_explanations/feature_explanations_6.json"
```

**Output**: JSON file containing human-readable explanations for each feature.

### 4. Training Transport Operators

Train linear transport operators to map activations between layers:

```bash
python train_transport_operators.py --config-path=configs --config-name=default
```

**Configuration**: Edit `configs/experiment/` files to specify:
- `L`: Source layer numbers (list)
- `k`: Target layer offsets (list) - target layer = L + k
- Training parameters (regularization, method, etc.)

**Key parameters:**
- `method`: Regression method (`"ridge"`, `"linear"`, `"lasso"`, `"elasticnet"`)
- `regularization`: Regularization strength
- `auto_tune`: Whether to use cross-validation for hyperparameter tuning
- `cv_folds`: Number of CV folds for tuning

**Output**: Trained models saved to `cache/transport_model_*.pkl`

**Example experiment configuration:**
```yaml
# configs/experiment/residual_stream.yaml
L: [5, 6, 10]      # Source layers
k: [1, 2, 4, 8]    # Target layer offsets
```

### 5. Evaluating Transport Operators & Matched-Rank Analysis

Evaluate the quality of trained transport operators and run matched-rank analysis:

```bash
python eval.py --config-path=configs --config-name=eval
```

**Configuration**: Edit `configs/eval.yaml` to specify:
- `eval_mode`: `"pretrained"` (use saved models) or `"baselines"` (identity operators)
- `Ls`: Layers to evaluate
- `ks`: Layer offsets to evaluate
- `run_matched_rank`: Whether to run matched-rank analysis
- `matched_rank_only`: Skip other evaluations if true

**Key evaluation metrics:**
- **R² scores**: How well transport operators predict downstream activations
- **Feature-wise metrics**: Performance on individual SAE features
- **Matched-rank analysis**: Compare transported features against all possible features at target layer

**Output**: 
- JSON files with evaluation results in `outputs/`
- Plots and visualizations in `plots/`
- Matched-rank analysis results showing feature similarity rankings

### 6. Causal Perplexity Evaluation

Measure the causal impact of transport interventions on model behavior:

```bash
python causal_perplexity_eval.py --config-path=configs --config-name=causal_eval
```

**This evaluation:**
1. Applies transport operator interventions during model inference
2. Measures changes in model perplexity on text sequences
3. Compares interventions against baselines (no intervention, zero intervention)

**Configuration**: Edit `configs/causal_eval.yaml` and `configs/causal_eval/` subdirectories to specify:
- Intervention layers and target layers
- Text dataset for evaluation
- Number of sequences to evaluate
- Specific features or feature sets to intervene on

**Key intervention types:**
- **Transport interventions**: Replace target layer activations with transported upstream activations
- **Zero interventions**: Zero out specific feature activations
- **Feature-specific interventions**: Target specific SAE features

**Output**: JSON files with perplexity changes, measuring causal effects of different interventions.

## Configuration System

The repository uses Hydra for configuration management. Key configuration directories:

- `configs/model/`: Model specifications (gemma_2_2b, etc.)
- `configs/sae/`: SAE configurations (residual_stream, mlp, etc.)
- `configs/experiment/`: Experiment parameters for training
- `configs/datasets/`: Dataset configurations
- `configs/logger/`: Logging and W&B setup

**Common configuration patterns:**
```bash
# Use different model
python script.py model=gemma_2_9b

# Override specific parameters
python script.py experiment.L=[8,12,16] experiment.k=[2,4]

# Use different SAE type
python script.py sae=mlp

# Change experiment name for tracking
python script.py experiment_name=my_experiment
```

## Example Workflow

Here's a complete example workflow:

```bash
# 1. Collect activations (if not already available)
python collect_activations.py --config-name=activation_collection

# 2. Get feature explanations for layer 6
python explain_features.py sae_layer=6

# 3. Train transport operators for layers 5→6, 6→10
python train_transport_operators.py experiment.L=[5,6] experiment.k=[1,4]

# 4. Evaluate transport quality
python eval.py eval.Ls=[5,6] eval.ks=[1,4] run_matched_rank=true

# 5. Run causal evaluation
python causal_perplexity_eval.py causal_eval=layer_6_to_10
```

## Key Files and Directories

- `collect_activations.py`: Activation extraction from models
- `train_transport_operators.py`: Transport operator training
- `eval.py`: Evaluation and matched-rank analysis
- `causal_perplexity_eval.py`: Causal intervention evaluation
- `explain_features.py`: Feature explanation fetching
- `src/`: Core implementation modules
  - `transport_operator.py`: Transport operator implementation
  - `activation_loader.py`: Activation data loading utilities
  - `sae_loader.py`: SAE loading and configuration
  - `causal_eval/`: Causal evaluation hooks and utilities
- `configs/`: Hydra configuration files
- `cache/`: Cached models and data
- `feature_ids/`: High-quality feature selections per layer
- `outputs/`: Evaluation results and analysis

## Citation

If you use this code in your research, please cite the associated paper:

```
@misc{szablewski2025activationtransportoperators,
      title={Activation Transport Operators}, 
      author={Andrzej Szablewski and Marek Masiak},
      year={2025},
      eprint={2508.17540},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.17540}, 
}
```

## License

[License information to be added]
