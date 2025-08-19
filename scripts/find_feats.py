#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from transformer_lens import HookedTransformer
from datasets import load_dataset
# ---------------------------
# Utils
# ---------------------------


def auto_device(choice: str = "auto") -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_tokens(tokens: torch.Tensor, max_ctx: int, stride: int):
    """
    Yield overlapping chunks of token ids for evaluation.
    tokens: [T]
    """
    T = tokens.shape[0]
    if T <= max_ctx:
        yield tokens.unsqueeze(0)
        return
    start = 0
    while start < T - 1:
        end = min(start + max_ctx, T)
        yield tokens[start:end].unsqueeze(0)
        if end == T:
            break
        start = end - stride


def safe_softmax_entropy(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    p = x.float().log_softmax(dim=dim).exp()
    logp = x.float().log_softmax(dim=dim)
    ent = -(p * logp).sum(dim=dim)
    return ent


# ---------------------------
# SAE Loader (no SAELens required)
# ---------------------------


def load_gemma_scope_sae(
    layer: int, width: str = "16k", stream: str = "resid_pre", l0_target: int = 100
):
    """
    Load a Gemma Scope SAE from HuggingFace.

    Args:
        layer: Layer number (0-25 for Gemma-2-2B)
        width: SAE width, either "16k" or "65k"
        stream: Stream type, one of "resid_pre", "resid_post", "mlp_pre_act", "mlp_post_act"
        l0_target: Target L0 sparsity (will pick closest available)

    Returns:
        dict: State dict with SAE weights
    """
    # Map stream names to Gemma Scope repository names
    stream_to_repo = {
        "resid_pre": "google/gemma-scope-2b-pt-res",
        "resid_post": "google/gemma-scope-2b-pt-res",
        "mlp_pre_act": "google/gemma-scope-2b-pt-mlp",
        "mlp_post_act": "google/gemma-scope-2b-pt-mlp",
    }

    if stream not in stream_to_repo:
        raise ValueError(
            f"Unsupported stream: {stream}. Choose from: {list(stream_to_repo.keys())}"
        )

    repo_id = stream_to_repo[stream]

    print(f"[info] Loading SAE from {repo_id} for layer {layer}, width {width}")

    try:
        from huggingface_hub import list_repo_files
        import numpy as np

        # List available files for this layer and width
        all_files = list_repo_files(repo_id)
        layer_files = [
            f
            for f in all_files
            if f.startswith(f"layer_{layer}/width_{width}/") and f.endswith(".npz")
        ]

        if not layer_files:
            raise ValueError(
                f"No SAE files found for layer {layer}, width {width} in {repo_id}"
            )

        print(
            f"[info] Available L0 values: {[f.split('/')[-2].replace('average_l0_', '') for f in layer_files]}"
        )

        # Find the file with L0 closest to target
        def extract_l0(filename):
            try:
                return int(filename.split("/")[-2].replace("average_l0_", ""))
            except Exception:
                return float("inf")

        best_file = min(layer_files, key=lambda f: abs(extract_l0(f) - l0_target))
        actual_l0 = extract_l0(best_file)

        print(f"[info] Selected SAE with L0={actual_l0} (target was {l0_target})")
        print(f"[info] Downloading: {best_file}")

        # Download the file
        sae_path = hf_hub_download(
            repo_id=repo_id, filename=best_file, local_files_only=False
        )

        # Load the weights from npz
        data = np.load(sae_path, allow_pickle=True)
        state_dict = {}

        for key in data.keys():
            state_dict[key] = torch.from_numpy(data[key]).float()

        return state_dict

    except Exception as e:
        print(f"[error] Failed to load SAE: {e}")
        print(
            f"[info] Available layers: 0-25, widths: 16k/65k, streams: {list(stream_to_repo.keys())}"
        )
        raise


class SAE:
    """
    Minimal SAE wrapper:
      z = relu(x @ W_enc + b_enc)  # [*, d_feat]
      recon = z @ W_dec.T + b_dec  # [*, d_model]
    """

    def __init__(self, state_dict: dict, d_model: int, d_feat: int):
        # Try many common key names
        cand = {
            "W_enc": ["W_enc", "encoder.weight", "W_enc.weight", "W_e", "W_in"],
            "b_enc": ["b_enc", "encoder.bias", "W_enc.bias", "b_e", "b_in"],
            "W_dec": ["W_dec", "decoder.weight", "W_dec.weight", "W_d", "W_out"],
            "b_dec": ["b_dec", "decoder.bias", "W_dec.bias", "b_d", "b_out"],
        }

        def pick(keys):
            for k in keys:
                if k in state_dict:
                    return state_dict[k]
            return None

        W_enc = pick(cand["W_enc"])
        b_enc = pick(cand["b_enc"])
        W_dec = pick(cand["W_dec"])
        b_dec = pick(cand["b_dec"])

        if W_enc is None or W_dec is None:
            raise ValueError(
                "Could not find W_enc/W_dec in SAE checkpoint. Keys found: "
                + str(list(state_dict.keys()))
            )

        self.W_enc = W_enc.float()
        self.W_dec = W_dec.float()
        self.b_enc = b_enc.float() if b_enc is not None else torch.zeros(d_feat)
        self.b_dec = b_dec.float() if b_dec is not None else torch.zeros(d_model)

        # Sanity checks / transpose if needed
        # Expect shapes: W_enc: [d_model, d_feat]  or [d_feat, d_model]
        #                W_dec: [d_feat, d_model]  or [d_model, d_feat]
        if self.W_enc.shape == (d_feat, d_model):  # [feat, model] -> transpose
            self.W_enc = self.W_enc.T
        if self.W_dec.shape == (d_model, d_feat):  # [model, feat] -> transpose
            self.W_dec = self.W_dec.T

        assert self.W_enc.shape == (d_model, d_feat), (
            f"W_enc wrong shape {self.W_enc.shape} expected {(d_model, d_feat)}"
        )
        assert self.W_dec.shape == (d_feat, d_model), (
            f"W_dec wrong shape {self.W_dec.shape} expected {(d_feat, d_model)}"
        )
        assert self.b_enc.shape[0] == d_feat, f"b_enc wrong shape {self.b_enc.shape}"
        assert self.b_dec.shape[0] == d_model, f"b_dec wrong shape {self.b_dec.shape}"

        # Precompute decoder norms for speed
        self.dec_col_norm = self.W_dec.norm(dim=1)  # [d_feat]

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d_model]
        z = (x @ self.W_enc) + self.b_enc
        return F.relu(z)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # z: [..., d_feat]
        return (z @ self.W_dec) + self.b_dec

    @torch.no_grad()
    def feature_contribution(self, z: torch.Tensor, f_idx: int) -> torch.Tensor:
        """
        Returns decoded contribution of one feature: z_f * W_dec[f]
        z: [..., d_feat]
        out: [..., d_model]
        """
        wf = self.W_dec[f_idx]  # [d_model]
        zf = z[..., f_idx]  # [...]
        return zf.unsqueeze(-1) * wf

    def to(self, device):
        """Move SAE to device"""
        self.W_enc = self.W_enc.to(device)
        self.W_dec = self.W_dec.to(device)
        self.b_enc = self.b_enc.to(device)
        self.b_dec = self.b_dec.to(device)
        self.dec_col_norm = self.dec_col_norm.to(device)
        return self


# ---------------------------
# Metrics
# ---------------------------


@torch.no_grad()
def compute_logit_focus(
    W_dec: torch.Tensor, W_U: torch.Tensor, topk: int = 20, batch_size: int = 1024
):
    """
    For each feature f, compute vocab-space vector v_f = W_U^T @ W_dec[f]
    Return:
      focus_score[f] = (top1_logit - mean_rest) / std_rest  (a simple "peakiness" z-score)
      and entropy over softmaxed logits as secondary measure (lower = sharper).
    Shapes:
      W_dec: [d_feat, d_model]
      W_U:   [d_model, vocab]
    """
    d_feat = W_dec.shape[0]
    device = W_dec.device

    z_peak = torch.zeros(d_feat, device=device)
    ent = torch.zeros(d_feat, device=device)

    # Process features in batches to save memory
    for start_idx in range(0, d_feat, batch_size):
        end_idx = min(start_idx + batch_size, d_feat)
        W_dec_batch = W_dec[start_idx:end_idx]  # [batch_size, d_model]

        # v: [batch_size, vocab]
        v = W_dec_batch @ W_U

        # compute top1 z-score style metric
        top_vals, _ = v.topk(k=1, dim=1)
        mean = v.mean(dim=1, keepdim=True)
        std = v.std(dim=1, unbiased=False, keepdim=True) + 1e-6
        z_peak_batch = ((top_vals - mean) / std).squeeze(1)  # [batch_size]

        # entropy
        ent_batch = safe_softmax_entropy(v, dim=1)  # [batch_size]

        z_peak[start_idx:end_idx] = z_peak_batch
        ent[start_idx:end_idx] = ent_batch

    return z_peak, ent


def normalize_array(x: np.ndarray, invert: bool = False, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    if invert:
        x = -x
    mn, mx = np.nanmin(x), np.nanmax(x)
    if mx - mn < eps:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn + eps)


# ---------------------------
# Hook helpers
# ---------------------------


def resid_hook_name(layer_idx: int, stream: str):
    """
    stream in {"resid_pre", "resid_post", "mlp_pre_act", "mlp_post_act"}
    """
    if stream == "resid_pre":
        return f"blocks.{layer_idx}.hook_resid_pre"
    elif stream == "resid_post":
        return f"blocks.{layer_idx}.hook_resid_post"
    elif stream == "mlp_pre_act":
        return f"blocks.{layer_idx}.mlp.hook_pre"
    elif stream == "mlp_post_act":
        return f"blocks.{layer_idx}.mlp.hook_post"
    else:
        raise ValueError(
            f"Unsupported stream: {stream}. Choose from: resid_pre, resid_post, mlp_pre_act, mlp_post_act"
        )


# ---------------------------
# Evaluation pass to collect stats
# ---------------------------


@torch.no_grad()
def collect_feature_stats(
    model: HookedTransformer,
    sae: SAE,
    eval_sequences: list,  # list of torch.Tensor sequences
    hook_name: str,
    max_ctx: int,
    stride: int,
    device: torch.device,
    vocab_size: int,
    sample_positions_per_chunk: int = 2048,
):
    """
    Returns dict of per-feature stats gathered over streamed data.
    """
    d_feat = sae.W_dec.shape[0]
    act_count = np.zeros(d_feat, dtype=np.int64)
    act_sum_abs = np.zeros(d_feat, dtype=np.float64)
    token_counter = defaultdict(lambda: defaultdict(int))
    topk_per_feat = 32

    # For redundancy: collect a small random sketch of z to estimate correlations
    # We'll downsample features for the correlation baseline to save memory.
    rnd = np.random.default_rng(0)
    sample_feat = min(512, d_feat)
    feat_sample_idx = torch.tensor(
        sorted(rnd.choice(d_feat, size=sample_feat, replace=False).tolist()),
        dtype=torch.long,
    )

    # Sum of (z - mean) cross-products requires two passes; we do a simpler proxy:
    # collect mean and std per feature, and pairwise corr with sampled subset.
    # We'll collect covariance proxy: E[z_f * z_g] over random positions
    eg_zfzg = torch.zeros(sample_feat, d_feat, device=device)
    cnt_positions = 0

    hook_handle = None

    def cache_resid_hook(resid, hook):
        # resid: [batch, pos, d_model]
        # We return resid unmodified during stats collection.
        hook.ctx["resid"] = resid.detach()

    hook_handle = model.add_hook(hook_name, cache_resid_hook)

    total_tokens = sum(seq.numel() for seq in eval_sequences)
    processed = 0

    # Process each sequence individually
    for seq_idx, sequence in enumerate(
        tqdm(eval_sequences, desc="Processing sequences")
    ):
        # Process sequence in chunks if it's longer than max_ctx
        for chunk in chunk_tokens(sequence, max_ctx, stride):
            chunk = chunk.to(device)
            model(chunk)
            resid = model.hook_dict[hook_name].ctx["resid"]  # [1, L, d_model]
            z = sae.encode(resid)  # [1, L, d_feat]

            # Basic stats
            z_abs = z.abs()
            fired = (z > 0).sum(dim=(0, 1))  # [d_feat]
            act_count += fired.cpu().numpy()
            act_sum_abs += z_abs.sum(dim=(0, 1)).cpu().numpy()

            # Token coherence proxy: record tokens corresponding to top activations
            # For efficiency, sample a subset of positions
            L = chunk.shape[1]
            if L > 0:
                pos_idx = torch.randint(
                    0, L, (min(sample_positions_per_chunk, L),), device=device
                )
                z_samp = z[0, pos_idx]  # [S, d_feat]
                tok_samp = chunk[0, pos_idx]  # [S]
                # For each feature, take topk positions in this sample and record tokens
                # We'll do this by taking per-feature topk over S positions.
                S = z_samp.shape[0]
                if S > 0:
                    k = min(topk_per_feat, S)
                    top_vals, top_pos = torch.topk(
                        z_samp.transpose(0, 1), k=k, dim=1
                    )  # [d_feat, k]
                    toks = tok_samp[top_pos]  # [d_feat, k]
                    # Count token frequencies per feature (sparse)
                    toks_cpu = toks.cpu().numpy()
                    for f in range(d_feat):
                        for t in toks_cpu[f]:
                            token_counter[f][int(t)] += 1

                # Redundancy proxy: E[z_f * z_g] across sampled positions
                z_samp2 = z_samp  # [S, d_feat]
                z_samp2 = z_samp2 / (z_samp2.std(dim=0, keepdim=True) + 1e-6)
                eg_zfzg += (
                    z_samp2[:, feat_sample_idx].transpose(0, 1)  # [S, sample_feat]
                    @ z_samp2
                ) / z_samp2.shape[0]  # [sample_feat, d_feat]
                cnt_positions += 1

            processed += chunk.shape[1]

    print(f"[info] Processed {processed} tokens out of {total_tokens} total.")
    # Remove hook
    model.reset_hooks()

    # proportion of positions firing
    activation_rate = act_count / max(1, processed)
    mean_abs_activation = act_sum_abs / max(1, processed)

    # Token entropy per feature
    token_entropy = np.zeros(d_feat, dtype=np.float64)
    for f in tqdm(range(d_feat), desc="Token entropy"):
        # gather counts for this feature
        counts = list(token_counter[f].values())
        if not counts:
            # very high (bad/unknown)
            token_entropy[f] = math.log(vocab_size + 1)
        else:
            arr = np.array(counts, dtype=np.float64)
            p = arr / arr.sum()
            ent = -(p * np.log(p + 1e-12)).sum()
            token_entropy[f] = ent

    print(
        f"[info] Activation rate: {activation_rate.mean():.4f}, "
        f"mean abs activation: {mean_abs_activation.mean():.4f}, "
        f"token entropy: {token_entropy.mean():.4f}"
    )
    # Redundancy: for each feature, max correlation with sampled features
    # Convert eg_zfzg to correlations (already standardized along positions).
    if cnt_positions > 0:
        corr = (eg_zfzg / cnt_positions).abs()  # [sample_feat, d_feat]
        max_corr = corr.max(dim=0).values.clamp_(0, 1).cpu().numpy()
    else:
        max_corr = np.zeros(d_feat, dtype=np.float64)

    return {
        "activation_rate": activation_rate,
        "mean_abs_activation": mean_abs_activation,
        "token_entropy": token_entropy,
        "redundancy_maxcorr": max_corr,
    }


# ---------------------------
# Causal ablation score
# ---------------------------


@torch.no_grad()
def causal_effect_scores(
    model: HookedTransformer,
    sae: SAE,
    hook_name: str,
    probe_token_batches: list,  # list[Tensor[B, L]]
    candidate_features: np.ndarray,  # [K]
    device: torch.device,
):
    """
    For each candidate feature f, measure average Δloss when subtracting its decoded contribution.
    Returns: np.ndarray [d_feat] with zeros for non-candidates.
    """
    d_feat = sae.W_dec.shape[0]
    effect = np.zeros(d_feat, dtype=np.float64)

    # Pre-capture baseline losses for each batch
    baseline_losses = []
    for toks in probe_token_batches:
        toks = toks.to(device)
        out = model(toks)
        # token-level NLL; take mean over tokens excluding BOS
        loss = model.loss_fn(out, toks).mean().item()
        baseline_losses.append(loss)

    def make_ablate_hook(f_idx):
        def hook_fn(resid, hook):
            # resid: [B, L, d_model]
            z = sae.encode(resid)  # [B, L, d_feat]
            contrib = sae.feature_contribution(z, f_idx)  # [B, L, d_model]
            return resid - contrib  # subtract this feature's decoded effect

        return hook_fn

    for f in tqdm(candidate_features.tolist(), desc="Causal ablation"):
        deltas = []
        h = model.add_hook(hook_name, make_ablate_hook(int(f)))
        try:
            for i, toks in enumerate(probe_token_batches):
                toks = toks.to(device)
                out = model(toks)
                loss = model.loss_fn(out, toks).mean().item()
                deltas.append(loss - baseline_losses[i])
        finally:
            # Use reset_hooks to clear all hooks instead of individual hook removal
            model.reset_hooks()
        effect[f] = float(np.mean(deltas)) if deltas else 0.0

    return effect


# ---------------------------
# Main
# ---------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="TransformerLens model name, e.g., 'pythia-2.8b', 'gemma-2-2b', etc.",
    )
    ap.add_argument(
        "--sae-path",
        type=str,
        default="",
        help="Path to SAE checkpoint .pt/.bin with W_enc/W_dec etc. If empty, will load from Gemma Scope.",
    )
    ap.add_argument(
        "--sae-l0",
        type=int,
        default=100,
        help="Target L0 sparsity for Gemma Scope SAE (will pick closest available). Only used if --sae-path is empty.",
    )
    # ap.add_argument("--hook-layer", type=int, required=True,
    #                 help="Layer index where the SAE was trained (e.g., 10).")
    # ap.add_argument("--hook-stream", type=str, default="resid_post",
    #                 choices=["resid_pre", "resid_post", "mlp_pre_act", "mlp_post_act"])
    # ap.add_argument("--eval-text-file", type=str, default="",
    #                 help="Text file for evaluation. If omitted, uses small built-in prompts.")
    # ap.add_argument("--extra-prompts", type=str, nargs="*", default=[],
    #                 help="Optional extra short prompts to append to eval text.")
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=120_000,
        help="Upper bound of tokens to process for stats.",
    )
    ap.add_argument(
        "--ctx-window",
        type=int,
        default=1024,
        help="Context window used for chunking during eval.",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=256,
        help="Stride between chunks to avoid overcounting.",
    )
    ap.add_argument(
        "--top-percent",
        type=float,
        default=5.0,
        help="Select top X%% by composite score.",
    )
    ap.add_argument(
        "--candidate-percent",
        type=float,
        default=15.0,
        help="Percent of features to send to causal testing.",
    )
    ap.add_argument(
        "--causal-probe-prompts",
        type=str,
        nargs="*",
        default=[
            "The capital of France is",
            "In Python, a list comprehension",
            "The derivative of x^2 is",
            "Translate to Spanish: good morning",
        ],
    )
    ap.add_argument(
        "--device", type=str, default="auto", help="'auto', 'cuda', 'cpu', or 'mps'"
    )
    args = ap.parse_args()

    device = auto_device(args.device)
    print(f"[info] Using device: {device}")

    # Load model
    print("[info] Loading model...")
    # For Gemma-2-2B, we can use the HuggingFace model name directly
    if args.model_name in ["gemma-2-2b", "google/gemma-2-2b"]:
        model = HookedTransformer.from_pretrained(
            "google/gemma-2-2b", device=str(device)
        )
    else:
        model = HookedTransformer.from_pretrained(args.model_name, device=str(device))
    model.eval()

    d_model = model.cfg.d_model
    vocab_size = model.cfg.d_vocab
    print(f"[info] d_model={d_model}, vocab={vocab_size}")
    # Prepare evaluation tokens
    print("[info] Preparing evaluation sequences...")
    dataset = load_dataset(
        "DKYoon/SlimPajama-6B", split="train", streaming=True, cache_dir="./cache"
    )
    # num_samples = 75
    num_samples = 75_000
    dataset_iterator = iter(dataset)
    eval_sequences = []
    total_tokens = 0

    try:
        for i in tqdm(range(num_samples), desc="Loading sequences"):
            sample = next(dataset_iterator)
            tokens = model.to_tokens(
                sample["text"], prepend_bos=True, truncate=True
            ).squeeze(0)
            # Limit individual sequence length
            max_seq_len = min(args.ctx_window, 256)
            if tokens.numel() > max_seq_len:
                tokens = tokens[:max_seq_len]

            if tokens.numel() >= 10:  # Only keep sequences with reasonable length
                eval_sequences.append(tokens)
                total_tokens += tokens.numel()

                # Stop if we've reached our token limit
                if total_tokens >= args.max_tokens:
                    break

    except StopIteration:
        print(f"[warning] Dataset ended early after {i} samples")

    print(
        f"[info] Loaded {len(eval_sequences)} sequences with {total_tokens} total tokens"
    )

    with torch.no_grad():
        for layer_idx in tqdm(range(model.cfg.n_layers)):
            # print(resid_hook_name(layer_idx, 'resid_post'))

            hook_name = resid_hook_name(layer_idx, "resid_post")
            if hook_name not in model.hook_dict:
                print(
                    f"Model {args.model_name} does not have a hook for layer {layer_idx} and stream resid_post. "
                    "Check the model configuration or use a different hook stream."
                )
                # print available hooks
                available_hooks = [
                    k
                    for k in model.hook_dict.keys()
                    if k.startswith(f"blocks.{layer_idx}.")
                ]
                print(hook_name in available_hooks)
                # print(
                #     f"[info] Available hooks for layer {layer_idx}: {available_hooks}")
                continue
            # Load SAE
            print("[info] Loading SAE weights...")
            # if args.sae_path and os.path.exists(args.sae_path):
            #     # Load from local file
            #     sd = torch.load(args.sae_path, map_location=device)
            #     # if it's a wrapper dict with 'state_dict' field, unwrap
            #     if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            #         sd = sd["state_dict"]
            # else:
            #     # Load from Gemma Scope
            sd = load_gemma_scope_sae(
                layer=layer_idx, width="16k", stream="resid_post", l0_target=args.sae_l0
            )

            # guess d_feat from known dims
            d_feat = None
            for v in sd.values():
                if isinstance(v, torch.Tensor):
                    if v.dim() == 2 and (
                        v.shape[0] == d_model or v.shape[1] == d_model
                    ):
                        other = v.shape[0] if v.shape[1] == d_model else v.shape[1]
                        if other >= 1024:  # heuristic
                            d_feat = other
                            break
            if d_feat is None:
                raise ValueError(
                    "Could not infer SAE feature dimension from checkpoint."
                )
            print(f"[info] SAE feature width (inferred): {d_feat}")

            sae = SAE(sd, d_model=d_model, d_feat=d_feat)

            # Move SAE to the same device as the model
            sae.to(device)

            # Collect stats
            print(f"[info] Collecting feature stats for layer {layer_idx}...")
            stats = collect_feature_stats(
                model=model,
                sae=sae,
                eval_sequences=eval_sequences,
                hook_name=hook_name,
                max_ctx=args.ctx_window,
                stride=args.stride,
                device=device,
                vocab_size=vocab_size,
            )

            activation_rate = stats["activation_rate"]
            mean_abs_activation = stats["mean_abs_activation"]
            token_entropy = stats["token_entropy"]
            redundancy = stats["redundancy_maxcorr"]

            # Logit focus (model unembed)
            print("[info] Computing logit focus...")
            # TransformerLens unembed: W_U: [d_model, d_vocab]
            W_U = model.W_U.detach().to(device)

            # Use smaller batch size for MPS to avoid memory issues
            batch_size = 512 if device.type == "mps" else 512

            try:
                z_peak, ent_vocab = compute_logit_focus(
                    sae.W_dec, W_U, topk=20, batch_size=batch_size
                )
                z_peak = z_peak.cpu().numpy()
                ent_vocab = ent_vocab.cpu().numpy()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(
                        "[warning] GPU out of memory during logit focus computation. Try using --device cpu or reducing --sae-width"
                    )
                    print("[info] Falling back to CPU for logit focus computation...")
                    # Move to CPU for this computation
                    # Create a CPU copy
                    sae_cpu = SAE(sd, d_model=d_model, d_feat=d_feat)
                    W_U_cpu = model.W_U.detach().cpu()
                    z_peak, ent_vocab = compute_logit_focus(
                        sae_cpu.W_dec, W_U_cpu, topk=20, batch_size=1024
                    )
                    z_peak = z_peak.numpy()
                    ent_vocab = ent_vocab.numpy()
                else:
                    raise

            # Build composite score
            print("[info] Scoring features...")
            # Lower entropy is better; redundancy lower is better; higher activation (but not too high) is okay.
            # Normalize each piece to 0..1
            # low entropy -> high score
            coh_token = normalize_array(token_entropy, invert=True)
            focus_vocab = normalize_array(z_peak, invert=False)

            # Penalize redundancy & extreme sparsity/deadness
            red_pen = normalize_array(redundancy, invert=False)
            # lower rate -> higher penalty (invert later)
            dead_pen = normalize_array(activation_rate, invert=True)

            # Initial score without causal
            score_pre = (
                0.40 * coh_token + 0.35 * focus_vocab - 0.15 * red_pen - 0.10 * dead_pen
            )

            # Choose candidates for causal testing
            K = max(32, int(len(score_pre) * (args.candidate_percent / 100.0)))
            candidate_features = np.argpartition(-score_pre, K)[:K]
            candidate_features = candidate_features[
                np.argsort(-score_pre[candidate_features])
            ]

            # Make probe token batches
            probe_batches = []
            for s in args.causal_probe_prompts:
                t = model.to_tokens(s, prepend_bos=True)
                probe_batches.append(t)

            # Causal effect
            print("[info] Running causal ablations on candidate features...")
            effects = causal_effect_scores(
                model=model,
                sae=sae,
                hook_name=hook_name,
                probe_token_batches=probe_batches,
                candidate_features=candidate_features,
                device=device,
            )
            # we treat positive Δloss as useful
            effects = np.maximum(effects, 0.0)
            # Normalize causal only over candidates to keep scale sane
            if effects[candidate_features].max() > 0:
                causal_norm = np.zeros_like(effects)
                cn = normalize_array(effects[candidate_features], invert=False)
                causal_norm[candidate_features] = cn
            else:
                causal_norm = np.zeros_like(effects)

            # Final score
            score = (
                0.35 * coh_token
                + 0.25 * focus_vocab
                + 0.30 * causal_norm
                - 0.07 * red_pen
                - 0.03 * dead_pen
            )

            # Select top percent
            pct = float(args.top_percent)
            # pct = max(0.1, float(args.top_percent))
            cutoff_idx = int(len(score) * (pct / 100.0))
            top_ids = np.argpartition(-score, cutoff_idx)[:cutoff_idx]
            top_ids = top_ids[np.argsort(-score[top_ids])]

            # Persist metrics
            out = {
                "high_quality_feature_ids": top_ids.tolist(),
                "scores": {
                    "final_score": score.tolist(),
                    "coherence_token_entropy_inv": coh_token.tolist(),
                    "focus_vocab_peak": focus_vocab.tolist(),
                    "causal_effect_norm": causal_norm.tolist(),
                    "redundancy_maxcorr": redundancy.tolist(),
                    "activation_rate": activation_rate.tolist(),
                    "mean_abs_activation": mean_abs_activation.tolist(),
                    "entropy_vocab": ent_vocab.tolist(),
                    "raw_token_entropy": token_entropy.tolist(),
                },
                "meta": {
                    "model_name": args.model_name,
                    "sae_path": args.sae_path,
                    "hook_layer": layer_idx,
                    "hook_stream": "resid_post",
                    "eval_tokens": int(total_tokens),
                    "eval_sequences": len(eval_sequences),
                    "candidate_features": candidate_features.tolist(),
                    "top_percent": pct,
                    "candidate_percent": float(args.candidate_percent),
                },
            }

            with open(f"feature_scores_{layer_idx}.json", "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)

            # Print the Python list so you can copy-paste immediately
            high_quality_feature_ids = out["high_quality_feature_ids"]
            print("\n# ------------------------------")
            print("# high_quality_feature_ids")
            print("# ------------------------------")
            print(high_quality_feature_ids)
            print(f"\n[info] Wrote ./feature_scores_{layer_idx}.json")


if __name__ == "__main__":
    main()
