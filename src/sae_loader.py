import logging
from typing import Any, Dict, Optional, Tuple

import torch
from sae_lens import SAE

try:
    from omegaconf import OmegaConf
except Exception:  # noqa: BLE001
    OmegaConf = None  # type: ignore[assignment]


def pick_device(preferred: Optional[str] = None) -> str:
    """Choose a device string. Honors a preferred device if valid.
    Order: preferred -> cuda -> mps -> cpu.
    """
    if preferred is not None:
        p = preferred.lower()
        if p in {"auto", "any", "default"}:
            preferred = None  # fall through to auto-selection
        elif p.startswith("cuda") and torch.cuda.is_available():
            return "cuda"
        elif (
            p in {"mps", "metal"}
            and getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ):
            return "mps"
        elif p == "cpu":
            return "cpu"
        else:
            logging.warning(
                f"Preferred device '{preferred}' not available; auto-selecting."
            )

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_sae_from_cfg(cfg) -> Tuple[SAE, Dict[str, Any], Optional[torch.Tensor]]:
    """Load a Gemma Scope SAE using fields under cfg.sae.

    Supports release (explicit) or release_template (templated via ${model.size}).
    """
    if not hasattr(cfg, "sae"):
        raise ValueError("Config missing 'sae' section.")

    release = cfg.sae.release
    sae_id = cfg.sae.sae_id
    force_download = bool(cfg.sae.force_download)
    preferred_device = cfg.sae.device

    if not sae_id:
        raise ValueError("cfg.sae.id must be set (e.g. 'layer_20/width_16k/canonical')")

    device = pick_device(preferred_device)
    logging.info(
        f"Loading SAE from release='{release}', id='{sae_id}' on device='{device}'..."
    )

    sae, sae_cfg, log_sparsities = SAE.from_pretrained_with_cfg_and_sparsity(
        release=release,
        sae_id=sae_id,
        device=device,
        force_download=force_download,
    )

    # Basic summary
    n_feats = getattr(sae.cfg, "d_sae", None) or getattr(sae, "d_sae", None)
    hook_name = getattr(getattr(sae, "cfg", None), "metadata", None)
    hook_name = getattr(hook_name, "hook_name", None)
    logging.info(
        f"Loaded SAE: features={n_feats}, hook='{hook_name}', arch='{sae.cfg.architecture() if hasattr(sae, 'cfg') else 'unknown'}'"
    )

    if log_sparsities is not None:
        try:
            avg_l0 = float(torch.exp(log_sparsities).mean().item())
            logging.info(f"Average L0 (approx): {avg_l0:.2f}")
        except Exception:  # noqa: BLE001
            pass

    return sae, sae_cfg, log_sparsities
