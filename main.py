from dotenv import load_dotenv
import wandb
import torch
import hydra
import random
import numpy as np
from transformers import set_seed
from omegaconf import DictConfig, OmegaConf
import logging
import os
import socket
from typing import Any, cast

# New: utility to load SAEs
from src.sae_loader import load_sae_from_cfg


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="default"
)
def main(cfg: DictConfig):
    load_dotenv()
    print(socket.gethostname())
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    set_seed(cfg.get("seed", 42))
    wandb.init(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        name=cfg.experiment_name,
        config=cast(dict[str, Any] | None,
                    OmegaConf.to_container(cfg, resolve=True)),
        mode=cfg.logger.wandb_mode  # NOTE: disabled by default
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y‑%m‑%d %H:%M:%S",
    )
    logging.info("Loaded configuration:")
    logging.info(cfg)

    # Load the SAE from Gemma Scope per config
    sae, sae_cfg, log_sparsities = load_sae_from_cfg(cfg)
    logging.info("SAE ready for use.")


if __name__ == '__main__':
    main()
