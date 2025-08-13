import logging
import random
import socket
from typing import Any, cast
from src.transport_operator import TransportOperator
from src.activation_loader import get_train_val_test_datasets

import hydra
import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="default")
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
        config=cast(dict[str, Any] | None, OmegaConf.to_container(cfg, resolve=True)),
        mode=cfg.logger.wandb_mode,  # NOTE: disabled by default
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("Loaded configuration:")
    logging.info(cfg)

    transport_operators = {}
    for L in cfg.experiment.L:
        for k in cfg.experiment.k:
            train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(L, k)
            transport_operators[(L, k)] = TransportOperator()
            # training loop
            transport_operators[(L, k)].fit()
            
if __name__ == "__main__":
    main()
