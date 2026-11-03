"""Train improved activation transport operators.

    python train_improved_operators.py experiment.L=[5,9] experiment.k=[1,5]
    python train_improved_operators.py improved.correction.enabled=true

Writes {output_dir}/improved_L{L}_k{k}.pkl, loadable with
NormAwareTransportOperator.load or `python eval.py eval_mode=improved`.
"""
import logging
import os
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.activation_loader import ActivationLoader, get_train_val_test_datasets
from src.improved_transport_operator import NormAwareTransportOperator, fit_sparse_correction


@hydra.main(version_base=None, config_path="configs", config_name="improved")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.info(OmegaConf.to_yaml(cfg))
    loader = ActivationLoader(cfg.activation_dir)
    ic = cfg.improved
    os.makedirs(cfg.output_dir, exist_ok=True)
    for L in cfg.experiment.L:
        for k in cfg.experiment.k:
            train_ds, val_ds, _ = get_train_val_test_datasets(L, k, loader)
            op = NormAwareTransportOperator(
                L, k, norm_aware=ic.norm_aware, identity_prior=ic.identity_prior,
                special_case_bos=ic.special_case_bos, rel_alphas=list(ic.rel_alphas),
            ).fit(train_ds, val_ds)
            if ic.correction.enabled:
                fit_sparse_correction(op, train_ds, val_ds, h=ic.correction.width,
                                      k=ic.correction.k, epochs=ic.correction.epochs,
                                      lr=ic.correction.lr, seed=cfg.seed)
            path = os.path.join(cfg.output_dir, f"improved_L{L}_k{k}.pkl")
            op.save(path)
            logging.info("saved %s (alpha_rel=%s, val curve=%s)", path, op.alpha_rel_, op.val_curve_)


if __name__ == "__main__":
    main()
