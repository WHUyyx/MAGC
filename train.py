from argparse import ArgumentParser
import os

# python train.py  \
# --config configs/train_cldm.yaml

import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch

from utils.common import instantiate_from_config, load_state_dict


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/train_cldml')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    pl.seed_everything(config.lightning.seed, workers=True)
    
    data_module = instantiate_from_config(config.data)
    model = instantiate_from_config(OmegaConf.load(config.model.config)) # 加载模型结构

    # TODO: resume states saved in checkpoint.
    if config.model.get("resume"): # 加载模型权重
        load_state_dict(model, torch.load(config.model.resume, map_location="cpu"), strict=False)
    
    callbacks = []
    for callback_config in config.lightning.callbacks:
        callbacks.append(instantiate_from_config(callback_config))
    trainer = pl.Trainer(callbacks=callbacks, **config.lightning.trainer)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
