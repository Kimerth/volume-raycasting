import logging
import os
from typing import Any

import git
import hydra
from hydra.experimental.callback import Callback
from omegaconf import DictConfig, OmegaConf

from data import load_dataset
from train import train

repo = git.Repo('.', search_parent_directories=True)

os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['TOP_PROJECT_PATH'] = repo.working_tree_dir


class DataCallback(Callback):
    def __init__(self) -> None:
        pass

    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        pass


@hydra.main(config_path='conf', config_name='config')
def my_app(cfg: DictConfig) -> None:
    dataloader = load_dataset(cfg['data'])
    train(cfg['hparams'], dataloader)


if __name__ == '__main__':
    my_app()
