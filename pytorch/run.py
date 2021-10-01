import logging
import os
from typing import Any
import sys

sys.path.append(f'{os.path.dirname(__file__)}/../externals/code/')

import git
import hydra
from hydra.experimental.callback import Callback
from omegaconf import DictConfig, OmegaConf

from data import load_dataset
from train import train

import torch

repo = git.Repo('.', search_parent_directories=True)

os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['TOP_PROJECT_PATH'] = repo.working_tree_dir

if torch.cuda.is_available():
    torch.cuda.empty_cache()

class DataCallback(Callback):
    def __init__(self) -> None:
        pass

    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        pass


@hydra.main(config_path='conf', config_name='config')
def my_app(cfg: DictConfig) -> None:
    os.environ['OUTPUT_PATH'] = os.getcwd()

    data_loader = load_dataset(cfg['data'])
    train(cfg['hparams'], data_loader)


if __name__ == '__main__':
    my_app()
