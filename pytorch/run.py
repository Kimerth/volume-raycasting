import os
import re
import sys
import logging
from typing import Any

from dotenv import load_dotenv
from hydra.core.utils import JobReturn

load_dotenv()

import hydra
import torch
from hydra.experimental.callback import Callback
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd

from data import get_data_loader
from train import train

from google.colab import drive
import shutil
from datetime import datetime

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# TODO in C++: possibility to load different models
# TODO switch to pathlib

# FIXME
class DataCallback(Callback):
    def __init__(self) -> None:
        ...

    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        ...

    def on_run_end(elf, config: DictConfig, job_return: JobReturn, **kwargs: Any) -> None:
        drive.mount('/content/drive')
        shutil.make_archive(f'/content/drive/MyDrive/{config["hparams"]["drive_save_path"]}/{datetime.now()}', 'zip', os.environ['OUTPUT_PATH'])
        drive.flush_and_unmount()


@hydra.main(config_path='conf', config_name='config')
def my_app(cfg: DictConfig) -> None:
    log = logging.getLogger(__name__)
    os.environ['OUTPUT_PATH'] = os.getcwd()
    log.debug(f'output path: {os.getcwd()}')
    os.chdir(get_original_cwd())
    log.debug(f'cwd: {get_original_cwd()}')

    data_loader = get_data_loader(cfg['data'])
    train(cfg['hparams'], data_loader)


if __name__ == '__main__':
    my_app()
