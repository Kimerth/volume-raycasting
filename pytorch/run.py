import os
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

import jobs

import google
import shutil
from datetime import datetime
import traceback

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# TODO in C++: possibility to load different models
# TODO switch to pathlib

class DataCallback(Callback):
    def __init__(self) -> None:
        ...

    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        if hasattr(google, 'colab'):
            getattr(google, 'colab').drive.mount('/content/drive')

    def on_run_end(self, config: DictConfig, job_return: JobReturn, **kwargs: Any) -> None:
        if hasattr(google, 'colab'):
            shutil.make_archive(f'/content/drive/MyDrive/{config["hparams"]["drive_save_path"]}/{datetime.now()}', 'zip', os.environ['OUTPUT_PATH'])
            getattr(google, 'colab').drive.flush_and_unmount()


@hydra.main(config_path='conf', config_name='config')
def my_app(cfg: DictConfig) -> None:
    log = logging.getLogger(__name__)
    os.environ['OUTPUT_PATH'] = os.getcwd()
    log.debug(f'output path: {os.getcwd()}')
    os.chdir(get_original_cwd())
    log.debug(f'cwd: {get_original_cwd()}')

    dependencies = {}
    for job_name, job_config in cfg['jobs'].items():
        log.info(f'Starting stage === {job_name} ===')
        try:
            dependencies.update(getattr(jobs, job_config['fun'])(job_config, dependencies))
        except Exception as e:
            log.error(f'Error in job {job_name}: {e}')
            log.debug(traceback.format_exc())


if __name__ == '__main__':
    my_app()
