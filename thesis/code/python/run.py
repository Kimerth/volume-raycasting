import logging
import os
from typing import Any

from dotenv import load_dotenv
from hydra.core.utils import JobReturn

load_dotenv()

import shutil
from datetime import datetime

import google
import hydra
import torch
from hydra.experimental.callback import Callback
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

import jobs

if torch.cuda.is_available():
    torch.cuda.empty_cache()


class DataCallback(Callback):
    def __init__(self) -> None:
        ...

    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        if hasattr(google, "colab"):
            getattr(google, "colab").drive.mount("/content/drive")

    def on_run_end(
        self, config: DictConfig, job_return: JobReturn, **kwargs: Any
    ) -> None:
        if hasattr(google, "colab") and "train_model" in config.jobs:
            shutil.make_archive(
                f"/content/drive/MyDrive/{config.jobs.train_model.drive_save_path}/{datetime.now()}",
                "zip",
                os.environ["OUTPUT_PATH"],
            )
            getattr(google, "colab").drive.flush_and_unmount()


@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    log = logging.getLogger(__name__)
    os.environ["OUTPUT_PATH"] = os.getcwd()
    log.debug(f"output path: {os.getcwd()}")
    os.chdir(get_original_cwd())
    log.debug(f"cwd: {get_original_cwd()}")

    dependencies = {}
    for job_name, job_config in cfg["jobs"].items():
        log.info(f"Starting stage === {job_name} ===")
        try:
            dependencies.update(
                getattr(jobs, job_config["fun"])(job_config, dependencies)
            )
        except Exception as e:
            log.exception(f"Error in job {job_name}: {e}")


if __name__ == "__main__":
    my_app()
