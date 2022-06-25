import os

import torch
from omegaconf import DictConfig

from models.train import device


def deploy_model(cfg: DictConfig, dependencies: dict) -> dict:
    if "model" not in dependencies:
        raise Exception("Cannot deploy! Missing model...")

    example = torch.rand(1, 1, 192, 96, 192).to(device)

    dependencies["model"].train(False)
    script_module = torch.jit.trace(dependencies["model"], example)  # type: ignore

    script_module.save(f"{os.environ['OUTPUT_PATH']}/torchscript_module_model.pt")

    return {}
