import torch
from omegaconf import DictConfig
import os

from models.train import device


def deploy_model(cfg: DictConfig, dependencies: dict) -> dict:
    if 'model' not in dependencies:
        raise Exception('Cannot deploy! Missing model...')

    # FIXME hardcode
    example = torch.rand(1, 1, 32, 32, 32).to(device)

    dependencies['model'].train(False)
    script_module = torch.jit.trace(dependencies['model'], example)  # type: ignore

    script_module.save(f"{os.environ['OUTPUT_PATH']}/torchscript_module_model.pt")

    return {}
