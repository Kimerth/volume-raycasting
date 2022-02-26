import torch
from omegaconf import DictConfig
import os


def deploy_model(cfg: DictConfig, dependencies: dict) -> dict:
    if 'model' not in dependencies:
        raise Exception('Cannot deploy! Missing model...')

    # FIXME hardcode
    example = torch.rand(1, 1, 32, 32, 32)

    script_module = torch.jit.trace(dependencies['model'], example)

    script_module.save(f"{os.environ['OUTPUT_PATH']}/torchscript_module_model.pt")

    return {}
