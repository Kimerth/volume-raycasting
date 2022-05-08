import torch
from omegaconf import DictConfig
from .nets import get_net
from .train import device

def load_checkpoint(cfg: DictConfig, _: dict) -> dict:
    checkpoint = torch.load(cfg['load_checkpoint_path'])

    model = get_net(cfg).to(device)
    model.load_state_dict(checkpoint['model'])

    return {
        'model': model,
        'checkpoint': checkpoint
    }
