import torch
from omegaconf import DictConfig
from models.unet3d import UNet3D
from models.train import device

def load_checkpoint(cfg: DictConfig, _: dict) -> dict:
    checkpoint = torch.load(cfg['load_checkpoint_path'])

    model = UNet3D(cfg).to(device)
    model.load_state_dict(checkpoint['model'])

    return {
        'model': model,
        'checkpoint': checkpoint
    }
