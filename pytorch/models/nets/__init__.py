from monai.networks.nets.unet import UNet
from monai.networks.nets.unetr import UNETR
from omegaconf import DictConfig
import torch


def get_net(cfg: DictConfig) -> torch.nn.Module:
    if cfg["net_name"] == "UNet":
        return UNet(
            spatial_dims=3,
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            channels=cfg.channels,
            strides=cfg.strides,
            dropout=cfg.dropout,
        )
    elif cfg["net_name"] == "UNETR":
        return UNETR(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            img_size=cfg.img_size,
            dropout_rate=cfg.dropout,
        )

    raise NotImplementedError
