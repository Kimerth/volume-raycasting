from collections import OrderedDict

import torch
import torch.nn as nn
from omegaconf import DictConfig

from monai.networks.nets.vit import ViT


class UNETR(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(UNETR, self).__init__()

        self.patch_size = (16, 16, 16)
        self.feat_size = (
            cfg.img_size[0] // self.patch_size[0],
            cfg.img_size[1] // self.patch_size[1],
            cfg.img_size[2] // self.patch_size[2],
        )

        self.vit = ViT(
            in_channels=1,
            img_size=cfg.img_size,
            patch_size=self.patch_size,
            hidden_size=cfg.hidden_size,
            mlp_dim=cfg.mlp_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            pos_embed='perceptron',
            classification=False,
            dropout_rate=cfg.dropout_rate,
        )


    def forward(self, x: torch.Tensor):
        ...



    @staticmethod
    def _upsampling_block():
        ...

    @staticmethod
    def _projection_upsampling_block():
        ...
