from collections import OrderedDict

import torch
import torch.nn as nn
from omegaconf import DictConfig


class UNet3D(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(UNet3D, self).__init__()

        features: int = int(cfg['init_features'])
        self.nb_blocks = cfg['nb_blocks']

        self.encoders = nn.ModuleList([
            UNet3D._block(cfg['in_channels'], features, name='enc_1'),
            nn.MaxPool3d(kernel_size=2, stride=2)
        ])
        for idx in range(1, self.nb_blocks):
            self.encoders += [
                UNet3D._block(features, features * 2, name=f'enc_{idx + 1}'),
                nn.MaxPool3d(kernel_size=2, stride=2)
            ]
            features *= 2

        self.bottleneck = UNet3D._block(features, features * 2, name='bottleneck')
        features *= 2

        self.decoders = nn.ModuleList()
        for idx in range(self.nb_blocks, 0, -1):
            self.decoders += [
                nn.ConvTranspose3d(features, features // 2, kernel_size=2, stride=2),
                UNet3D._block(features, features // 2, name=f'dec_{idx}')
            ]
            features //= 2

        self.output_conv = nn.Conv3d(
            in_channels=features, out_channels=cfg['out_channels'], kernel_size=1)

    def forward(self, x: torch.Tensor):
        last_output: torch.Tensor = x
        enc_outputs: list[torch.Tensor] = []

        for (encoder, pool) in zip(self.encoders[0::2], self.encoders[1::2]):
            enc_outputs.append(encoder(last_output))
            last_output = pool(enc_outputs[-1])

        last_output = self.bottleneck(last_output)

        for idx, (upconv, decoder) in enumerate(
            zip(self.decoders[0::2], self.decoders[1::2])
        ):
            idx = (self.nb_blocks - 1) - idx
            last_output = torch.cat(
                (
                    upconv(last_output),
                    enc_outputs[idx]
                ), dim=1)
            last_output = decoder(last_output)

        return self.output_conv(last_output)

    @staticmethod
    def _block(in_channels: int, features: int, name: str):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        f'{name}_conv1',
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (f'{name}_norm1', nn.BatchNorm3d(num_features=features)),
                    (f'{name}_relu1', nn.ReLU(inplace=True)),
                    (
                        f'{name}_conv2',
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (f'{name}_norm2', nn.BatchNorm3d(num_features=features)),
                    (f'{name}_relu2', nn.ReLU(inplace=True)),
                ]
            )
        )
