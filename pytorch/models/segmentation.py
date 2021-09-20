from collections import OrderedDict

import torch
import torch.nn as nn
from omegaconf import DictConfig


class UNet3D(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(UNet3D, self).__init__()

        features: int = int(cfg['init_features'])

        self.encoders = [
            (
                UNet3D._block(cfg['in_channels'],
                              features, name='enc_1'),
                nn.MaxPool3d(kernel_size=2, stride=2)
            )
        ]
        for idx in range(1, cfg['nb_blocks']):
            features *= 2
            self.encoders += [
                (
                    UNet3D._block(features, features * 2,
                                  name=f'enc_{idx + 1}'),
                    nn.MaxPool3d(kernel_size=2, stride=2)
                )
            ]

        self.bottleneck = UNet3D._block(features * 2, features * 4, name='bottleneck')

        self.decoders = []
        for idx in range(cfg['nb_blocks'], 0, -1):
            features //= 2
            self.decoders += [
                (
                    nn.ConvTranspose3d(features, features // 2,
                                       kernel_size=2, stride=2),
                    UNet3D._block(features, features // 8, name=f'dec_{idx}')
                )
            ]

        self.output_conv = nn.Conv3d(
            in_channels=features, out_channels=cfg['out_channels'], kernel_size=1)

    def forward(self, x):
        outputs = OrderedDict([('x', x)])

        def get_last_output():
            k = next(reversed(outputs))
            return outputs[k]

        for idx, (encoder, pool) in enumerate(self.encoders):
            outputs[f'enc_{idx}'] = encoder(get_last_output())
            outputs[f'pool_{idx}'] = pool(outputs[f'enc_{idx}'])

        outputs['bottleneck'] = self.bottleneck(outputs[-1])

        for idx, (upconv, decoder) in reversed(list(enumerate(self.decoders))):
            outputs[f'upconv_{idx}'] = torch.cat(
                (
                    upconv(get_last_output()),
                    outputs[f'enc_{idx}']
                ), dim=1)
            outputs[f'dec_{idx}'] = decoder(outputs[f'upconv_{idx}'])

        return self.output_conv(get_last_output())

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
