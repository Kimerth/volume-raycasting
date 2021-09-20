import logging
import os

import torch
from omegaconf import DictConfig
from torch.nn import BCEWithLogitsLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from models.segmentation import UNet3D


def train(cfg: DictConfig, dataloader: torch.utils.data.DataLoader) -> torch.nn.Module:
    log = logging.getLogger(__name__)

    model = UNet3D(cfg)
    optimizer: Optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg['init_lr']
    )
    scheduler = StepLR(
        optimizer,
        step_size=cfg['scheduer_step_size'],
        gamma=cfg['scheduer_gamma']
    )

    model.cuda()

    criterion = BCEWithLogitsLoss().cuda()

    writer = SummaryWriter(cfg['tb_output_dir'])

    model.train()

    for epoch in range(cfg['total_epochs']):
        log.info(f'Starting epoch: {epoch + 1}...')

        for i, batch in enumerate(dataloader):
            log.info(f'epoch: {epoch}: batch: {i + 1}/{len(dataloader)}')

            optimizer.zero_grad()

            x = batch['source']['data'].type(torch.FloatTensor).cuda()
            y = batch['labels']['data'].type(torch.FloatTensor).cuda()

            outputs = model(x)

            logits = torch.sigmoid(outputs)
            labels = (logits > 0.5).float()

            loss: torch.Tensor = criterion(outputs, y)

            loss.backward()
            optimizer.step()

            # TODO: metrics

        scheduler.step()

        def get_state_dict():
            return {
                    'epoch': epoch,

                    'model':     model.state_dict(),
                    'optim':     optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
            }

        torch.save(
            get_state_dict(),
            os.path.join(cfg['checkpoints_dir'], cfg['latest_checkpoint_file'])
        )

        if epoch % cfg['save_every'] == 0:
            torch.save(
                get_state_dict(),
                os.path.join(cfg['checkpoints_dir'], f'checkpoint_{epoch:04d}.pt')
            )

    writer.close()
    return model
