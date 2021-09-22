import logging
import os

import torch
from omegaconf import DictConfig
from torch._C import DeviceObjType
from torch.nn import BCEWithLogitsLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from models.segmentation import UNet3D
from torchsummary import summary


def train(cfg: DictConfig, dataloader: torch.utils.data.DataLoader) -> torch.nn.Module:
    log = logging.getLogger(__name__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    log.info(f'Using device: {device}')

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

    model.to(device)
    # TODO figure out a way for better logging
    if device.type == 'cuda':
        log.info(f'{model.__class__.__name__} Memory Usage on {torch.cuda.get_device_name()}:')
        model_memory_allocated = round(torch.cuda.memory_allocated()/1024**3,1)
        log.info(f'\tAllocated: {model_memory_allocated} GB')

    criterion = BCEWithLogitsLoss().to(device)

    writer = SummaryWriter(cfg['tb_output_dir'])

    first_iter = True

    # FIXME when loading checkpoint resume epochs
    # TODO maybe use progress bar
    for epoch in range(cfg['total_epochs']):
        log.info(f'Starting epoch: {epoch + 1}...')

        for i, batch in enumerate(dataloader):
            log.info(f'epoch {epoch + 1}/{cfg["total_epochs"]} - batch {i + 1}/{len(dataloader)}')

            optimizer.zero_grad()

            # TODO check out torchtyping
            x: torch.Tensor = batch['source']['data']
            y: torch.Tensor = batch['labels']['data']

            if first_iter:
                log.info(f'Getting summary of {model.__class__.__name__}...')
                log.info(
                    summary(model, input_size=x.shape[1:], device=str(device))
                )

            x.to(device)
            y.to(device)

            # FIXME this is plain ugly
            if first_iter and device.type == 'cuda':
                memory_allocated = round(torch.cuda.memory_allocated()/1024**3,1)

                log.info(f'Data Memory Usage on {torch.cuda.get_device_name()}:')
                log.info(f'\tAllocated: {memory_allocated - model_memory_allocated} GB')

                log.info(f'Total Memory Usage on {torch.cuda.get_device_name()}:')
                log.info(f'\tAllocated: {memory_allocated} GB')

                first_iter = False

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
