import logging
import os
import time

import torch
from git import exc
from omegaconf import DictConfig
from torch._C import DeviceObjType
from torch.nn import BCEWithLogitsLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from util import metric

try:
    if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm

from models.segmentation import UNet3D


def train(cfg: DictConfig, data_loader: torch.utils.data.DataLoader) -> torch.nn.Module:
    log = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    log.info(f'Using device: {device}')

    model = UNet3D(cfg).half().to(device)
    # TODO figure out a way for better logging
    if device.type == 'cuda':
        log.info(f'{model.__class__.__name__} Memory Usage on {torch.cuda.get_device_name()}:')
        model_memory_allocated = round(torch.cuda.memory_allocated()/1024**3,1)
        log.info(f'\tAllocated: {model_memory_allocated} GB')


    optimizer: Optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg['init_lr']
    )
    scheduler = StepLR(
        optimizer,
        step_size=cfg['scheduer_step_size'],
        gamma=cfg['scheduer_gamma']
    )

    criterion = BCEWithLogitsLoss().half().to(device)

    writer = SummaryWriter(cfg['tb_output_dir'])

    first_iter = True

    # FIXME when loading checkpoint resume epochs
    iteration = 0
    metrics = None
    for epoch in range(cfg['total_epochs']):
        log.info(f'Starting epoch: {epoch + 1}...')

        optimizer.zero_grad()

        for batch_idx, batch in tqdm(enumerate(data_loader), position=0, leave=True, total=len(data_loader)):
            # TODO check out torchtyping
            x: torch.Tensor = batch['source']['data']
            y: torch.Tensor = batch['labels']['data']

            # FIXME add code in repo as external (not submodule) and change it to suit needs
            # if first_iter:
            #     log.info(f'Getting summary of {model.__class__.__name__}...')
            #     log.info(
            #         summary(model, input_size=x.shape[1:], device=str(device))
            #     )

            # FIXME this is plain ugly (maybe check tqdm handlers/events)
            if first_iter and device.type == 'cuda':
                memory_allocated = round(torch.cuda.memory_allocated()/1024**3,1)

                log.info(f'Data Memory Usage on {torch.cuda.get_device_name()}:')
                log.info(f'\tAllocated: {memory_allocated - model_memory_allocated} GB')

                log.info(f'Total Memory Usage on {torch.cuda.get_device_name()}:')
                log.info(f'\tAllocated: {memory_allocated} GB')

                first_iter = False

            with torch.set_grad_enabled(True):
                outputs = model(x.half().to(device))
                loss: torch.Tensor = criterion(outputs, y.half().to(device))

                log.info(f'epoch {epoch + 1}/{cfg["total_epochs"]} - batch {batch_idx + 1}/{len(data_loader)}: loss {loss.item()}')

                loss = loss / cfg['accum_iter']
                loss.backward()

            logits = torch.sigmoid(outputs)
            labels = (logits > 0.5).float()

            if metrics is None:
                metrics = metric(y.cpu(), labels.cpu())
            else:
                for k, v in metric(y.cpu(), labels.cpu()).items():
                    metrics[k] += v / cfg['accum_iter']

            if (batch_idx + 1) % cfg['accum_iter'] == 0 or batch_idx + 1 == len(data_loader):
                log.info('Performing optimization step...')
                start = time.time()

                optimizer.step()
                optimizer.zero_grad()

                end = time.time()
                log.info(f'Optimization step done in {end - start}')

                iteration += 1
                log.info(f'Metrics for optimization iteration {iteration}')
                writer.add_scalar('training/loss', loss.item(), iteration)
                log.info(f'\ttraining/loss: {loss.item()}')
                if (batch_idx + 1) % cfg['accum_iter'] == 0:
                    for k, v in metrics.items():
                        writer.add_scalar(f'training/{k}', v, iteration)
                        log.info(f'\ttraining/{k}: {v}')

        scheduler.step()

        def save_model(name):
            if not os.path.exists(cfg['checkpoints_dir']):
                os.makedirs(cfg['checkpoints_dir'])
            torch.save(
                {
                    'epoch': epoch,

                    'model':     model.state_dict(),
                    'optim':     optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                },
                os.path.join(cfg['checkpoints_dir'], name)
            )

        save_model(cfg['latest_checkpoint_file'])

        if epoch % cfg['save_every'] == 0:
            save_model(f'checkpoint_{epoch:04d}.pt')

    writer.close()
    return model
