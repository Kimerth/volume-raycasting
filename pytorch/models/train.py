import logging
import os
from typing import Sequence

import torch
from omegaconf import DictConfig
from torch.nn import BCEWithLogitsLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
from monai.metrics.cumulative_average import CumulativeAverage
# from torchsummary import summary
from data.visualization import train_visualizations


from util import metric, metrics_map

# FIXME
# try:
#     if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
#         from tqdm.notebook import tqdm
#     else:
#         from tqdm import tqdm
# except NameError:
#     from tqdm import tqdm
from tqdm.notebook import tqdm

from models.unet3d import UNet3D

device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _train_epoch(
        data_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: Optimizer
    ) -> CumulativeAverage:

    epoch_metrics = CumulativeAverage()

    # TODO find a way to cache dataset
    for batch_idx, batch in tqdm(
        enumerate(data_loader),
        total=len(data_loader),
        position=1,
        leave=False,
        desc='Train Steps'
    ):
        batch_idx += 1

        # TODO check out torchtyping
        x: torch.Tensor = batch['image']['data']
        y: torch.Tensor = batch['seg']['data']

        # FIXME add code in repo as external (not submodule) and change it to suit needs
        # if first_iter:
        #     log.info(f'Getting summary of {model.__class__.__name__}...')
        #     log.info(
        #         summary(model, input_size=x.shape[1:], device=str(device))
        #     )

        # FIXME this is plain ugly (maybe check tqdm handlers/events)
        # if first_iter and device.type == 'cuda':
        #     memory_allocated = round(torch.cuda.memory_allocated()/1024**3,1)

        #     log.info(f'Data Memory Usage on {torch.cuda.get_device_name()}:')
        #     log.info(f'\tAllocated: {memory_allocated - model_memory_allocated} GB')

        #     log.info(f'Total Memory Usage on {torch.cuda.get_device_name()}:')
        #     log.info(f'\tAllocated: {memory_allocated} GB')

        #     first_iter = False

        optimizer.zero_grad()

        logits = model(x.to(device))
        loss: torch.Tensor = criterion(logits, y.to(torch.float).to(device))

        loss.backward()
        # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        optimizer.step()

        y_pred = (torch.sigmoid(logits) > 0.5).float()
        epoch_metrics.append(
            [
                loss.item(),
                *metric(y.cpu(), y_pred.cpu())
            ]
        )
    return epoch_metrics


def train(cfg: DictConfig, dependencies: dict) -> torch.nn.Module:
    log = logging.getLogger(__name__)

    if 'data_loaders' not in dependencies:
        raise Exception("Missing required dependency: data_loaders")

    train_loader, val_loader = dependencies['data_loaders']

    # TODO get device from config
    log.info(f'Using device: {device}')

    torch.autograd.set_detect_anomaly(cfg['anomaly_detection'])

    if 'model' in dependencies:
        model = dependencies['model']
    else:
        model = UNet3D(cfg).to(device)

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

    criterion = BCEWithLogitsLoss().to(device)

    writer = SummaryWriter(
        os.path.join(
            os.environ['OUTPUT_PATH'],
            cfg['tb_output_dir']
        )
    )

    start_epoch = 1

    if 'checkpoint' in dependencies:
        checkpoint = dependencies['checkpoint']
        optimizer.load_state_dict(checkpoint['optim'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = int(checkpoint['epoch']) + 1

    tqdm_obj = tqdm(
        range(start_epoch, start_epoch + cfg['total_epochs']),
        initial=start_epoch,
        total=cfg['total_epochs'],
        position=0,
        leave=True,
        desc='Epoch'
    )
    for epoch in tqdm_obj:
        epoch_metrics = _train_epoch(train_loader, model, criterion, optimizer)
        scheduler.step()

        average_metrics = epoch_metrics.aggregate()
        tqdm_obj.set_postfix({name:average_metrics[idx] for idx, name in enumerate(['loss'] + metrics_map)})
        for idx, name in enumerate(['loss'] + metrics_map):
            writer.add_scalar(f'training/{name}', average_metrics[idx], epoch)
            if epoch % cfg['metrics_every'] == 0:
                log.info(f'\ttraining/{name}: {average_metrics[idx]}')

        if epoch % cfg['metrics_every'] == 0:
            train_visualizations(writer, epoch, model, val_loader, device, f'{os.environ["OUTPUT_PATH"]}/{cfg["plots_output_path"]}')

        def save_model(name):
            checkpoints_path = os.path.join(
                os.environ['OUTPUT_PATH'],
                cfg['checkpoints_dir']
            )
            if not os.path.exists(checkpoints_path):
                os.makedirs(checkpoints_path)
            torch.save(
                {
                    'epoch': epoch,

                    'model':     model.state_dict(),
                    'optim':     optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                },
                os.path.join(
                    checkpoints_path,
                    name
                )
            )

        # save_model(cfg['latest_checkpoint_file'])

        if epoch % cfg['save_every'] == 0:
            save_model(f'checkpoint_{epoch:04d}.pt')

    writer.flush()
    writer.close()
    return model