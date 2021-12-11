import logging
import os

import torch
from omegaconf import DictConfig
from torch.nn import BCEWithLogitsLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
from monai.metrics.cumulative_average import CumulativeAverage
# from torchsummary import summary
from data.visualization import train_visualizations


from util import metric, metrics_dict

# FIXME not working
# try:
#     if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
#         from tqdm.notebook import tqdm
#     else:
#         from tqdm import tqdm
# except NameError:
#     from tqdm import tqdm
from tqdm.notebook import tqdm

from models.unet3d import UNet3D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        leave=True,
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


def train(cfg: DictConfig, data_loader: torch.utils.data.DataLoader) -> torch.nn.Module:
    log = logging.getLogger(__name__)

    # TODO get device from config
    log.info(f'Using device: {device}')

    torch.autograd.set_detect_anomaly(cfg['anomaly_detection'])

    # model = UNet3D(cfg).to(device)
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

    if cfg['load_checkpoint_path'] is not None:
        log.info(f'loading checkpoint: {cfg["load_checkpoint_path"]}...')

        checkpoint = torch.load(cfg['load_checkpoint_path'])
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        # TODO option to ignore resuming of scheduler
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = int(checkpoint['epoch']) + 1

    # first_iter = True
    for epoch in tqdm(
        range(start_epoch, cfg['total_epochs'] + 1),
        total=cfg['total_epochs'],
        position=0,
        leave=False,
        desc='Epoch'
    ):
        epoch_metrics = _train_epoch(data_loader, model, criterion, optimizer)
        scheduler.step()

        if epoch % cfg['metrics_every'] == 0:
            log.info(f'metrics for epoch {epoch}/{cfg["total_epochs"]}')
            train_visualizations(writer, epoch, model, data_loader, device, cfg['plots_output_path'])

        average_metrics = epoch_metrics.aggregate()
        for idx, name in enumerate(['loss'] + metrics_dict):
            writer.add_scalar(f'training/{name}', average_metrics[idx], epoch)
            if epoch % cfg['metrics_every'] == 0:
                log.info(f'\ttraining/{name}: {average_metrics[idx]}')

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
