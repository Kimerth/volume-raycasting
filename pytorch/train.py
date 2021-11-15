import logging
import os
import time

import torch
from omegaconf import DictConfig
from torch.nn import BCEWithLogitsLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary

from util import metric

# FIXME not working
# try:
#     if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
#         from tqdm.notebook import tqdm
#     else:
#         from tqdm import tqdm
# except NameError:
#     from tqdm import tqdm
from tqdm.notebook import tqdm

from models.segmentation import UNet3D


def train(cfg: DictConfig, data_loader: torch.utils.data.DataLoader) -> torch.nn.Module:
    log = logging.getLogger(__name__)

    # TODO get device from config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    writer = SummaryWriter(
        os.path.join(
            os.environ['OUTPUT_PATH'],
            cfg['tb_output_dir']
        )
    )

    first_iter = True

    # FIXME when loading checkpoint resume epochs
    iteration: int = 0
    metrics: dict[str, float] = None
    cumulative_loss: float = 0
    for epoch in tqdm(
                range(cfg['total_epochs']),
                total=cfg['total_epochs'],
                position=0,
                leave=False,
                desc='Data Loader'
    ):
        log.info(f'Starting epoch: {epoch + 1}...')

        optimizer.zero_grad()

        # TODO find a way to cache dataset
        for batch_idx, batch in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            position=1,
            leave=True,
            desc='Train Steps'
        ):
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

                if torch.isnan(loss):
                    raise Exception("NaN loss")

                loss = loss / float(cfg['accum_iter'])
                loss.backward()

                cumulative_loss += loss.item()

            logits = torch.sigmoid(outputs)
            labels = (logits > 0.5).float()

            if metrics is None:
                metrics = metric(y.cpu(), labels.cpu())
            else:
                for k, v in metric(y.cpu(), labels.cpu()).items():
                    metrics[k] += v / cfg['accum_iter']

            if (batch_idx + 1) % cfg['accum_iter'] == 0 or batch_idx + 1 == len(data_loader):
                log.info(f'epoch {epoch + 1}/{cfg["total_epochs"]} - batch {batch_idx + 1}/{len(data_loader)}')

                log.info('\tPerforming optimization step...')
                start = time.time()

                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
                optimizer.step()

                end = time.time()
                log.info(f'\tOptimization step done in {(end - start):.2f}s')

                iteration += 1
                log.info(f'\tMetrics for optimization iteration {iteration}')
                writer.add_scalar('training/loss', cumulative_loss, iteration)
                log.info(f'\t\ttraining/loss: {cumulative_loss}')
                if (batch_idx + 1) % cfg['accum_iter'] == 0:
                    for k, v in metrics.items():
                        writer.add_scalar(f'training/{k}', v, iteration)
                        log.info(f'\t\ttraining/{k}: {v}')

                optimizer.zero_grad()
                cumulative_loss = 0
                metrics = None

        scheduler.step()

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

        save_model(cfg['latest_checkpoint_file'])

        if epoch % cfg['save_every'] == 0:
            save_model(f'checkpoint_{epoch:04d}.pt')

    writer.close()
    return model
