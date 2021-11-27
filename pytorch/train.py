import logging
import os

import torch
from omegaconf import DictConfig
from torch.nn import CrossEntropyLoss
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

# from models.unet3d import UNet3D
from models.ext_resnet import ResidualUNet3D
# from models.ext_unetr import UNETR


def train(cfg: DictConfig, data_loader: torch.utils.data.DataLoader) -> torch.nn.Module:
    log = logging.getLogger(__name__)

    # TODO get device from config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Using device: {device}')

    torch.autograd.set_detect_anomaly(cfg['anomaly_detection'])

    # model = UNet3D(cfg).to(device)
    model = ResidualUNet3D(cfg['in_channels'], cfg['out_channels']).to(device)

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

    criterion = CrossEntropyLoss().to(device)

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
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = int(checkpoint['epoch']) + 1

    # first_iter = True

    # FIXME when loading checkpoint resume epochs
    cumulative_loss: float = 0
    for epoch in tqdm(
                range(start_epoch, cfg['total_epochs'] + 1),
                total=cfg['total_epochs'],
                position=0,
                leave=False,
                desc='Epoch'
    ):
        log.info(f'Starting epoch: {epoch}...')

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
            # logits = torch.sigmoid(logits)
            loss: torch.Tensor = criterion(logits, y.to(torch.float).to(device))

            loss.backward()
            # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            optimizer.step()
            scheduler.step()

            cumulative_loss += loss.item()

            if batch_idx % cfg['metrics_every'] == 0:
                cumulative_loss /= cfg['metrics_every']

                log.info(f'epoch {epoch}/{cfg["total_epochs"]} - batch {batch_idx}/{len(data_loader)}')

                writer.add_scalar('training/loss', cumulative_loss, batch_idx * epoch)
                log.info(f'\t\ttraining/loss: {cumulative_loss}')

                # FIXME monai.metrics.Cumulative
                labels = (logits > 0.5).float()
                metrics = metric(y.cpu(), labels.cpu())
                for k, v in metrics.items():
                    writer.add_scalar(f'training/{k}', v, batch_idx * epoch)
                    log.info(f'\t\ttraining/{k}: {v}')

                # plot_segmentation(x[0].cpu(), labels[0].cpu(), os.path.join(
                #         os.environ['OUTPUT_PATH'],
                #         cfg['validation_plots_dir'],
                #         f'epoch{epoch}-batch{(batch_idx + 1)}'
                #     )
                # )

                cumulative_loss = 0

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

    writer.flush()
    writer.close()
    return model
