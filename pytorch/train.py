import logging
import os

import torch
from omegaconf import DictConfig
from torch.nn import BCEWithLogitsLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from monai.metrics.cumulative_average import CumulativeAverage
# from torchsummary import summary
from torchio.data.subject import Subject
from data.visualization import plot_subject
from torchio import GridAggregator
from torchio import ScalarImage, LabelMap


from util import metric, metrics_dict, random_subject_from_loader, batches_from_sampler

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
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = int(checkpoint['epoch']) + 1

    # first_iter = True

    # FIXME when loading checkpoint resume epochs
    train_metrics = CumulativeAverage()

    for epoch in tqdm(
                range(start_epoch, cfg['total_epochs'] + 1),
                total=cfg['total_epochs'],
                position=0,
                leave=False,
                desc='Epoch'
    ):
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
            scheduler.step()

            y_pred = (torch.sigmoid(logits) > 0.5).float()
            train_metrics.append(
                [
                    loss.item(),
                    *metric(y.cpu(), y_pred.cpu())
                ]
            )

        # if batch_idx % cfg['metrics_every'] == 0:
        log.info(f'metrics for epoch {epoch}/{cfg["total_epochs"]}')

        average_metrics = train_metrics.aggregate()
        for idx, name in enumerate(['loss'] + metrics_dict):
            writer.add_scalar(f'training/{name}', average_metrics[idx], batch_idx * epoch)
            log.info(f'\ttraining/{name}: {average_metrics[idx]}')
        train_metrics.reset()

        sampler = random_subject_from_loader(data_loader)
        aggregator_x = GridAggregator(sampler)
        aggregator_y = GridAggregator(sampler)
        aggregator_y_pred = GridAggregator(sampler)
        for batch, locations in batches_from_sampler(sampler, data_loader.batch_size):
            x: torch.Tensor = batch['image']['data']
            aggregator_x.add_batch(x, locations)
            y: torch.Tensor = batch['seg']['data']
            aggregator_y.add_batch(y, locations)

            logits = model(x.to(device))
            y_pred = (torch.sigmoid(logits) > 0.5).float()
            aggregator_y_pred.add_batch(y_pred, locations)

        plot_subject(Subject(
                image=ScalarImage(tensor=aggregator_x.get_output_tensor()),
                true_seg=LabelMap(tensor=aggregator_y.get_output_tensor()),
                pred_seg=LabelMap(tensor=aggregator_y_pred.get_output_tensor())
            ), os.path.join(
                os.environ['OUTPUT_PATH'],
                cfg['validation_plots_dir'],
                f'epoch{epoch}-batch{(batch_idx + 1)}'
            )
        )

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
