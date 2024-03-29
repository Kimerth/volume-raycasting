import logging
import os

import torch
from data.visualization import train_visualizations
from monai.metrics.cumulative_average import CumulativeAverage
from omegaconf import DictConfig
from pytorch_model_summary import summary
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from util import import_tqdm, metric, metrics_map

from .nets import get_net

tqdm = import_tqdm()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train_epoch(
    data_loader: torch.utils.data.DataLoader,  # type: ignore
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: Optimizer,
) -> CumulativeAverage:
    model.train()
    epoch_metrics = CumulativeAverage()

    for batch in tqdm(
        data_loader,
        total=len(data_loader),
        position=1,
        leave=False,
        desc="Train Steps",
    ):
        x: torch.Tensor = batch["image"]["data"]
        y: torch.Tensor = batch["seg"]["data"]

        optimizer.zero_grad()

        logits = model(x.to(device).float())
        loss: torch.Tensor = criterion(logits, y.to(torch.float).to(device))

        loss.backward()
        # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        optimizer.step()

        y_pred = (torch.sigmoid(logits) > 0.5).float()
        epoch_metrics.append([loss.item(), *metric(y.cpu(), y_pred.cpu())])

    return epoch_metrics


def _get_metrics_for_model(
    data_loader: torch.utils.data.DataLoader,  # type: ignore
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    step: str,
):
    epoch_metrics = CumulativeAverage()

    for batch in tqdm(
        data_loader,
        total=len(data_loader),
        position=1,
        leave=False,
        desc=f"{step} Steps",
    ):
        x: torch.Tensor = batch["image"]["data"]
        y: torch.Tensor = batch["seg"]["data"]

        logits = model(x.to(device).float())
        loss: torch.Tensor = criterion(logits, y.to(torch.float).to(device))

        y_pred = (torch.sigmoid(logits) > 0.5).float()
        epoch_metrics.append([loss.item(), *metric(y.cpu(), y_pred.cpu())])

    return epoch_metrics


def _validate_model(
    data_loader: torch.utils.data.DataLoader,  # type: ignore
    model: torch.nn.Module,
    criterion: torch.nn.Module,
):
    model.eval()
    return _get_metrics_for_model(data_loader, model, criterion, "Validation")


def _test_model(
    data_loader: torch.utils.data.DataLoader,  # type: ignore
    model: torch.nn.Module,
    criterion: torch.nn.Module,
):
    model.eval()
    return _get_metrics_for_model(data_loader, model, criterion, "Test")


def train(cfg: DictConfig, dependencies: dict) -> dict:
    log = logging.getLogger(__name__)

    if "data_loader_train" not in dependencies or "data_loader_val" not in dependencies:
        raise Exception("Missing required dependencies: data loaders")

    train_loader, val_loader = (
        dependencies["data_loader_train"],
        dependencies["data_loader_val"],
    )

    log.info(f"Using device: {device}")

    torch.autograd.set_detect_anomaly(cfg["anomaly_detection"])  # type: ignore

    if "model" in dependencies:
        model = dependencies["model"]
        dependencies["model"].train(True)
    else:
        model = get_net(cfg).to(device)

    try:
        log.info(f"Getting summary of {model.__class__.__name__}...")
        log.info(
            summary(
                model,
                torch.zeros((1, 1, 64, 32, 64)).to(device),
                show_input=True,
                show_hierarchical=True,
                show_parent_layers=True,
            )
        )
    except Exception as e:
        log.warn(f"Failed to get summary of {model.__class__.__name__}: {e}")

    optimizer: Optimizer = torch.optim.Adam(model.parameters(), lr=cfg["init_lr"])
    scheduler = StepLR(
        optimizer, step_size=cfg["scheduer_step_size"], gamma=cfg["scheduer_gamma"]
    )

    criterion = BCEWithLogitsLoss().to(device)

    writer = SummaryWriter(
        os.path.join(os.environ["OUTPUT_PATH"], cfg["tb_output_dir"])
    )

    if device.type == "cuda":
        log.info(
            f"{model.__class__.__name__} Memory Usage on {torch.cuda.get_device_name()}:"
        )
        model_memory_allocated = round(torch.cuda.memory_allocated() / 1024**3, 1)
        log.info(f"\tAllocated: {model_memory_allocated} GB")

    best_val_loss = float("inf")
    early_stopping_patience_counter = 0
    early_stopping = False

    start_epoch = 1

    if "checkpoint" in dependencies:
        checkpoint = dependencies["checkpoint"]
        optimizer.load_state_dict(checkpoint["optim"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1

    tqdm_obj = tqdm(
        range(start_epoch, start_epoch + cfg["total_epochs"]),
        initial=start_epoch,
        total=cfg["total_epochs"],
        position=0,
        leave=True,
        desc="Epoch",
    )
    for epoch in tqdm_obj:
        epoch_metrics = _train_epoch(train_loader, model, criterion, optimizer)
        scheduler.step()

        average_metrics = epoch_metrics.aggregate()
        tqdm_obj.set_postfix(
            {
                name: average_metrics[idx].item()
                for idx, name in enumerate(["loss"] + metrics_map)
            }
        )

        if epoch % cfg["metrics_every"] == 0:
            log.info(f"Training metrics for epoch {epoch}")
        for idx, name in enumerate(["loss"] + metrics_map):
            writer.add_scalar(f"training/{name}", average_metrics[idx], epoch)
            if epoch % cfg["metrics_every"] == 0:
                log.info(f"\ttraining/{name}: {average_metrics[idx]}")

        if epoch % cfg["metrics_every"] == 0:
            train_visualizations(
                writer,
                epoch,
                model,
                val_loader,
                device,
                f'{os.environ["OUTPUT_PATH"]}/{cfg["plots_output_path"]}',
            )

        def save_model(name):
            checkpoints_path = os.path.join(
                os.environ["OUTPUT_PATH"], cfg["checkpoints_dir"]
            )
            if not os.path.exists(checkpoints_path):
                os.makedirs(checkpoints_path)
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                os.path.join(checkpoints_path, name),
            )

        if epoch % cfg["validate_every"] == 0:
            val_metrics = _validate_model(val_loader, model, criterion)
            average_metrics = val_metrics.aggregate()

            average_val_loss = average_metrics[0]
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                early_stopping_patience_counter = 0
                save_model(f"checkpoint_best.pt")
            elif (
                average_val_loss > best_val_loss + 1e-3
                and epoch > cfg["early_stop_ignore"]
            ):
                log.warn(
                    f"Validation loss is increasing. Early stopping in {cfg['early_stop_patience'] - early_stopping_patience_counter}."
                )
                early_stopping_patience_counter += 1

                if early_stopping_patience_counter > cfg["early_stop_patience"]:
                    early_stopping = True

            log.info(f"Validation metrics for epoch {epoch}")
            for idx, name in enumerate(["loss"] + metrics_map):
                writer.add_scalar(f"validation/{name}", average_metrics[idx], epoch)
                log.info(f"\tvalidation/{name}: {average_metrics[idx]}")

        # save_model(cfg['latest_checkpoint_file'])

        if epoch % cfg["save_every"] == 0 or early_stopping:
            save_model(f"checkpoint_{epoch:04d}.pt")

        if early_stopping:
            break

    writer.flush()
    writer.close()

    return {"model": model}
