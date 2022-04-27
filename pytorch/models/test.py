import logging

from omegaconf import DictConfig
from torch.nn import BCEWithLogitsLoss
from util import metric, metrics_map

from .train import _test_model


def test(cfg: DictConfig, dependencies: dict) -> dict:
    log = logging.getLogger(__name__)

    criterion = BCEWithLogitsLoss()

    test_metrics = _test_model(dependencies["data_loader_test"], dependencies["model"], criterion)
    average_metrics = test_metrics.aggregate()
    for idx, name in enumerate(["loss"] + metrics_map):
        log.info(f"test/{name}: {average_metrics[idx]}")

    return {}
