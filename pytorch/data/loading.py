import logging
import os
import random
from pprint import pformat
from typing import List
from torchio import SubjectsDataset

import torchio.datasets
from omegaconf import open_dict
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader
from torchio.data.queue import Queue
from torchio.data.sampler import GridSampler
from torchio.transforms import *

import numpy as np

import data.datasets as datasets
from .visualization import plot_subject
from datasets.generic import Dataset


def __create_data_loader(
    dataset: Dataset,
    queue_max_length: int,
    queue_samples_per_volume: int,
    sampler,
    batch_size,
    verbose,
) -> DataLoader:
    queue = Queue(
        subjects_dataset=dataset,
        max_length=queue_max_length,
        samples_per_volume=queue_samples_per_volume,
        sampler=sampler,
        verbose=verbose,
        num_workers=2,
    )

    return DataLoader(
        queue, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True
    )


def split_dataset(dataset: Dataset, lengths: List[int]) -> List[Dataset]:
    """
    Split a dataset into multiple datasets.

    Args:
        dataset: The dataset to split.
        lengths: The lengths of the splits.

    Returns:
        A list of datasets.
    """
    div_points = [0] + lengths + [len(dataset)]

    datasets = []
    for i in range(len(div_points)):
        st = div_points[i]
        end = div_points[i + 1]
        datasets.append(SubjectsDataset(dataset._subjects[st:end]))

    return datasets


def get_data_loader(cfg: DictConfig, _) -> dict:
    log = logging.getLogger(__name__)

    # maintain consitent preprocessing across datasets
    transform = Compose(
        [
            # ToCanonical(),
            ZNormalization(),
            RandomMotion(),
            RandomBiasField(),
            RandomNoise(),
            RandomFlip(axes=(0,)),
        ]
    )

    log.info(f"Data loader selected: {cfg['dataset']}")
    try:
        log.info("Attempting to use defined data loader")
        dataset = getattr(datasets, cfg["dataset"])(cfg, transform)
    except ImportError:
        log.info("Not a defined data loader... Attempting to use torchio loader")
        dataset = getattr(torchio.datasets, cfg["dataset"])(
            root=cfg["base_path"], transform=transform, download=True
        )

    for subject in random.sample(dataset._subjects, cfg["plot_number"]):
        plot_subject(
            subject,
            os.path.join(
                os.environ["OUTPUT_PATH"], cfg["save_plot_dir"], subject["subject_id"]
            ),
        )

    sampler = GridSampler(patch_size=cfg["patch_size"])
    samples_per_volume = len(sampler._compute_locations(dataset[0]))  # type: ignore

    with open_dict(cfg):
        cfg["size"] = dataset[0].spatial_shape
        # cfg['batch'] = samples_per_volume

    # FIXME test_set number of samples is hardcoded
    val_size = max(1, int(0.2 * len(dataset)))
    test_set, train_set, val_set = split_dataset(
        dataset, [21, len(dataset) - val_size - 21, val_size]
    )

    test_set.set_transform(ZNormalization())
    val_set.set_transform(ZNormalization())

    train_loader = __create_data_loader(
        train_set,
        queue_max_length=samples_per_volume * cfg["queue_length"],
        queue_samples_per_volume=samples_per_volume,
        sampler=sampler,
        verbose=log.level > 0,
        batch_size=cfg["batch"],
    )

    val_loader = __create_data_loader(
        val_set,
        queue_max_length=samples_per_volume * cfg["queue_length"],
        queue_samples_per_volume=samples_per_volume,
        sampler=sampler,
        verbose=log.level > 0,
        batch_size=cfg["batch"],
    )

    test_loader = __create_data_loader(
        test_set,
        queue_max_length=samples_per_volume * cfg["queue_length"],
        queue_samples_per_volume=samples_per_volume,
        sampler=sampler,
        verbose=log.level > 0,
        batch_size=cfg["batch"],
    )

    return {
        "data_loader_train": train_loader,
        "data_loader_val": val_loader,
        "data_loader_test": test_loader,
    }
