import logging
import os
import random
from pprint import pformat

import torchio.datasets
from omegaconf import dictconfig, open_dict
from torch.utils.data import DataLoader
from torchio.data.queue import Queue
from torchio.data.sampler import GridSampler
from torchio.transforms import *

import data.datasets as datasets
from .visualization import plot_subject


def get_data_loader(cfg: dictconfig) -> DataLoader:
    log = logging.getLogger(__name__)

    # maintain consitent preprocessing across datasets
    transform = Compose(
        [
            # ToCanonical(),
            # Resize(cfg['size']),
            RandomMotion(),
            RandomBiasField(),
            RandomNoise(),
            RandomFlip(axes=(0,)),
        ]
    )

    log.info(f"Data loader selected: {cfg['dataset']}")
    try:
        log.info("Attempting to use defined data loader")
        dataset = getattr(datasets, cfg['dataset'])(cfg, transform)
    except ImportError:
        log.info("Not a defined data loader... Attempting to use torchio loader")
        dataset = getattr(torchio.datasets, cfg['dataset'])(
            root=cfg['base_path'],
            transform=transform,
            download=True
        )

    for subject in random.sample(dataset._subjects, cfg['plot_number']):
        plot_subject(subject, os.path.join(
                os.environ['OUTPUT_PATH'],
                cfg['save_plot_dir'],
                subject["subject_id"]
            )
        )

    sampler = GridSampler(patch_size=cfg['patch_size'])
    samples_per_volume = len(sampler._compute_locations(dataset[0]))

    with open_dict(cfg):
        cfg['size'] = dataset[0].spatial_shape
        # cfg['batch'] = samples_per_volume

    queue = Queue(
        subjects_dataset=dataset,
        max_length=samples_per_volume * cfg['queue_length'],
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        verbose=log.level > 0,
        num_workers=2
    )

    log.info(queue)

    return DataLoader(
        queue,
        batch_size=cfg['batch'],
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )
