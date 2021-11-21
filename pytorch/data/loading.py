import logging
from importlib import import_module
from pprint import pformat

from omegaconf import dictconfig
from torch.utils.data import DataLoader
from torchio.data.queue import Queue
from torchio.data.sampler.uniform import UniformSampler
from torchio.transforms import *

import sys
import os
import random


# FIXME
sys.path.append(os.path.dirname(__file__))

from visualization import plot_segmentation

def get_data_loader(cfg: dictconfig) -> DataLoader:
    log = logging.getLogger(__name__)

    # maintain consitent preprocessing across datasets
    transform = Compose(
        [
            ToCanonical(),
            CropOrPad(
                cfg['crop_or_pad_size'],
                padding_mode='reflect'
            ),
            # RandomMotion(),
            # RandomBiasField(),
            RandomNoise(),
            # RandomFlip(axes=(0,)),
            OneOf(
                {
                    RandomAffine(): 0.8,
                    RandomElasticDeformation(): 0.2,
                }
            )
        ]
    )

    log.info(f"Data loader selected: {cfg['dataset']}")
    try:
        log.info("Attempting to use defined data loader")
        dataset = getattr(
            import_module(f"datasets.{cfg['dataset']}"),
            'Dataset'
        )(
            cfg['base_path'] + cfg['scan_pattern'],
            transform
        )
    except ImportError:
        log.info("Not a defined data loader... Attempting to use torchio loader")
        dataset = getattr(
            import_module(f"torchio.datasets"),
            cfg['dataset']
        )(
            root=cfg['base_path'],
            transform=transform,
            download=True
        )

    for subject in random.sample(dataset._subjects, cfg['plot_number']):
        plot_segmentation(subject['image'], subject['seg'], os.path.join(
                os.environ['OUTPUT_PATH'],
                cfg['save_plot_dir'],
                f'{subject["subject_id"]}.png'
            )
        )

    queue = Queue(
        subjects_dataset=dataset,
        max_length=cfg['queue_length'],
        samples_per_volume=cfg['samples_per_volume'],
        sampler=UniformSampler(cfg['patch_size']),
        verbose=log.level > 0
    )

    log.info(queue)

    return DataLoader(
        queue,
        batch_size=cfg['batch'],
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
