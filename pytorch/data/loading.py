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

# FIXME
sys.path.append(os.path.dirname(__file__))


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
            RandomMotion(),
            RandomBiasField(),
            RandomNoise(),
            RandomFlip(axes=(0,)),
            OneOf(
                {
                    RandomAffine(): 0.8,
                    RandomElasticDeformation(): 0.2,
                }
            ),
            ZNormalization()
        ]
    )

    log.info(f"Data loader selected: {cfg['dataset']}")
    try:
        log.info("Attempting to use defined data loader")
        dataset = getattr(import_module(f"datasets.{cfg['dataset']}"), 'Dataset')(cfg['base_path'] + cfg['scan_pattern'], transform)
    except ImportError:
        log.info("Not a defined data loader... Attempting to use torchio loader")
        dataset = import_module(f"torchio.datasets.{cfg['dataset']}")(
            root=cfg['base_path'],
            transform=transform,
            download=True
        )

    # save_plot_path = os.path.join(
    #     os.environ['OUTPUT_PATH'],
    #     cfg['save_plot_dir']
    # )
    # if not os.path.exists(save_plot_path):
    #     os.makedirs(save_plot_path)
    # for subject in random.sample(subjects, cfg['plot_number']):
    #     plot_subject(subject, save_plot_path)

    queue = Queue(
        subjects_dataset=dataset,
        max_length=cfg['queue_length'],
        samples_per_volume=cfg['samples_per_volume'],
        sampler=UniformSampler(cfg['patch_size']),
        num_workers=4,
        start_background=True,
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