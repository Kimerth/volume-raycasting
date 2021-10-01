import logging
import os
import random
from glob import glob
from pprint import pformat
from typing import Dict, List

from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchio import LabelMap, Queue, ScalarImage, Subject, SubjectsDataset
from torchio.data import UniformSampler
from torchio.transforms import (Compose, CropOrPad, OneOf, RandomAffine,
                                RandomBiasField, RandomElasticDeformation,
                                RandomFlip, RandomMotion, RandomNoise,
                                ToCanonical, ZNormalization)


def _create_data_map(cfg: DictConfig) -> Dict[str, List[str]]:
    data_paths: List[str] = glob(
        cfg['base_path'] + cfg['scan_pattern'],
        recursive=True
    )

    return {
        image_path: glob(
            os.path.join(
                os.path.dirname(image_path),
                'segmentations',
                '*'
            )
        ) for image_path in data_paths
    }


def _load_base_data(data_map: Dict[str, List[str]]) -> List[Subject]:
    return [
        Subject(
                name=os.path.basename(os.path.dirname(path)),
                source=ScalarImage(path, check_nans=True),
                labels=LabelMap(label_paths),
        ) for path, label_paths in data_map.items()
    ]


def _get_transform(cfg: DictConfig):
    return Compose(
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

def load_dataset(cfg: DictConfig) -> DataLoader:
    log = logging.getLogger(__name__)

    data_map = _create_data_map(cfg)
    log.debug(
        pformat({
            os.path.basename(os.path.dirname(volume_path)):[
                os.path.basename(label_path) for label_path in labels
            ]
            for (volume_path, labels) in data_map.items()
        })
    )

    subjects = _load_base_data(data_map)
    save_plot_path = os.path.join(
        os.environ['OUTPUT_PATH'],
        cfg['save_plot_dir']
    )
    if not os.path.exists(save_plot_path):
        os.makedirs(save_plot_path)
    for subject in random.sample(subjects, cfg['plot_number']):
        subject.plot(
            output_path=os.path.join(
                save_plot_path,
                subject['name']
            )
        )

    transform = _get_transform(cfg)

    dataset = SubjectsDataset(subjects, transform)

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
