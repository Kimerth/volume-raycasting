import logging
from glob import glob
from os.path import dirname
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
        # FIXME paths
        image_path: glob(f'{dirname(image_path)}/segmentations/*') for image_path in data_paths
    }


def _load_base_data(data_map: Dict[str, List[str]]) -> List[Subject]:
    return [
        Subject(
                source=ScalarImage(path),
                labels=LabelMap(label_paths)
        ) for path, label_paths in data_map.items()
    ]


def _get_transform(cfg: DictConfig):
    return Compose(
        [
            # ToCanonical(),
            CropOrPad(
                cfg['crop_or_pad_size'],
                padding_mode='reflect'
            ),
            # RandomMotion(),
            RandomBiasField(),
            ZNormalization(),
            RandomNoise(),
            RandomFlip(axes=(0,)),
            OneOf(
                {
                    RandomAffine(): 0.8,
                    RandomElasticDeformation(): 0.2,
                }
            )
        ]
    )

# TODO: more prints
# TODO: get subject ids
def load_dataset(cfg: DictConfig) -> DataLoader:
    log = logging.getLogger(__name__)

    data_map = _create_data_map(cfg)
    # TODO: better print
    log.debug(data_map)

    subjects = _load_base_data(data_map)
    transform = _get_transform(cfg)
    dataset = SubjectsDataset(subjects, transform)

    queue = Queue(
        dataset,
        cfg['queue_length'],
        cfg['samples_per_volume'],
        UniformSampler(cfg['patch_size'])
    )
    log.info(queue)
    # TODO find out if you can add tqdm to data loader
    return DataLoader(
        queue,
        batch_size=cfg['batch'],
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
