from glob import glob
from os.path import dirname
from typing import Dict, List

from omegaconf import DictConfig
from torchio import LabelMap, ScalarImage, Subject, SubjectsDataset, Queue
from torchio.data import UniformSampler
from torchio.transforms import (Compose, CropOrPad, OneOf, RandomAffine,
                                RandomBiasField, RandomElasticDeformation,
                                RandomFlip, RandomMotion, RandomNoise,
                                ToCanonical, ZNormalization)


def _create_data_map(cfg: DictConfig) -> Dict[str, List[str]]:
    data_paths: List[str] = glob(
        cfg["data"]["base_path"] + cfg["data"]["scan_pattern"] + cfg["data"]["file_name"], recursive=True)

    return {image_path: glob(f'${image_path}/${dirname(image_path)}/*') for image_path in data_paths}


def _load_base_data(data_map: Dict[str, List[str]]) -> List[Subject]:
    return [Subject(ScalarImage(path), LabelMap(*label_paths)) for path, label_paths in data_map]


def _get_transform(cfg: DictConfig):
    return Compose([
        # ToCanonical(),
        CropOrPad((cfg["data"]["crop_or_pad_size"]),
                  padding_mode='reflect'),
        # RandomMotion(),
        RandomBiasField(),
        ZNormalization(),
        RandomNoise(),
        RandomFlip(axes=(0,)),
        OneOf({
            RandomAffine(): 0.8,
            RandomElasticDeformation(): 0.2,
        }), ])


def load_dataset(cfg: DictConfig) -> Queue:
    data_map = _create_data_map(cfg)

    subjects = _load_base_data(data_map)
    transform = _get_transform(cfg)
    dataset = SubjectsDataset(subjects, transform)

    return Queue(dataset,
                 cfg["data"]["queue_length"],
                 cfg["data"]["samples_per_volume"],
                 UniformSampler(cfg["data"]["patch_size"]))
