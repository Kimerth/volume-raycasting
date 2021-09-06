from typing import Dict, List, Tuple
import torch
from omegaconf import DictConfig, OmegaConf
import torchio as tio

def create_data_map(cfg : DictConfig) -> Dict[str, List[str]]:
    pass

def load_base_data(data_map: Dict[str, List[str]]) -> Tuple[List[tio.ScalarImage], List[tio.LabelMap]]:
    images: List[tio.ScalarImage] = []
    labels: List[tio.LabelMap] = []
    for path, label_paths in data_map:
        images.append(tio.ScalarImage(path))
        labels.append(tio.LabelMap(*label_paths))
    return images