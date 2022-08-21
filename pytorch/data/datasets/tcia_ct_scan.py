from glob import glob
import os
from typing import Dict, List
from omegaconf.dictconfig import DictConfig
from torchio.transforms.transform import Transform
from .generic import Dataset


class TCIADataset(Dataset):
    def __init__(self, cfg: DictConfig, transform: Transform, **kwargs):
        super().__init__(cfg, transform, **kwargs)

    def _create_data_map(self) -> Dict[str, List[str]]:
        scan_pattern: str = self.cfg["base_path"] + self.cfg["scan_pattern"]
        Dataset.log.debug(f"scan_pattern={scan_pattern}")

        return {
            image_path: glob(
                os.path.join(os.path.dirname(image_path), "segmentations", "*")
            )
            for image_path in glob(scan_pattern, recursive=True)
        }

    def _get_subject_id(self, path: str) -> str:
        return os.path.basename(os.path.dirname(path))
