import os
import re
import zipfile
from glob import glob
from typing import Dict

from omegaconf.dictconfig import DictConfig
from torchio.transforms.transform import Transform

from .generic import Dataset


class CTORGDataset(Dataset):
    def __init__(self, cfg: DictConfig, transform: Transform, **kwargs):
        if not os.path.isdir(cfg["base_path"]):
            os.makedirs(cfg["base_path"])

            with zipfile.ZipFile(f'/content/drive/{cfg["drive_path"]}', "r") as zip_ref:
                zip_ref.extractall(f'{cfg["base_path"]}/../')

        super().__init__(cfg, transform, **kwargs)

    def _create_data_map(self) -> Dict[str, str]:
        scan_pattern: str = self.cfg["base_path"] + self.cfg["scan_pattern"]
        Dataset.log.debug(f"scan_pattern={scan_pattern}")

        return {
            image_path: os.path.join(
                os.path.dirname(image_path),
                os.path.basename(image_path).replace("volume", "labels"),
            )
            for image_path in sorted(glob(scan_pattern, recursive=True), key=lambda x: int(re.search(r"\d+", x).group()))  # type: ignore
        }

    def _get_subject_id(self, path: str) -> str:
        return os.path.basename(path).split(".")[0]
