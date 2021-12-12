from glob import glob
import os
from typing import Dict, List
from omegaconf import dictconfig
from torchio.transforms import transform
from .generic import Dataset
from google.colab import drive
import zipfile


class CTORGDataset(Dataset):
    def __init__(self, cfg: dictconfig, transform: transform, **kwargs):
        if not os.path.isdir(cfg['base_path']):
            os.makedirs(cfg['base_path'])

            drive.mount('/content/drive')
            with zipfile.ZipFile(f'/content/drive/{cfg["drive_path"]}', 'r') as zip_ref:
                for member in zip_ref.namelist():
                    zip_ref.extract(member, cfg['base_path'])
            drive.flush_and_unmount()

        super().__init__(cfg, transform, **kwargs)

    def _create_data_map(self) -> Dict[str, str]:
        scan_pattern: str = self.cfg['base_path'] + self.cfg['scan_pattern']
        Dataset.log.debug(f'scan_pattern={scan_pattern}')

        return {
            image_path: os.path.join(os.path.dirname(image_path), os.path.basename(image_path).replace('volume', 'labels'))
            for image_path in glob(scan_pattern, recursive=True)
        }

    def _get_subject_id(self, path: str) -> str:
        return os.path.basename(path).split('.')[0]