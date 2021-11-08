import os
from glob import glob
from typing import Dict, List

from torchio import LabelMap, ScalarImage, Subject
from pprint import pformat
from torchio.data.dataset import SubjectsDataset
from torchio.transforms.transform import Transform
import logging


class Dataset(SubjectsDataset):
    log = logging.getLogger(__name__)

    def __init__(self, scan_pattern: str, transform: Transform, **kwargs):
        super().__init__(
            subjects=self._get_subjects_list(scan_pattern),
            transform=transform,
            **kwargs
        )

    @staticmethod
    def _get_subjects_list(scan_pattern: str) -> List[Subject]:
        Dataset.log.debug(f'scan_pattern={scan_pattern}')

        def _create_data_map() -> Dict[str, List[str]]:
            return {
                image_path: glob(
                    os.path.join(
                        os.path.dirname(image_path),
                        'segmentations',
                        '*'
                    )
                ) for image_path in glob(scan_pattern, recursive=True)
            }

        data_map = _create_data_map()
        Dataset.log.debug(
            pformat({
                os.path.basename(os.path.dirname(volume_path)):[
                    os.path.basename(label_path) for label_path in labels
                ]
                for (volume_path, labels) in data_map.items()
            })
        )

        return [
            Subject(
                    name=os.path.basename(os.path.dirname(path)),
                    source=ScalarImage(path, check_nans=True),
                    labels=LabelMap(label_paths, check_nans=True)
            ) for path, label_paths in data_map.items()
        ]
