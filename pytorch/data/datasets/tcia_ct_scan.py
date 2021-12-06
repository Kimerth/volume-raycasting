import os
from glob import glob
from typing import Dict, List

import torch
from torchio import LabelMap, ScalarImage, Subject
from pprint import pformat
from torchio.data.dataset import SubjectsDataset
from torchio.transforms.transform import Transform
from torch.nn.functional import max_pool3d
import logging
from omegaconf import dictconfig
from tqdm.notebook import tqdm


class Dataset(SubjectsDataset):
    log = logging.getLogger(__name__)

    def __init__(self, cfg: dictconfig, transform: Transform, **kwargs):
        self.cfg = cfg

        super().__init__(
            subjects=self._get_subjects_list(),
            transform=transform,
            **kwargs
        )

    def _get_subjects_list(self) -> List[Subject]:
        scan_pattern: str = self.cfg['base_path'] + self.cfg['scan_pattern']
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

        if not os.path.exists(self.cfg['pt_path']):
            os.makedirs(self.cfg['pt_path'])

            data_map = _create_data_map()
            Dataset.log.debug(
                pformat({
                    os.path.basename(os.path.dirname(volume_path)):[
                        os.path.basename(label_path) for label_path in labels
                    ]
                    for (volume_path, labels) in data_map.items()
                })
            )

            for path, label_paths in tqdm(
                data_map.items(),
                total=len(data_map),
                position=0,
                leave=False,
                desc='Caching pooled data'
            ):
                subject_id = os.path.basename(os.path.dirname(path))

                image = ScalarImage(path, check_nans=True)
                label_map = LabelMap(label_paths, check_nans=True)

                image.load()
                label_map.load()

                image, label_map = self._max_pool_data(image, label_map)

                torch.save(
                    {
                        'subject_id': subject_id,
                        'image': image.data,
                        'seg': label_map.data
                    },
                    os.path.join(self.cfg['pt_path'], f'{subject_id}.pt')
                )

        subjects = []
        for pt_name in os.listdir(self.cfg['pt_path']):
            pt_data = torch.load(
                os.path.join(self.cfg['pt_path'], pt_name)
            )

            subjects.append(
                Subject(
                    subject_id=pt_data['subject_id'],
                    image=ScalarImage(tensor=pt_data['image']),
                    seg=LabelMap(tensor=pt_data['seg'])
                )
            )
        return subjects

    def _max_pool_data(self, image: ScalarImage, label_map: LabelMap):
        def retrieve_elements_from_indices(tensor, indices):
            flattened_tensor = tensor.flatten(start_dim=1)
            repeated_indices = torch.repeat_interleave(indices, tensor.shape[0], 0)
            flattened_indices = repeated_indices.flatten(start_dim=1)
            return flattened_tensor.gather(dim=1, index=flattened_indices).view_as(repeated_indices)

        output, indices = max_pool3d(
            image.data.to(torch.float), tuple(self.cfg['kernel_size']), return_indices=True
        )
        image.data = output.to(torch.int16)
        label_map.data = retrieve_elements_from_indices(label_map.data, indices)

        return image, label_map
