import os
from typing import Dict, List, Union

import torch
from torchio import LabelMap, ScalarImage, Subject
from pprint import pformat
from torchio.data.dataset import SubjectsDataset
from torchio.transforms import Transform, OneHot
from torch.nn.functional import max_pool3d
import logging
from omegaconf import dictconfig
from tqdm.notebook import tqdm
from torchio.transforms import Resize
import numpy as np


class Dataset(SubjectsDataset):
    log = logging.getLogger(__name__)

    def __init__(self, cfg: dictconfig, transform: Transform, **kwargs):
        self.cfg = cfg

        super().__init__(
            subjects=self._get_subjects_list(),
            transform=transform,
            **kwargs
        )

    # TODO ABC
    def _create_data_map(self) -> Union[Dict[str, List[str]], Dict[str, str]]:
        ...

    def _get_subject_id(self, path: str) -> str:
        ...

    def _get_subjects_list(self) -> List[Subject]:
        if not os.path.exists(self.cfg['pt_path']):
            os.makedirs(self.cfg['pt_path'])

            data_map = self._create_data_map()

            for path, label_paths in tqdm(
                data_map.items(),
                total=len(data_map),
                position=0,
                leave=False,
                desc='Caching pooled data'
            ):
                subject_id = self._get_subject_id(path)

                image = ScalarImage(path, check_nans=True)
                label_map = LabelMap(label_paths, check_nans=True)
                if not isinstance(label_map, list):
                    one_hot = OneHot(self.cfg['num_classes'] + 1)
                    label_map = one_hot(label_map)
                    label_map = LabelMap(tensor=label_map.tensor[1:])

                image.load()
                label_map.load()

                image, label_map = self._resize(image, label_map)

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

    # TODO determine if needed
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

    def _resize(self, image: ScalarImage, label_map: LabelMap):
        # if min(image.spatial_shape) < min(self.cfg['desired_size']):
        #     return None
        res_subject = Resize(self.cfg['desired_size']).apply_transform(
            Subject(
                image=ScalarImage(tensor=image.data),
                seg=LabelMap(tensor=label_map.data)
            )
        )
        return res_subject['image'], res_subject['seg']
