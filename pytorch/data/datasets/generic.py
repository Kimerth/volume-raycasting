import logging
import os
from typing import Dict, List, Union

import multitasking
import torch
from omegaconf.dictconfig import DictConfig
from torchio import LabelMap, ScalarImage, Subject
from torchio.data.dataset import SubjectsDataset
from torchio.transforms import OneHot, Resize, Transform
from util import import_tqdm

tqdm = import_tqdm()


class Dataset(SubjectsDataset):
    log = logging.getLogger(__name__)

    def __init__(self, cfg: DictConfig, transform: Transform, **kwargs):
        self.cfg = cfg

        super().__init__(
            subjects=self._get_subjects_list(), transform=transform, **kwargs
        )

    def _create_data_map(self) -> Union[Dict[str, List[str]], Dict[str, str]]:
        ...

    def _get_subject_id(self, path: str) -> str:
        ...

    def _get_subjects_list(self) -> List[Subject]:
        if not os.path.exists(self.cfg["cache_path"]):
            self._generate_cache()
            multitasking.wait_for_tasks()

        subjects = []
        for pt_name in os.listdir(self.cfg["cache_path"]):
            pt_data = torch.load(os.path.join(self.cfg["cache_path"], pt_name))

            subjects.append(
                Subject(
                    subject_id=pt_data["subject_id"],
                    image=ScalarImage(tensor=pt_data["image"]),
                    seg=LabelMap(tensor=pt_data["seg"]),
                )
            )
        return subjects

    @multitasking.task
    def _generate_cache(self):
        os.makedirs(self.cfg["cache_path"])

        data_map = self._create_data_map()

        for path, label_paths in tqdm(
            data_map.items(),
            total=len(data_map),
            position=0,
            leave=False,
            desc="Caching pooled data",
        ):
            subject_id = self._get_subject_id(path)

            image = ScalarImage(path, check_nans=True)
            label_map = LabelMap(label_paths, check_nans=True)
            if not isinstance(label_map, list):
                one_hot = OneHot(self.cfg["num_classes"] + 1)
                label_map = one_hot(label_map)
                label_map = LabelMap(tensor=label_map.tensor[1:])  # type: ignore

            image.load()
            label_map.load()

            image, label_map = self._resize(image, label_map)

            torch.save(
                {"subject_id": subject_id, "image": image.data, "seg": label_map.data},
                os.path.join(self.cfg["cache_path"], f"{subject_id}.pt"),
            )

    def _resize(self, image: ScalarImage, label_map: LabelMap):
        res_subject = Resize(self.cfg["desired_size"]).apply_transform(
            Subject(
                image=ScalarImage(tensor=image.data),
                seg=LabelMap(tensor=label_map.data),
            )
        )
        return res_subject["image"], res_subject["seg"]
