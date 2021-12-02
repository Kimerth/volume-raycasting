import torch
from torchio.data.image import LabelMap
from torchio.data.subject import Subject
from torchio.datasets.fpg import GIF_COLORS
import os


def squeeze_segmentation(seg):
    squeezed_seg = torch.zeros_like(seg.data[0])
    for idx, label in enumerate(seg.data):
        seg_mask = squeezed_seg == 0
        squeezed_seg[seg_mask] += (idx + 1) * label[seg_mask]
    return squeezed_seg.unsqueeze(0)


def plot_subject(subject: Subject, save_plot_path: str):
    os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)

    data_dict = {}
    for name, image in subject.get_images_dict(intensity_only=False).items():
        if isinstance(image, LabelMap):
            data_dict[name] = LabelMap(tensor=squeeze_segmentation(image))
        else:
            data_dict[name] = image

    Subject(data_dict).plot(output_path=save_plot_path)