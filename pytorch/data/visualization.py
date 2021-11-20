import torch
from torchio.data.image import LabelMap
from torchio.data.subject import Subject
from torchio.datasets.fpg import GIF_COLORS
import os


# TODO count labeled vs not labeled voxels

def plot_segmentation(image, seg, save_plot_path: str):
    if not os.path.exists(save_plot_path):
        os.makedirs(save_plot_path)

    squeezed_seg = torch.zeros_like(seg.data[0])
    for idx, label in enumerate(seg.data):
        seg_mask = squeezed_seg == 0
        squeezed_seg[seg_mask] += (idx + 1) * label[seg_mask]

    Subject(
        image=image,
        seg=LabelMap(tensor=squeezed_seg.unsqueeze(0))
    ).plot(output_path=save_plot_path)