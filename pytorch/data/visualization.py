import torch
from torchio.data.image import LabelMap
from torchio.data.subject import Subject
from torchio.datasets.fpg import GIF_COLORS
import os


# TODO count labeled vs not labeled voxels

def plot_subject(subject: Subject, save_plot_path: str):
    squeezed_labels = torch.zeros_like(subject['labels'].data[0])
    for idx, label in enumerate(subject['labels'].data):
        squeezed_labels += (idx + 1) * label

    Subject(
        source=subject['source'],
        labels=LabelMap(tensor=squeezed_labels.unsqueeze(0))
    ).plot(
        output_path=os.path.join(
            save_plot_path,
            subject['name']
        ),
        cmap_dict={
            # TODO replace colors (also add legend for colors)
            'labels': GIF_COLORS
        }
    )