import logging
import os

import numpy as np
import torch
from monai.metrics.cumulative_average import CumulativeAverage
from monai.visualize.class_activation_maps import GradCAM
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from monai.visualize.occlusion_sensitivity import OcclusionSensitivity
from tensorboardX import SummaryWriter
from torchio import GridAggregator, LabelMap, ScalarImage
from torchio.data.image import LabelMap
from torchio.data.subject import Subject
from torchio.datasets.fpg import GIF_COLORS
from torchio.visualization import import_mpl_plt
from util import batches_from_sampler, random_subject_from_loader


def squeeze_segmentation(seg):
    squeezed_seg = torch.zeros_like(seg.data[0])
    for idx, label in enumerate(seg.data):
        seg_mask = squeezed_seg == 0
        squeezed_seg[seg_mask] += (idx + 1) * label[seg_mask]
    return squeezed_seg.unsqueeze(0)


def plot_subject(subject: Subject, save_plot_path: str = None):
    if save_plot_path:
        os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)

    data_dict = {}
    for name, image in subject.get_images_dict(intensity_only=False).items():
        if isinstance(image, LabelMap):
            data_dict[name] = LabelMap(tensor=squeeze_segmentation(image))
        else:
            data_dict[name] = image

    Subject(data_dict).plot(output_path=save_plot_path)


def plot_aggregated_image(
        writer: SummaryWriter,
        epoch: int,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device
    ):
    sampler = random_subject_from_loader(data_loader)
    aggregator_x = GridAggregator(sampler)
    aggregator_y = GridAggregator(sampler)
    aggregator_y_pred = GridAggregator(sampler)
    for batch, locations in batches_from_sampler(sampler, data_loader.batch_size):
        x: torch.Tensor = batch['image']['data']
        aggregator_x.add_batch(x, locations)
        y: torch.Tensor = batch['seg']['data']
        aggregator_y.add_batch(y, locations)

        logits = model(x.to(device))
        y_pred = (torch.sigmoid(logits) > 0.5).float()
        aggregator_y_pred.add_batch(y_pred, locations)

    whole_x = aggregator_x.get_output_tensor()
    whole_y = aggregator_y.get_output_tensor()
    whole_y_pred = aggregator_y_pred.get_output_tensor()

    plot_subject(
        Subject(
            image=ScalarImage(tensor=whole_x),
            true_seg=LabelMap(tensor=whole_y),
            pred_seg=LabelMap(tensor=whole_y_pred)
        )
    )

    whole_x = torch.transpose(whole_x, 1, 3)
    whole_y = torch.transpose(squeeze_segmentation(whole_y), 1, 3)
    whole_y_pred = torch.transpose(squeeze_segmentation(whole_y_pred), 1, 3)

    whole_x = (whole_x - whole_x.min()) / (whole_x.max() - whole_x.min())

    def color_labels(array: np.ndarray) -> np.ndarray:
        max_label = int(array.max())
        _, plt = import_mpl_plt()
        cmap = plt.get_cmap('cubehelix')

        si, sj, sk = array.shape
        rgb: np.ndarray = np.zeros((si, sj, sk, 3), dtype=np.uint8)
        for label in range(1, max_label + 1):
            rgb[array == label] = cmap(label / max_label)[:3]
        return np.expand_dims(rgb.transpose((3, 0, 1, 2)), axis=0)

    plot_2d_or_3d_image(data=whole_x.repeat((3, 1, 1, 1)).unsqueeze(0), writer=writer, step=epoch, max_channels=3, max_frames=5, tag='training/visualization/image')
    plot_2d_or_3d_image(data=color_labels(whole_y.squeeze().detach().numpy()), writer=writer, step=epoch, max_channels=3, max_frames=5, tag='training/visualization/true_seg')
    plot_2d_or_3d_image(data=color_labels(whole_y_pred.squeeze().detach().numpy()), writer=writer, step=epoch, max_channels=3, max_frames=5, tag='training/visualization/pred_seg')


def train_visualizations(
        writer: SummaryWriter,
        epoch: int,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device
    ):
    log = logging.getLogger(__name__)
    log.info(f'visualisations for epoch: {epoch}')

    plot_aggregated_image(writer, epoch, model, data_loader, device)

    # FIXME modules not named (see ModelDict)
    # cam = CAM(nn_module=model, target_layers='bottleneck', fc_layers='output_conv')(torch.rand((1, 1, 64, 64, 32)).to(device))
    # add_animated_gif(writer, 'training/cam', cam, global_step=epoch)

    # FIXME out of memory
    # occ_map, most_probable_class = OcclusionSensitivity(nn_module=model)(torch.rand((1, 1, 64, 64, 32)).to(device))
    # add_animated_gif(writer, 'training/occlusion/map', occ_map, global_step=epoch)
    # add_animated_gif(writer, 'training/occlusion/most_probable_class', most_probable_class, global_step=epoch)
