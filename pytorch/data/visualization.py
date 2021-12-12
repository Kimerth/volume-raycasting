import logging
import os
from glob import glob

import imageio
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torchio import GridAggregator, LabelMap, ScalarImage
from torchio.data.image import LabelMap
from torchio.data.subject import Subject
from util import batches_from_sampler, random_subject_from_loader
from torchio.visualization import import_mpl_plt


def squeeze_segmentation(seg):
    squeezed_seg = torch.zeros_like(seg.data[0])
    for idx, label in enumerate(seg.data):
        seg_mask = squeezed_seg == 0
        squeezed_seg[seg_mask] += (idx + 1) * label[seg_mask]
    return squeezed_seg.unsqueeze(0)


def create_gifs(path_in: str, path_out: str):
    filenames = [path for path in glob(f'{path_in}/*') if os.path.isfile(path)]
    filenames.sort()

    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(path_out, images)


def plot_subject(subject: Subject, save_plot_path: str = None):
    if save_plot_path:
        os.makedirs(save_plot_path, exist_ok=True)

    data_dict = {}
    sx, sy, sz = subject.spatial_shape
    sx, sy, sz = min(sx, sy, sz) / sx, min(sx, sy, sz) / sy, min(sx, sy, sz) / sz
    for name, image in subject.get_images_dict(intensity_only=False).items():
        if isinstance(image, LabelMap):
            data_dict[name] = LabelMap(tensor=squeeze_segmentation(image), affine=np.eye(4) * np.array([sx, sy, sz, 1]))
        else:
            data_dict[name] = ScalarImage(tensor=image.data, affine=np.eye(4)* np.array([sx, sy, sz, 1]))

    out_subject = Subject(data_dict)
    out_subject.plot(reorient=False, show=True)

    for x in range(min(out_subject.spatial_shape)):
        _, plt = import_mpl_plt()
        plt.ioff()
        out_subject.plot(reorient=False, indices= (x,x,x), output_path=f'{save_plot_path}/{x:03d}.png', show=False)
        plt.ion()

    out_gif_path = f'{save_plot_path}/{os.path.basename(save_plot_path)}.gif'
    create_gifs(save_plot_path, out_gif_path)

    return out_gif_path


def plot_aggregated_image(
        writer: SummaryWriter,
        epoch: int,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        save_path: str
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

    out_gif_path = plot_subject(
        Subject(
            image=ScalarImage(tensor=whole_x),
            true_seg=LabelMap(tensor=whole_y),
            pred_seg=LabelMap(tensor=whole_y_pred)
        ),
        f'{save_path}/{epoch}'
    )

    # FIXME
    # gif_np_array = imageio.mimread(out_gif_path)
    # writer.add_video('training/visualization', gif_np_array, global_step=epoch)


def train_visualizations(
        writer: SummaryWriter,
        epoch: int,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        save_path: str
    ):
    log = logging.getLogger(__name__)
    log.info(f'visualisations for epoch: {epoch}')

    plot_aggregated_image(writer, epoch, model, data_loader, device, save_path)
