import warnings
from monai.metrics.confusion_matrix import get_confusion_matrix
import torch
from torchio import GridSampler
import random
from torch.utils.data import DataLoader


def import_tqdm():
    try:
        if get_ipython().__class__.__name__ == "ZMQInteractiveShell" or "google.colab" in str(get_ipython()):  # type: ignore
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
    except NameError:
        from tqdm import tqdm

    return tqdm


metrics_map = ["acc", "fpr", "fnr", "precision", "recall", "f1"]

# TODO return dict
def metric(y, y_pred):
    acc = torch.sum(y == y_pred) / torch.numel(y)

    confusion_matrix = get_confusion_matrix(y_pred, y)

    tp, fp, tn, fn = torch.sum(confusion_matrix, (0, 1))

    eps = 0.001
    precision = tp / (torch.sum(y_pred) + eps)
    recall = tp / (torch.sum(y) + eps)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # FIXME (RuntimeWarning: invalid value encountered in double_scalars)
        f1 = 2 * (precision * recall) / (precision + recall + eps)

    fpr = fp / (fp + tn + eps)
    fnr = fn / (fn + tp + eps)

    return acc, fpr, fnr, precision, recall, f1


def random_subject_from_loader(data_loader):
    random_subject = data_loader.dataset.subjects_dataset[
        random.randrange(0, len(data_loader.dataset.subjects_dataset))
    ]
    return (
        GridSampler(
            subject=random_subject, patch_size=data_loader.dataset.sampler.patch_size
        ),
        random_subject["subject_id"],
    )


def batches_from_sampler(sampler, batch_size):
    loader = iter(DataLoader(sampler, batch_size=batch_size))
    idx = 0
    while idx < len(sampler):
        yield next(loader), torch.Tensor(
            sampler.locations[idx : idx + batch_size]
        ).int()
        idx += batch_size
