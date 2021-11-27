import numpy as np
import copy
import warnings
from monai.metrics.confusion_matrix import get_confusion_matrix
import torch


def metric(y, y_pred):
    confusion_matrix = get_confusion_matrix(y_pred, y)

    tp, fp, tn, fn = torch.sum(confusion_matrix, (0, 1))

    eps = 0.001
    precision = tp / (torch.sum(y_pred) + eps)
    recall = tp / (torch.sum(y) + eps)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # TODO fix this (RuntimeWarning: invalid value encountered in double_scalars)
        f1 = 2 * (precision * recall) / (precision + recall + eps)

    fpr = fp / (fp + tn + eps)
    fnr = fn / (fn + tp + eps)

    return {
        'fpr': fpr,
        'fnr': fnr,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
