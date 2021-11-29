import warnings
from monai.metrics.confusion_matrix import get_confusion_matrix
import torch

metrics_dict = ['acc,' 'fpr', 'fnr', 'precision', 'recall', 'f1']

def metric(y, y_pred):
    acc = torch.sum(y == y_pred) / torch.numel(y)

    confusion_matrix = get_confusion_matrix(y_pred, y)

    tp, fp, tn, fn = torch.sum(confusion_matrix, (0, 1))

    eps = 0.001
    precision = tp / (torch.sum(y_pred) + eps)
    recall = tp / (torch.sum(y) + eps)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # FIXME (RuntimeWarning: invalid value encountered in double_scalars)
        f1 = 2 * (precision * recall) / (precision + recall + eps)

    fpr = fp / (fp + tn + eps)
    fnr = fn / (fn + tp + eps)

    return acc, fpr, fnr, precision, recall, f1
