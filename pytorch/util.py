import logging
import tqdm
import numpy as np
import copy


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def metric(gt, pred):
    preds = pred.detach().numpy()
    gts = gt.detach().numpy()

    pred = preds.astype(int)
    gdth = gts.astype(int)
    fp_array = copy.deepcopy(pred)
    fn_array = copy.deepcopy(gdth)
    gdth_sum = np.sum(gdth)
    pred_sum = np.sum(pred)
    intersection = gdth & pred
    union = gdth | pred

    tp_array = intersection

    tmp = pred - gdth
    fp_array[tmp < 1] = 0

    tmp2 = gdth - pred
    fn_array[tmp2 < 1] = 0

    tn_array = np.ones(gdth.shape) - union

    tp, fp, fn, tn = np.sum(tp_array), np.sum(
        fp_array), np.sum(fn_array), np.sum(tn_array)

    eps = 0.001
    precision = tp / (pred_sum + eps)
    recall = tp / (gdth_sum + eps)
    f1 = 2 * (precision * recall) / (precision + recall)

    fpr = fp / (fp + tn + eps)
    fnr = fn / (fn + tp + eps)

    return {
        'fpr': fpr,
        'fnr': fnr,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
