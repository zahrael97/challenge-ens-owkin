import sys
from collections.abc import Iterable

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.transforms import ToPILImage
from sklearn.utils.extmath import _incremental_mean_and_var


class Convert:

    def __init__(self):
        self.to_pil = ToPILImage()

    def __call__(self, pic):
        return self.to_pil(cv2.normalize(pic, None, 0, 255, cv2.NORM_MINMAX))

    def __repr__(self):
        return self.__class__.__name__ + '()'


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input_tensor):
        return input_tensor.view(*self.shape)


def get_id_from_path(pth):
    return pth.split('.')[-2].split('_')[-1]


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class IncrementalMeanVar:
    def __init__(self):
        self.mean = 0
        self.var = 0
        self.count = 0

    def update(self, values):
        self.mean, self.var, self.count =\
            _incremental_mean_and_var(self.mean, self.var, self.count)


def get_hms(seconds):
    seconds = int(seconds)
    minutes = seconds // 60
    rseconds = seconds - 60 * minutes
    return '{}m{}s'.format(minutes, rseconds)


def create_experiment_name(name, parameters):
    for key, val in parameters.items():
        if not isinstance(val, Iterable):
            name += f"_{key}_{val}"
    return name


def _lower(x):
    if isinstance(x, str):
        return x.lower()
    if isinstance(x, pd.Series):
        try:
            return x.str.lower()
        except AttributeError:
            return x
    return x


def count_parameters(model):
    "https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7"
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_difference_plot(y_true, y_pred):
    fig = plt.figure()
    plt.xticks([])
    ids = y_true.index.values
    plt.scatter(ids, y_true.SurvivalTime - y_pred.SurvivalTime)
    plt.title("Difference between true and prediction values")
    return fig


def create_txt_config(model, optimizer, dataset):
    return f"""
    Executed command
    ================
    {" ".join(sys.argv)}

    Dataset
    =======
    {dataset}

    Model summary
    =============
    {model}

    {count_parameters(model)} trainable parameters

    Optimizer
    ========
    {optimizer}
    """
