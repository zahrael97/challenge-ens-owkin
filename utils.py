import cv2
import numpy as np
import pandas as pd
import torch.nn as nn
from torchvision.transforms import ToPILImage


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


def weights_init(layer):
    if isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0.0, 0.02)
    elif isinstance(layer, nn.Conv2d):
        layer.weight.data.normal_(0.0, 0.02)
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)


class AverageMeter(object):
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


def get_hms(seconds):
    seconds = int(seconds)
    minutes = seconds // 60
    rseconds = seconds - 60 * minutes
    return '{}m{}s'.format(minutes, rseconds)


def make_metric_dataframe(patient_info, survival_time):
    if len(survival_time.shape) == 1:
        survival_time = np.expand_dims(survival_time, axis=1)
    df = np.hstack((patient_info, survival_time))
    df = pd.DataFrame(df, columns=["PatientID", "Event", "SurvivalTime"])
    df = df.set_index('PatientID')
    return df
