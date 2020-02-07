from pathlib import Path

import torch
import torch.nn as nn


class Tumor3DNet(nn.Module):
    """
    Architecture inspired from :
    https://arxiv.org/abs/1911.06687
    """
    def __init__(
        self,
        in_channels,
        linear_in,
        conv1_filters=10,
        conv2_filters=10,
        dropout3d=.8,
        dropout=.5,
        conv_bias=True,
        conv_stride=2,
        conv_padding=0,
        hidden_layer=128,
        linear_bias=True,
        regress=True,
    ):
        super(Tumor3DNet, self).__init__()
        self.regress = regress

        self.bn = nn.BatchNorm3d(in_channels)
        self.conv1 = nn.Conv3d(
            in_channels, conv1_filters, (2, 2, 2),
            stride=conv_stride, padding=conv_padding, bias=conv_bias)
        self.bn1 = nn.BatchNorm3d(conv1_filters)
        self.conv2 = nn.Conv3d(
            conv1_filters, conv2_filters, (2, 2, 2),
            stride=conv_stride, padding=conv_padding, bias=conv_bias)
        self.bn2 = nn.BatchNorm3d(conv2_filters)
        self.dense = nn.Linear(linear_in, hidden_layer, bias=linear_bias)
        self.regressor = nn.Linear(hidden_layer, 1, bias=linear_bias)

        self.pooling = nn.MaxPool3d((2, 2, 2))
        self.activation = nn.ReLU()
        self.dropout3d = nn.Dropout3d(dropout3d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, images=None, clinical=None):
        # x = self.bn(images)
        # x = self.dropout3d(x)

        x = self.conv1(images)
        x = self.bn1(x)
        x = self.pooling(x)
        x = self.activation(x)
        x = self.dropout3d(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pooling(x)
        x = self.activation(x)
        x = self.dropout3d(x)

        x = x.view(x.size(0), -1)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)

        if self.regress:
            x = self.regressor(x)
        return x


class ClinicalNet(nn.Module):

    def __init__(
        self,
        hidden_layer=500,
        dropout=.5,
        regress=True,
        linear_bias=True,
    ):
        super(ClinicalNet, self).__init__()
        self.regress = regress

        layer_list = []
        for inf, outf in hidden_layer:
            layer_list += [
                nn.Linear(inf, outf, bias=linear_bias),
                nn.ReLU(),
                nn.BatchNorm1d(outf),
                # nn.Dropout(dropout),
            ]
        self.fc = nn.Sequential(*layer_list)
        self.regressor = nn.Linear(hidden_layer[-1][1], 1, bias=linear_bias)

    def forward(self, images=None, clinical=None):
        x = self.fc(clinical)

        if self.regress:
            x = self.regressor(x)
        return x


class MultiModalNet(nn.Module):

    def __init__(
        self,
        tumornet_kwargs,
        clinicalnet_kwargs=None,
        linear_in=256,
        linear_bias=True,
        tumornet_transfer=None,
        clinicalnet_transfer=None,
    ):
        super(MultiModalNet, self).__init__()

        tumornet_kwargs["regress"] = False
        clinicalnet_kwargs["regress"] = False

        self.tumornet = Tumor3DNet(**tumornet_kwargs)
        if clinicalnet_kwargs is None:
            clinicalnet_kwargs = {}
        self.clinicalnet = ClinicalNet(**clinicalnet_kwargs)
        self.regressor = nn.Linear(linear_in, 1, bias=linear_bias)

        if tumornet_transfer is not None:
            tumornet_transfer = Path(tumornet_transfer).expanduser()
            self.tumornet.load_state_dict(torch.load(tumornet_transfer))
        if clinicalnet_transfer is not None:
            clinicalnet_transfer = Path(clinicalnet_transfer).expanduser()
            self.clinicalnet.load_state_dict(torch.load(clinicalnet_transfer))

    def forward(self, images=None, clinical=None):
        images = self.tumornet(images=images)
        clinical = self.clinicalnet(clinical=clinical)

        x = torch.cat((images, clinical), dim=1)
        x = self.regressor(x)
        return x


if __name__ == '__main__':
    import datasets as D
    from utils import count_parameters
    d1 = D.Images3DDataset(
        "~/datasets/tumor/x_train/images/",
        csv_dir="~/datasets/tumor/y_train.csv")
    x1 = d1[0]['images'].float().unsqueeze(0).to('cpu')
    x1 = torch.cat((x1, x1), dim=0)
    net = Tumor3DNet(2, 1250).to('cpu')
    _ = net(images=x1)
    print(f"Tumor3DNet has {count_parameters(net)} trainable parameters")

    d2 = D.ClinicalDataset(
         "~/datasets/tumor/x_train/features/clinical_data.csv",
         "~/datasets/tumor/x_train/images/",
         csv_dir="~/datasets/tumor/y_train.csv",
         train_test="test")
    x2 = d2[0]['clinical'].float().unsqueeze(0).to('cpu')
    x2 = torch.cat((x2, x2), dim=0)
    net = ClinicalNet().to('cpu')
    _ = net(clinical=x2)
    print(f"ClinicalNet has {count_parameters(net)} trainable parameters")

    d3 = D.MultiModalDataset(
        {"images_dir": "~/datasets/tumor/x_train/images/",
         "csv_dir": "~/datasets/tumor/y_train.csv",
         "seed": 23, "train_test_split": .9, "train_test": "train"},
        {"data_path": "~/datasets/tumor/x_train/features/clinical_data.csv",
         "id_path": "~/datasets/tumor/x_train/images/",
         "csv_dir": "~/datasets/tumor/y_train.csv",
         "seed": 23, "train_test_split": .9, "train_test": "train"}
    )
    images = d3[0]['images'].float().unsqueeze(0).to('cpu')
    clinical = d3[0]['clinical'].float().unsqueeze(0).to('cpu')
    images = torch.cat((images, images), dim=0)
    clinical = torch.cat((clinical, clinical), dim=0)
    net = MultiModalNet(
        {"in_channels": 2, "linear_in": 1250},
        {},
    )
    _ = net(images=images, clinical=clinical)
    print(f"MultiModalNet has {count_parameters(net)} trainable parameters")
