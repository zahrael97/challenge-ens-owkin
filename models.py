import torch.nn as nn


class Tumor3DConv(nn.Module):

    def __init__(
        self,
        in_channels,
        linear_in,
        conv1_filters=10,
        conv2_filters=10,
        dropout3d=.8,
        conv_bias=True,
        conv_stride=2,
        conv_padding=0,
        linear_bias=True,
        regress=True,
    ):
        super(Tumor3DConv, self).__init__()
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

        self.regressor = nn.Linear(linear_in, 1, bias=linear_bias)

        self.pooling = nn.MaxPool3d((2, 2, 2))
        self.activation = nn.ReLU()
        self.dropout3d = nn.Dropout3d(dropout3d)

    def forward(self, x):
        x = self.bn(x)
        x = self.dropout3d(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pooling(x)
        x = self.activation(x)
        x = self.dropout3d(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pooling(x)
        x = self.activation(x)
        x = self.dropout3d(x)

        if self.regress:
            x = x.view(x.size(0), -1)
            x = self.regressor(x)
        return x
