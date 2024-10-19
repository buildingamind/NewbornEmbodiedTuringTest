import torch as th
from torch import nn
from .layers import conv1x1, conv3x3

class BasicBlock(nn.Module):
    """
    Basic block for ResNet

    Args:
        inplanes (int): Number of input channels
        planes (int): Number of output channels
        stride (int, optional): Stride for the first convolution. Defaults to 1.
        downsample (nn.Module, optional): Downsample layer. Defaults to None.
        groups (int, optional): Number of groups for the 3x3 convolution. Defaults to 1.
        base_width (int, optional): number of channels per group. Defaults to 64.
        dilation (int, optional): Dilation rate  for the 3x3 convolution. Defaults to 1.
        norm_layer (nn.Module, optional): Normalization layer. Defaults to None.
    """
    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        # layers inside each basic block of a residual block
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        # last two are operations and not layers
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Forward pass in the network

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        # saving x to pass over the bridge connection
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    """
    Bottleneck block for ResNet

    Args:
        inplanes (int): input channels
        planes (int): output channels
        stride (int, optional): stride. Defaults to 1.
        downsample (nn.Module, optional): downsample. Defaults to None.
        groups (int, optional): groups. Defaults to 1.
        base_width (int, optional): base width. Defaults to 64.
        dilation (int, optional): dilation. Defaults to 1.
        norm_layer ([type], optional): normalization layer. Defaults to None.
    """
    expansion = 4

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x) -> th.Tensor:
        """
        Forward pass

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



