'''
This file contains the implementation of ResNet-18 with 3 blocks 
Output from the third block now gives 512 channels instead of 256
'''


import torch
from torch import nn as nn

from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

# if _TORCHVISION_AVAILABLE:
#     from torchvision.models.utils import load_state_dict_from_url
# else:  # pragma: no cover
#     warn_missing_pkg('torchvision')

__all__ = [
    'ResNet',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'resnext50_32x4d',
    'resnext101_32x8d',
    'wide_resnet50_2',
    'wide_resnet101_2',
]

MODEL_URLS = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1) -> nn.Conv2d:
    """
    3x3 convolution with padding

    Args:
        in_planes (int): number of input planes
        out_planes (int): number of output planes
        stride (int, optional): stride. Defaults to 1.
        groups (int, optional): number of groups. Defaults to 1.
        dilation (int, optional): dilation. Defaults to 1.

    Returns:
        nn.Conv2d: convolution layer
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )


def conv1x1(in_planes, out_planes, stride=1) -> nn.Conv2d:
    """
    1x1 convolution
    
    Args:
        in_planes (int): number of input planes
        out_planes (int): number of output planes
        stride (int, optional): stride. Defaults to 1.
    
    Returns:
        nn.Conv2d: convolution layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
    Basic block for ResNet

    Args:
        inplanes (int): number of input planes
        planes (int): number of planes
        stride (int, optional): stride. Defaults to 1.
        downsample (nn.Module, optional): downsample. Defaults to None.
        groups (int, optional): number of groups. Defaults to 1.
        base_width (int, optional): base width. Defaults to 64.
        dilation (int, optional): dilation. Defaults to 1.
        norm_layer (nn.Module, optional): normalization layer. Defaults to None.
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

    def forward(self, x) -> torch.Tensor:
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


class ResNet(nn.Module):
    """
    ResNet model

    Args:
        block (nn.Module): block type
        layers (list): list of layers
        num_classes (int, optional): number of classes. Defaults to 1000.
        zero_init_residual (bool, optional): If True, zero-initialize the last BN in each residual branch. Defaults to False.
        groups (int, optional): number of groups. Defaults to 1.
        width_per_group (int, optional): width per group. Defaults to 64.
        replace_stride_with_dilation (tuple, optional): replace stride with dilation. Defaults to None.
        norm_layer (nn.Module, optional): normalization layer. Defaults to None.
        return_all_feature_maps (bool, optional): If True, returns all feature maps. Defaults to False.
        first_conv (bool, optional): If True, uses a 7x7 kernel for the first convolution. Defaults to True.
        maxpool1 (bool, optional): If True, uses a maxpool layer after the first convolution. Defaults to True.
    """

    def __init__(
        self,
        block,
        layers,
        num_classes=1000, # what should be the right parameter?
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        return_all_feature_maps=False,
        first_conv=True, #pre-processing layers which makes the image size half [64->32]
        maxpool1=True, #pre-processing 
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.return_all_feature_maps = return_all_feature_maps

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        
        # ------ layers before first residual block ---------------

        if first_conv:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        if maxpool1:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
            
        # ------ residual blocks start here ------------------------

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
            
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False) -> nn.Sequential:
        """
        Create a layer of residual blocks
        
        Args:
            block (nn.Module): block type
            planes (int): number of planes
            blocks (int): number of blocks
            stride (int): stride
            dilate (bool): If True, use dilation

        Returns:
            nn.Sequential: layer of residual blocks
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass in the network
        
        Args:
            x (torch.Tensor): input tensor
        
        Returns:
            torch.Tensor: output tensor
        """
        
        # passing input from pre-processing layers
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)
        
        # passing input from residual blocks
        if self.return_all_feature_maps:
            x1 = self.layer1(x0)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            

            return [x0, x1, x2, x3]

        else:
            x0 = self.layer1(x0)
            x0 = self.layer2(x0)
            x0 = self.layer3(x0)
            

            x0 = self.avgpool(x0)
            x0 = torch.flatten(x0, 1)

            return x0


def _resnet(arch, block, layers, pretrained, progress, **kwargs) -> ResNet:
    """
    Constructs a ResNet model.

    Args:
        arch (str): model architecture
        block (nn.Module): block type
        layers (list): list of layers
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        ResNet: model
    """
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(MODEL_URLS[arch], progress=progress)
        model.load_state_dict(state_dict)
    # Remove the last fc layer, since we only need the encoder part of resnet.
    model.fc = nn.Identity()
    return model


def resnet_3blocks(pretrained: bool = False, progress: bool = True, **kwargs) -> ResNet:
    """ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        progress: If True, displays a progress bar of the download to stderr
    
    Returns:
        ResNet: model
    """
    
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)