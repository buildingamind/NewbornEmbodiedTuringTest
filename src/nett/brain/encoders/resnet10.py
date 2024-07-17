"""
Resnet10CNN feature extractor for stable-baselines3
"""
import pdb
import gymnasium


import torch as th
import torch.nn as nn
import torchvision

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import logging
logger = logging.getLogger(__name__)

class Resnet10CNN(BaseFeaturesExtractor):
    """
    Resnet10CNN feature extractor for stable-baselines3

    Args:
        observation_space (gymnasium.spaces.Box): Observation space
        features_dim (int, optional): Output dimension of features extractor. Defaults to 256.
    """

    def __init__(self, observation_space: gymnasium.spaces.Box, features_dim: int = 256):
        super(Resnet10CNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        n_input_channels = observation_space.shape[0]

        self.cnn = _resnet(BasicBlock, [2, 2, 2, 2],num_channels = n_input_channels)
        logger.info(f"Resnet10CNN Encoder: {self.cnn}")
        
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Forward pass of the feature extractor.

        Args:
            observations (torch.Tensor): The input observations.

        Returns:
            torch.Tensor: The extracted features.
        """
        # Cut off image
        # reshape to from vector to W*H
        # gray to color transform
        # application of ResNet
        # Concat features to the rest of observation vector
        # return
        return self.linear(self.cnn(observations))
    
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1) -> nn.Conv2d:
    """
    3x3 convolution with padding

    Args:
        in_planes (int): Number of input channels
        out_planes (int): Number of output channels
        stride (int, optional): Stride of the convolution. Defaults to 1.
        groups (int, optional): Number of groups for the convolution. Defaults to 1.
        dilation (int, optional): Dilation rate of the convolution. Defaults to 1.

    Returns:
        nn.Conv2d: Convolutional layer
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


def conv1x1(in_planes, out_planes, stride=1):
    """
    1x1 convolution

    Args:
        in_planes (int): Number of input channels
        out_planes (int): Number of output channels
        stride (int, optional): Stride of the convolution. Defaults to 1.

    Returns:
        nn.Conv2d: Convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
    Basic block used in the ResNet-18 architecture.
    
    Args:
        inplanes (int): Number of input channels
        planes (int): Number of output channels
        stride (int, optional): Stride of the convolution. Defaults to 1.
        downsample (nn.Module, optional): Downsample layer. Defaults to None.
        groups (int, optional): Number of groups for the convolution. Defaults to 1.
        base_width (int, optional): Base width for the convolution. Defaults to 64.
        dilation (int, optional): Dilation rate of the convolution. Defaults to 1.
        norm_layer ([type], optional): Normalization layer. Defaults to None.
    """
    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None
    ):
        super(BasicBlock, self).__init__()
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

class ResNet(nn.Module):
    """
    ResNet architecture used in the Resnet10CNN class.

    Args:
        block (nn.Module): Residual block to use
        layers (list): Number of layers in each block
        num_channels (int): Number of input channels
        num_classes (int, optional): Number of classes. Defaults to 1000.
        zero_init_residual (bool, optional): Zero initialization for the residual block. Defaults to False.
        groups (int, optional): Number of groups for the convolution. Defaults to 1.
        width_per_group (int, optional): Base width for the convolution. Defaults to 64.
        replace_stride_with_dilation (tuple, optional): Replace stride with dilation. Defaults to None.
        norm_layer ([type], optional): Normalization layer. Defaults to None.
        return_all_feature_maps (bool, optional): Return all feature maps. Defaults to False.
        first_conv (bool, optional): Pre-processing layers which makes the image size half [64->32]. Defaults to True.
        maxpool1 (bool, optional): Used in pre-processing. Defaults to True.
    """

    def __init__(
        self,
        block,
        layers,
        num_channels,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        return_all_feature_maps=False,
        first_conv=True, # pre-processing layers which makes the image size half [64->32]
        maxpool1=True # used in pre-processing
    ):
        super(ResNet, self).__init__()
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
            self.conv1 = nn.Conv2d(num_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        
        
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        if maxpool1:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

        # ------ residual blocks start here ------------------------

        # BLOCK - 1
        self.layer1 = self._make_layer(block, 64, layers[0])
        
        # BLOCK - 2
        self.layer2 = self._make_layer(block, 512, layers[1], stride=2, dilate=replace_stride_with_dilation[0])

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
        Helper function to create a residual layer.

        Args:
            block (nn.Module): Residual block to use
            planes (int): Number of output channels
            blocks (int): Number of blocks
            stride (int, optional): Stride of the convolution. Defaults to 1.
            dilate (bool, optional): Use dilation. Defaults to False.

        Returns:
            nn.Sequential: Residual layer
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

    def forward(self, x: th.Tensor) -> th.Tensor:
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
            x1 = self.layer1(x0) # block1
            x2 = self.layer2(x1) # block2

            return [x0, x1, x2]
        else:
            x0 = self.layer1(x0)
            x0 = self.layer2(x0)

            x0 = self.avgpool(x0)
            x0 = th.flatten(x0, 1)

            return x0


def _resnet(block, layers, **kwargs):
    """
    ResNet architecture used in the Resnet10CNN class.

    Args:
        block (nn.Module): Residual block to use
        layers (list): Number of layers in each block

    Returns:
        ResNet: ResNet model
    """
    model = ResNet(block, layers, **kwargs)
    model.fc = nn.Identity()
    return model
