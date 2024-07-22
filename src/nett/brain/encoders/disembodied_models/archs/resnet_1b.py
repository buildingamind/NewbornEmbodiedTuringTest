import torch
from torch import nn as nn

from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

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
        in_planes (int): number of input channels
        out_planes (int): number of output channels
        stride (int): stride for the convolution
        groups (int): number of groups for the convolution
        dilation (int): dilation rate for the convolution
    
    Returns:
        nn.Conv2d: 3x3 convolution layer
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
        in_planes (int): number of input channels
        out_planes (int): number of output channels
        stride (int): stride for the convolution
    
    Returns:
        nn.Conv2d: 1x1 convolution layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
    Basic block for ResNet

    Args:
        inplanes (int): number of input channels
        planes (int): number of output channels
        stride (int): stride for the first convolution
        downsample (nn.Module): downsample layer
        groups (int): number of groups for the 3x3 convolution
        base_width (int): number of channels per group
        dilation (int): dilation rate for the 3x3 convolution
        norm_layer (nn.Module): normalization layer
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
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass in the network

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

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """
    ResNet model

    Args:
        block (nn.Module): the block type to be used in the network
        layers (list of int): the number of layers for each block
        num_classes (int): number of classes
        zero_init_residual (bool): whether the residual block should be initialized to zero
        groups (int): number of groups for the 3x3 convolution
        width_per_group (int): number of channels per group
        replace_stride_with_dilation (tuple): replace stride with dilation
        norm_layer (nn.Module): normalization layer
        return_all_feature_maps (bool): whether to return all feature maps
        first_conv (bool): whether to use the first convolution
        maxpool1 (bool): whether to use the first maxpool
    """

    def __init__(
        self,
        block, # what kind of block, for ex - basic block or bottleneck
        layers, # how many layers or basic blocks in each residual block
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        return_all_feature_maps=False,
        first_conv=False,
        maxpool1=False 
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.return_all_feature_maps = return_all_feature_maps
        
        
        
        self.inplanes = 3 # what is inplanes and planes in CNN ???? it should be 64 as per original implementation, does not work with 64 in case of block 1
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
       
        self.layer1 = self._make_layer(block, 512, layers[0])

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
        Make a layer of blocks.

        Args:
            block (nn.Module): the block type to be used in the layer
            planes (int): number of output channels for the layer
            blocks (int): number of blocks to be used
            stride (int): stride for the first block. Default: 1
            dilate (bool): whether to apply dilation strategy to the layer. Default: False

        Returns:
            nn.Sequential: a layer of blocks
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
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)
        #print(x0.shape)
        if self.return_all_feature_maps:
            x1 = self.layer1(x)
            return [x0, x1]
        else:
            x0 = self.layer1(x)
  
            x0 = self.avgpool(x0) # output shape = [256X1X1]

            x0 = x0.reshape(x0.shape[0],-1)
            return x0


def _resnet(arch, block, layers, pretrained, progress, **kwargs) -> ResNet:
    """
    Constructs a ResNet model.

    Args:
        arch (str): model architecture
        block (nn.Module): the block type to be used in the network
        layers (list of int): the number of layers for each block
        pretrained (bool): if True, returns a model pre-trained on ImageNet
        progress (bool): if True, displays a progress bar of the download to stderr

    Returns:
        ResNet: model
    """
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(MODEL_URLS[arch], progress=progress)
        model.load_state_dict(state_dict)
    # Remove the last fc layer, since we only need the encoder part of resnet.
    #model.fc = nn.Identity()
    
    # we cannot remove the last fc layer in smaller version of the resnet because we are using the fc layer
    # to flatten the output and making it equal dimensions for testing and training the classifier.
    return model


def resnet_1block(pretrained: bool = False, progress: bool = True, **kwargs) -> ResNet:
    """ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        progress: If True, displays a progress bar of the download to stderr

    Returns:
        ResNet: ResNet-18 model
    """

    return _resnet('resnet18', BasicBlock, [1, 1, 1, 1], pretrained, progress, **kwargs)

