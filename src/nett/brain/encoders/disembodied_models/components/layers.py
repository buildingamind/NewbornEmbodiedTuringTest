
from torch.nn import Conv2d

def conv1x1(in_planes, out_planes, stride=1) -> Conv2d:
    """
    1x1 convolution

    Args:
        in_planes (int): Number of input channels
        out_planes (int): Number of output channels
        stride (int, optional): Stride for the convolution

    Returns:
        nn.Conv2d: 1x1 convolution layer"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1) -> Conv2d:
    """
    3x3 convolution with padding

    Args:
        in_planes (int): Number of input channels
        out_planes (int): Number of output channels
        stride (int, optional): Stride
        groups (int, optional): Number of groups
        dilation (int, optional): Dilation

    Returns:
        nn.Conv2d: 3x3 convolution layer
    """
    return Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )
