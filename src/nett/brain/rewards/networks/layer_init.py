import math

import numpy as np
import torch as th

def orthogonal_layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)
    return layer

def default_layer_init(layer):
    stdv = 1. / math.sqrt(layer.weight.size(1))
    layer.weight.data.uniform_(-stdv, stdv)
    if layer.bias is not None:
        layer.bias.data.uniform_(-stdv, stdv)
    return layer

def kaiming_he_init(layer):
    th.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
    if layer.bias is not None:
        th.nn.init.zeros_(layer.bias)
    return layer
