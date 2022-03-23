import torch
import numpy as np


from . import ConvBlock, YOLOLayer


def parse_conv(conv, weights, offset):
    """
    Args:
        conv (nn.Conv2d): pytorch convolution module.
        weights (numpy.ndarray): pretrained weights data.
        offset (int): current position in the weights file.
    Returns:
        offset (int): new position in the weights file.
    """

    # bias
    if hasattr(conv, 'bias') and conv.bias is not None:
        param_length = conv.bias.numel()
        param = torch.from_numpy(weights[offset:offset + param_length])
        conv.bias.data.copy_(param.view_as(conv.bias))
        offset += param_length

    # conv
    param_length = conv.weight.numel()
    param = torch.from_numpy(weights[offset:offset + param_length])
    conv.weight.data.copy_(param.view_as(conv.weight))
    offset += param_length
    return offset


def parse_bn(bn, weights, offset):
    """
    Args:
        bn (nn.Batchnorm2d): pytorch batch normalization module.
        weights (numpy.ndarray): pretrained weights data.
        offset (int): current position in the weights file.
    Returns:
        offset (int): new position in the weights file.
    """
    param_length = bn.bias.numel()
    for name in ['bias', 'weight', 'running_mean', 'running_var']:
        layerparam = getattr(bn, name)
        param = torch.from_numpy(weights[offset: offset + param_length])
        layerparam.data.copy_(param.view_as(layerparam))
        offset += param_length
    return offset


def parse_weights(model, weights_path):
    """
    Parse darknet pre-trained weights data to the pytorch model.
    Args:
        model (nn.Module) : pytorch module object.
        weights_path (str): path to the darknet pre-trained weights file.
    """
    with open(weights_path, "rb") as f:
        # skip the header
        _ = np.fromfile(f, dtype=np.int32, count=5)
        weights = np.fromfile(f, dtype=np.float32)

    offset = 0
    for name, module in model.named_modules():

        if isinstance(module, ConvBlock):
            # print(name)
            offset = parse_bn(module.bn, weights, offset)
            offset = parse_conv(module.conv, weights, offset)

        if isinstance(module, YOLOLayer):
            # print(name)
            offset = parse_conv(module.conv, weights, offset)

        if offset == len(weights):
            # if the pretrained model is part of network
            break
        assert offset < len(weights), "The model is larger than the weights"

    # the wieghts should be exactly consumed
    assert offset == len(weights), "The weights is larger than the model"
    del weights
