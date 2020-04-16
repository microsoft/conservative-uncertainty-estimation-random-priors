"""
Helper functions for the models.

Copyright 2019
Vincent Fortuin
Microsoft Research Cambridge
"""

import numpy as np
from core import *
from torch_backend import *
from collections import OrderedDict


def conv_bn(c_in, c_out, bn_weight_init=1.0, **kw):
    return {
        'conv': nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False),
        'bn': batch_norm(c_out, bn_weight_init=bn_weight_init, **kw),
        'relu': nn.ReLU(True)
    }


def residual(c, **kw):
    return {
        'in': Identity(),
        'res1': conv_bn(c, c, **kw),
        'res2': conv_bn(c, c, **kw),
        'add': (Add(), [rel_path('in'), rel_path('res2', 'relu')]),
    }


def basic_net(channels, weight, pool, num_outputs=10, **kw):
    return {
        'prep': conv_bn(3, channels['prep'], **kw),
        'layer1': dict(conv_bn(channels['prep'], channels['layer1'], **kw), pool=pool),
        'layer2': dict(conv_bn(channels['layer1'], channels['layer2'], **kw), pool=pool),
        'layer3': dict(conv_bn(channels['layer2'], channels['layer3'], **kw), pool=pool),

        'pool': nn.MaxPool2d(4),
        'flatten': Flatten(),
        'linear': nn.Linear(channels['layer3'], num_outputs),
        'out': Mul(weight),
    }


def large_net(channels, weight, pool, num_outputs=10, **kw):
    return {
        'prep': conv_bn(3, channels['prep'], **kw),
        'layer1': dict(conv_bn(channels['prep'], channels['layer1'], **kw), pool=pool),
        'layer2': dict(conv_bn(channels['layer1'], channels['layer2'], **kw), pool=pool),
        'layer3': dict(conv_bn(channels['layer2'], channels['layer3'], **kw), pool=pool),
        'layer4': dict(conv_bn(channels['layer3'], channels['layer4'], **kw)),
        'layer5': dict(conv_bn(channels['layer4'], channels['layer5'], **kw)),

        'pool': nn.MaxPool2d(4),
        'flatten': Flatten(),
        'linear': nn.Linear(channels['layer5'], num_outputs),
        'out': Mul(weight),
    }


def scale_prior(model, scaling_factor):
    old_params = model.prior.state_dict()
    new_params = OrderedDict()

    for k, v in old_params.items():
        if k.split(".")[-1] in ["weight", "bias"]:
            new_params[k] = v * scaling_factor
        else:
            new_params[k] = v

    model.prior.load_state_dict(new_params)


def net(channels=None, weight=1.0, pool=nn.MaxPool2d(2), extra_layers=(), output_size=10,
        res_layers=('layer1', 'layer3'), net_type="basic", **kw):
    channels = channels or {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512, 'layer4': 512, 'layer5': 512}
    if net_type == "basic":
        n = basic_net(channels, weight, pool, num_outputs=output_size, **kw)
    elif net_type == "large":
        n = large_net(channels, weight, pool, num_outputs=output_size, **kw)
    else:
        raise ValueError("Unknown net_type.")
    for layer in res_layers:
        n[layer]['residual'] = residual(channels[layer], **kw)
    for layer in extra_layers:
        n[layer]['extra'] = conv_bn(channels[layer], channels[layer], **kw)
    return n


class SpatialConcreteDropout(nn.Module):
    """
    Adapted from https://github.com/yaringal/ConcreteDropout
    """
    def __init__(self, weight_regularizer=1e-6,
                 dropout_regularizer=1e-5, init_min=0.1, init_max=0.1):
        super(SpatialConcreteDropout, self).__init__()

        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer

        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)

        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))

    def forward(self, x, layer=None):
        p = torch.sigmoid(self.p_logit)

        out = layer(self._spatial_concrete_dropout(x, p))

        sum_of_square = 0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))

        weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)

        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1. - p) * torch.log(1. - p)

        input_dimensionality = x[0].numel() # Number of elements of first item in batch
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality

        regularization = weights_regularizer + dropout_regularizer
        return out, regularization

    def _spatial_concrete_dropout(self, x, p):
        eps = 1e-4
        temp = 2./3.

        unif_noise = torch.rand((x.shape[0], x.shape[1], 1, 1), dtype=x.dtype, device=x.device) # batch, channel, H, W

        drop_prob = (torch.log(p + eps)
                    - torch.log(1 - p + eps)
                    + torch.log(unif_noise + eps)
                    - torch.log(1 - unif_noise + eps))

        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p

        x  = torch.mul(x, random_tensor)
        x /= retain_prob

        return x


def get_hparams(seed=0):
    np.random.seed(seed)

    m_space = [1,2,4,8,16,32,64,128,512,1024]
    init_space = [0.25, 0.5, 1., 2., 4.]
    lr_space = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    out_space = [0.1, 1., 10.]

    hparams = {}
    hparams["M"] = int(np.random.choice(m_space))
    hparams["init_scale"] = np.random.choice(init_space)
    hparams["lr"] = np.random.choice(lr_space)
    hparams["out_weight"] = np.random.choice(out_space)

    return hparams
