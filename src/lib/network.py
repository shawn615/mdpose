import abc
import math
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as func
from torchvision.ops import MultiScaleRoIAlign, roi_align
from .backbone import get_backbone_dict
from .external.non_local_block.embedded_gaussian import _NonLocalBlockND
from . import network_util as net_util
from . import util as lib_util

import matplotlib.pyplot as plt

def get_network_dict():
    return {
        'mmpe': MMPENetwork,
    }


class NetworkABC(abc.ABC, nn.Module):
    def __init__(self, global_args, network_args, loss_func):
        super(NetworkABC, self).__init__()
        self.global_args = global_args
        self.network_args = network_args
        self.loss_func = loss_func
        self.net = nn.ModuleDict()

    @ abc.abstractmethod
    def build(self):
        pass

    def save(self, save_path):
        if self.net is not None:
            torch.save(self.net.state_dict(), save_path)
            print('[NETWORK] save: %s' % save_path)

    def load(self, load_path):
        if self.net is not None:
            self.net.load_state_dict(torch.load(load_path, map_location='cpu'))
            print('[NETWORK] load: %s' % load_path)


class folder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, feature_map):
        N,_,H,W = feature_map.size()
        feature_map = func.unfold(feature_map,kernel_size=(3, 1),padding=1)
        feature_map = feature_map.view(N,-1,H,W)
        return feature_map

def top_module(in_channels, attn_len):
    return folder()


class MMPENetwork(NetworkABC):
    def __init__(self, global_args, network_args, loss_func):
        super(MMPENetwork, self).__init__(global_args, network_args, loss_func)
        self.coord_range = (global_args['coord_h'], global_args['coord_w'])
        self.img_size = (global_args['img_h'], global_args['img_w'])
        self.n_classes = global_args['n_classes']
        self.n_joints = global_args['n_joints']
        self.top_k = global_args['top_k']

        self.dynamic_ch = network_args['dynamic_ch']
        self.fmap_ch = network_args['fmap_ch']
        self.std_factor = network_args['std_factor']
        self.max_batch_size = global_args['max_batch_size']
        self.out_split_idxes = [1, 2 * self.n_joints, 2 * self.n_joints]
        self.out_ch = sum(self.out_split_idxes)

        self.std_factor = torch.from_numpy(np.array(self.std_factor)).float().view(-1)
        if len(self.std_factor) == self.n_joints:
            self.std_factor = torch.stack([self.std_factor] * 2, dim=1).view(1, -1, 1)
        self.def_coord = None

    def build(self):
        backbone = get_backbone_dict()[self.network_args['backbone']](self.network_args)
        backbone.build()

        self.net['backbone'] = backbone
        self.net['detector'] = nn.Sequential(
            nn.Conv2d(self.fmap_ch, self.fmap_ch, 3, 1, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.fmap_ch, self.fmap_ch, 3, 1, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.fmap_ch, self.fmap_ch, 3, 1, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.fmap_ch, self.fmap_ch, 3, 1, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.fmap_ch, self.fmap_ch, 3, 1, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.fmap_ch, self.fmap_ch, 3, 1, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.fmap_ch, self.fmap_ch, 3, 1, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.fmap_ch, self.out_ch, 3, 1, 1, bias=True))

        output_sizes = list()
        for fmap2img_ratio in backbone.get_fmap2img_ratios():
            output_h = math.ceil(self.img_size[0] * fmap2img_ratio)
            output_w = math.ceil(self.img_size[1] * fmap2img_ratio)
            output_sizes.append((output_h, output_w))

        def_coord, self.coords_map = net_util.create_def_coord(self.max_batch_size, output_sizes, self.coord_range)
        self.def_coord = torch.cat([def_coord[:, :2]] * self.n_joints, dim=1)

        self.output_scale, from_idx = self.def_coord.clone(), 0
        for i, output_size in enumerate(output_sizes):
            to_idx = from_idx + output_size[0] * output_size[1]
            self.output_scale[:, :, from_idx:to_idx] = 0.0625 * (2 ** i)
            from_idx = to_idx

    def __sycn_batch_and_device__(self, batch_size, device_idx):
        if self.def_coord.device.index != device_idx:
            self.def_coord = self.def_coord.cuda(device_idx)
            self.std_factor = self.std_factor.cuda(device_idx)
            self.output_scale = self.output_scale.cuda(device_idx)
            for i, c_map in enumerate(self.coords_map):
                self.coords_map[i] = c_map.cuda(device_idx)
        def_coord = self.def_coord[:batch_size]
        output_scale = self.output_scale[:batch_size]
        return def_coord, output_scale

    def get_subnetworks_params(self, attns, num_bases, channels):
        assert attns.dim() == 2
        n_inst = attns.size(0)

        w0, b0, w1, b1, w2, b2 = torch.split_with_sizes(attns, [
            num_bases * channels, channels,
            channels * channels, channels,
            channels * 69, 69
        ], dim=1)

        # out_channels x in_channels x 1
        w0 = w0.reshape(n_inst * channels, num_bases, 1, 1)
        b0 = b0.reshape(n_inst * channels)
        w1 = w1.reshape(n_inst * channels, channels, 1, 1)
        b1 = b1.reshape(n_inst * channels)
        w2 = w2.reshape(n_inst * 69, channels, 1, 1)
        b2 = b2.reshape(n_inst * 69)

        return [w0, w1, w2], [b0, b1, b2]

    def subnetworks_forward(self, inputs, weights, biases, n_subnets):
        '''
        :param inputs: a list of inputs
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert inputs.dim() == 4
        n_layer = len(weights)
        x = inputs
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = func.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=n_subnets
            )
            if i < n_layer - 1:
                x = func.silu(x)
        return x

    def forward(self, image, gt_dict=None, loss=False):
        # import time

        # t1 = time.time()
        batch_size = image.shape[0]
        def_coord, output_scale = self.__sycn_batch_and_device__(batch_size, image.device.index)

        # t2 = time.time()
        num_level = 5
        fmaps = self.net['backbone'].forward(image, num_level)
        out_tensor = torch.cat([
            self.net['detector'].forward(fmap).view(batch_size, self.out_ch, -1) for fmap in fmaps
        ], dim=2)
        o1, o2, o3 = torch.split(out_tensor, self.out_split_idxes, dim=1)

        fg = torch.sigmoid(o1)  # (32, 1, 2134)

        pi = fg / torch.sum(fg, dim=2, keepdim=True)

        mu = o2 * output_scale + def_coord

        mu_x, mu_y = mu[:, 0::2], mu[:, 1::2]

        w = torch.max(mu_x, dim=1)[0] - torch.min(mu_x, dim=1)[0]
        h = torch.max(mu_y, dim=1)[0] - torch.min(mu_y, dim=1)[0]
        scale = torch.clamp_min(torch.stack([w, h] * self.n_joints, dim=1), lib_util.epsilon)
        sig = torch.max(func.softplus(o3), scale.detach() * self.std_factor)

        if (loss is False) or (gt_dict is None):
            out_dict = {'pi': pi, 'mu': mu, 'sig': sig, 'prob': fg}

            return out_dict

        else:
            out_dict = {'pi': pi, 'mu': mu, 'sig': sig, 'prob': fg}
            loss_dict, value_dict = self.loss_func.forward(out_dict, gt_dict)

            return loss_dict, value_dict