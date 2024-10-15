import abc
import numpy as np
import torch
import torch.nn.functional as func
import torch.nn as nn
import torch.distributions as distributions
from . import util as lib_util
from . import loss_func_util as loss_util


def get_loss_func_dict():
    return {
        'mmpe': MMPELossFunction,
        # 'mmpe_simple': MMPESimpleLossFunc,
        'mmpe_crop': MMPECropLossFunction,

        'pose_ae': PoseAELossFunction,
    }


class LossFunctionABC(abc.ABC):
    def __init__(self, global_args, loss_args):
        self.global_args = global_args
        self.loss_args = loss_args

    @ abc.abstractmethod
    def forward(self, *x):
        pass


class MMPELossFunction(LossFunctionABC):
    def __init__(self, global_args, loss_func_args):
        super(MMPELossFunction, self).__init__(global_args, loss_func_args)
        self.is_coco = global_args['is_coco']
        self.n_joints = global_args['n_joints']
        self.n_group_joints = loss_func_args['n_group_joints']
        self.n_train_groups = loss_func_args['n_train_groups']

        joint_weight = loss_func_args['joint_weight']
        joint_weight = torch.from_numpy(np.array(joint_weight)).float().view(-1)
        self.joint_weight = torch.stack([joint_weight] * 2, dim=1).view(-1)

        # self.mse = torch.nn.MSELoss()

        self.joint_sampling = loss_func_args['joint_sampling']
        self.n_samples = loss_func_args['n_samples']
        self.oks_scale = loss_func_args['oks_scale']
        self.sim_thresh = loss_func_args['sim_thresh']
        self.with_box = loss_func_args['with_box']

        self.lw_dict = loss_func_args['lw_dict']
        assert 'joints_nll' in self.lw_dict.keys()
        assert 'prob_nll' in self.lw_dict.keys()

        self.ones = None
        self.boxes_vis_s = None
        self.bg_label_s = None
        self.fg_label_s = None

    def shuffle_joints(self, mu_s, sig_s, joints_s, joints_vis_s, random=True):
        # n_train_joints = self.n_group_joints * self.n_train_groups
        n_train_joints = self.n_joints  # due to coco joint num

        ### grouping adjacent joints or randomly
        if not random:
            # (leye, reye), (lsho, rsho), (lelb, relb), (lwri, rwri), (lhip, rhip), (lkne, rkne), (lank, rank), (lear, rear), (nose,)
            # joints_indices_s = torch.tensor([1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 3, 4, 0])
            joints_indices_s = torch.tensor([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 1, 2, 3, 4])
            joints_indices_s = torch.stack([joints_indices_s * 2, joints_indices_s * 2 + 1], dim=1).view(-1)
        else:
            joints_indices_s = torch.randperm(self.n_joints)[:n_train_joints]
            joints_indices_s = torch.stack([joints_indices_s * 2, joints_indices_s * 2 + 1], dim=1).view(-1)
        ### grouping adjacent joints or randomly
        if self.with_box:
            joints_indices_s = torch.cat([torch.arange(4), joints_indices_s], dim=0)

        # print(self.n_joints, n_train_joints, joints_s.shape, joints_indices_s.shape)
        # additional output for keypoint_vis
        shuffled_mu_s, shuffled_sig_s = mu_s[:, joints_indices_s], sig_s[:, joints_indices_s]
        shuffled_joints_s = joints_s[:, :, joints_indices_s]
        if joints_vis_s is None:
            return shuffled_mu_s, shuffled_sig_s, shuffled_joints_s, joints_indices_s
        else:
            shuffled_joints_vis_s = joints_vis_s[:, :, joints_indices_s]
            return shuffled_mu_s, shuffled_sig_s, shuffled_joints_s, shuffled_joints_vis_s, joints_indices_s

    def calc_group_joints_nll_s(self, pi_s, mu_s, sig_s, boxes_s, joints_s, joints_vis_s):
        # pi_s, mu_s, sig_s = pi_s[:, :, :700], mu_s[:, :, :700], sig_s[:, :, :700]
        # pi_s:         (1, 1, #gauss)
        # mu_s, sig_s:  (1, #joints * 2, #gauss)
        # joints_s:     (1, #people, #joints * 2)
        # joints_vis_s: (1, #people, #joints * 2)
        # vis_s:        (1, #joints * 2, #gauss)

        if self.joint_weight.device.index != mu_s.device.index:
            self.joint_weight = self.joint_weight.cuda(mu_s.device.index)
        n_gauss, n_people = pi_s.shape[2], joints_s.shape[1]

        if self.with_box:
            if self.boxes_vis_s is None:
                self.boxes_vis_s = torch.ones(1, 300, 4).cuda()
            # print(boxes_s.shape, joints_s.shape, self.boxes_vis_s.shape, joints_vis_s.shape)
            _joints_s = torch.cat([boxes_s, joints_s], dim=2)
            _joints_vis_s = torch.cat([self.boxes_vis_s[:, :joints_vis_s.shape[1]], joints_vis_s], dim=2)
        else:
            _joints_s, _joints_vis_s = joints_s, joints_vis_s

        # self.n_joints, self.n_train_groups, self.with_box = self.n_joints + 2, self.n_train_groups + 1, False
        shuffled_mu_s, shuffled_sig_s, shuffled_joints_s, shuffled_joints_vis_s, joints_indices_s = \
            self.shuffle_joints(mu_s, sig_s, _joints_s, _joints_vis_s, random=True)
        shuffled_joint_weight = self.joint_weight[joints_indices_s]

        train_pi_s = pi_s
        train_mu_s = shuffled_mu_s.unsqueeze(dim=1)
        train_sig_s = shuffled_sig_s.unsqueeze(dim=1)
        train_joints_s = shuffled_joints_s.unsqueeze(dim=3)
        train_joints_vis = torch.stack([shuffled_joints_vis_s] * n_gauss, dim=3)
        # mu_s, sig_s:  (1, 1, #joints * 2, #gauss)
        # pi_s:         (1, 1, #gauss)
        # joints_s:     (1, #people, #joints * 2, 1)
        # joints_vis_s: (1, #people, #joints * 2, #gauss)

        laplace_lh_s = lib_util.laplace_pdf(train_joints_s, train_mu_s, train_sig_s)

        if self.ones is None:
            self.ones = torch.ones((1, 100, laplace_lh_s.shape[2], laplace_lh_s.shape[3])).cuda()
        # print(train_joints_vis.shape, self.ones.shape, laplace_lh_s.shape)
        laplace_lh_s = torch.where(train_joints_vis == 0, self.ones[:, :laplace_lh_s.shape[1]], laplace_lh_s)  # for training with gt keypoint_vis

        if self.is_coco and (self.n_group_joints != 1 and self.n_group_joints != 17):
            temp_ones = torch.ones((laplace_lh_s.shape[0], laplace_lh_s.shape[1], 2, laplace_lh_s.shape[3])).cuda()
            laplace_lh_s = torch.cat([laplace_lh_s, temp_ones], dim=2)
        #############################################
        group_laplace_lh_s = laplace_lh_s.view(n_people, -1, self.n_group_joints * 2, n_gauss)

        # print('1:', group_laplace_lh_s.shape)
        ### ssh; due to coco joint num; exp 210702-164449
        if self.is_coco and (self.n_group_joints != 1 and self.n_group_joints != 17):
            # n_group_joints: 2
            group_laplace_lh_s_a = group_laplace_lh_s[:, :-2, :, :]
            group_laplace_lh_s_b = torch.cat([group_laplace_lh_s[:, -2, :, :].unsqueeze(1), group_laplace_lh_s[:, -1, :2, :].unsqueeze(1)], dim=2)
            group_multi_lh_s_a = torch.prod(group_laplace_lh_s_a + lib_util.epsilon, dim=2)
            group_multi_lh_s_b = torch.prod(group_laplace_lh_s_b + lib_util.epsilon, dim=2)
            group_multi_lh_s = torch.cat([group_multi_lh_s_a, group_multi_lh_s_b], dim=1)
        else:
            group_multi_lh_s = torch.prod(group_laplace_lh_s + lib_util.epsilon, dim=2)
        #############################################
        group_moc_lh_s = torch.sum(train_pi_s * group_multi_lh_s, dim=2)
        # gauss_lh_s:       (1, #people, #joints * 2, #gauss)
        # group_gauss_lh_s: (#people, #groups, #group_joints * 2, #gauss)
        # group_multi_lh_s: (#people, #groups, #gauss)
        # group_mog_lh_s:   (#people, #groups)

        group_joints_nll_s = -torch.log(group_moc_lh_s + lib_util.epsilon)

        if self.is_coco and (self.n_group_joints != 1 and self.n_group_joints != 17):
            temp_ones = torch.ones(2).cuda()
            shuffled_joint_weight = torch.cat([shuffled_joint_weight, temp_ones])
        #############################################
        group_joint_weight = shuffled_joint_weight.view(1, -1, self.n_group_joints * 2)

        if self.is_coco and (self.n_group_joints != 1 and self.n_group_joints != 17):
            # n_group_joints: 2
            group_joint_weight_a = group_joint_weight[:, :-2, :]
            group_joint_weight_b = torch.cat([group_joint_weight[:, -2, :].unsqueeze(1), group_joint_weight[:, -1, :2].unsqueeze(1)], dim=2)
            group_joint_weight_a = torch.prod(group_joint_weight_a, dim=2)
            group_joint_weight_b = torch.prod(group_joint_weight_b, dim=2)
            group_joint_weight = torch.cat([group_joint_weight_a, group_joint_weight_b], dim=1)
        else:
            group_joint_weight = torch.prod(group_joint_weight, dim=2)
        #############################################
        joints_nll_s = torch.mean(group_joints_nll_s * group_joint_weight, dim=1)
        # group_joints_nll_s:   (#people, #groups)
        # group_joint_weight:   (1, #groups)
        # joints_nll_s:         (#people)
        return joints_nll_s

    def forward(self, out_dict, gt_dict):
        boxes, joints, n_people = gt_dict['boxes'], gt_dict['joints'], gt_dict['n_people']
        mu, sig = out_dict['mu'], out_dict['sig']
        pi, prob = out_dict['pi'], out_dict['prob']

        batch_size = pi.shape[0]
        joints_nll, bbox_nll, prob_nll, max_sim, pos_ratio = list(), list(), list(), list(), list()
        for i in range(batch_size):
            mu_s, sig_s = mu[i:i + 1], sig[i:i + 1]
            pi_s, prob_s = pi[i:i + 1], prob[i:i + 1]
            boxes_s = boxes[i:i + 1, :n_people[i]]
            joints_s = joints[i:i + 1, :n_people[i], :, :2]
            joints_vis_s = joints[i:i + 1, :n_people[i], :, 2]

            # calc center_nll and joints_nll (localization) ------------------------------------------------------------
            _joints_s = joints_s.contiguous().view(1, n_people[i], self.n_joints * 2)
            _joints_vis_s = torch.stack([joints_vis_s] * 2, dim=3).view(1, n_people[i], self.n_joints * 2)
            center_nll_s = self.calc_group_joints_nll_s(pi_s, mu_s, sig_s, boxes_s, _joints_s, _joints_vis_s)
            if len(center_nll_s) > 0:
                joints_nll.append(center_nll_s)

        loss_dict, value_dict = dict(), dict()
        joints_nll = self.lw_dict['joints_nll'] * torch.cat(joints_nll, dim=0)
        loss_dict.update({'joints_nll': joints_nll})

        if len(prob_nll) > 0:
            prob_nll = self.lw_dict['prob_nll'] * torch.cat(prob_nll, dim=0)
            pos_ratio = torch.mean(torch.cat(pos_ratio, dim=0)).view(1)
            loss_dict.update({'prob_nll': prob_nll})
            value_dict.update({'pos_ratio': pos_ratio})

        return loss_dict, value_dict


class MMPECropLossFunction(MMPELossFunction):
    def forward(self, out_dict, gt_dict):
        gt_dict['center'], gt_dict['joints'], gt_dict['n_people'] = \
            gt_dict['crop_center'], gt_dict['crop_joints'], gt_dict['crop_n_people']
        return super(MMPECropLossFunction, self).forward(out_dict, gt_dict)


class PoseAELossFunction(LossFunctionABC):
    dist_dict = {
        'l1': func.l1_loss,
    }

    def __init__(self, global_args, loss_func_args):
        super(PoseAELossFunction, self).__init__(global_args, loss_func_args)
        self.n_joints = global_args['n_joints']
        self.dist_func = self.dist_dict[loss_func_args['dist_type']]

    def forward(self, out_dict, gt_dict):
        gt_crop_joints = gt_dict['crop_joints']
        gt_crop_joints_vis = gt_dict['crop_joints_vis']
        rc_crop_joints = out_dict['rc_crop_joints']
        gt_crop_joints_vis = torch.stack([gt_crop_joints_vis] * 2, dim=2).view(-1, self.n_joints * 2, 1, 1)

        recon_loss = self.dist_func(rc_crop_joints, gt_crop_joints, reduction='none')
        recon_loss *= gt_crop_joints_vis
        recon_loss = torch.sum(torch.sum(torch.sum(recon_loss, dim=1), dim=1), dim=1)
        return {'recon': recon_loss}, {}
