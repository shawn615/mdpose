import torch
from torchvision.ops.boxes import nms
from . import util as lib_util
from . import post_proc_util as post_util


def get_post_proc_dict():
    return {
        'mmpe': MMPEPostProc,
        'mmpe_box': MMPEBoxPostProc,
        'none': None,
    }


class MMPEPostProc(object):
    def __init__(self, global_args, post_proc_args):
        self.input_size = (global_args['img_h'], global_args['img_w'])
        self.coord_range = (global_args['coord_h'], global_args['coord_w'])
        self.n_classes = global_args['n_classes']
        self.n_joints = global_args['n_joints']

        self.pi_thresh = post_proc_args['pi_thresh']
        self.conf_thresh = post_proc_args['conf_thresh']

        self.oks_scale = post_proc_args['oks_scale']
        self.nms_thresh = post_proc_args['nms_thresh']

    def forward(self, out_dict):
        pi_s, prob_s = out_dict['pi'], out_dict['prob']
        mu_s, sig_s = out_dict['mu'], out_dict['sig']
        assert pi_s.shape[0] == 1

        joints_s = mu_s.transpose(1, 2).clone()
        joints_s[:, :, 0::2] = joints_s[:, :, 0::2] * (self.input_size[1] / self.coord_range[1])
        joints_s[:, :, 1::2] = joints_s[:, :, 1::2] * (self.input_size[0] / self.coord_range[0])
        joints_s = lib_util.clip_joints(joints_s, self.input_size)
        joints_s = joints_s.view(joints_s.shape[0], joints_s.shape[1], self.n_joints, 2)
        # joints_s = joints_s.view(joints_s.shape[0], joints_s.shape[1], self.n_joints, 2)[:, :, 1:]

        nor_pi_s = pi_s / torch.max(pi_s, dim=2, keepdim=True)[0]
        keep_idx_s = torch.nonzero(nor_pi_s[0, 0] > self.pi_thresh).view(-1)
        joints_s = joints_s[:, keep_idx_s]
        conf_s = prob_s[:, :, keep_idx_s]
        # print(joints_s.shape, conf_s.shape)

        keep_idx_s = torch.nonzero(conf_s[0] > self.conf_thresh).view(-1)
        joints_s = joints_s[:, keep_idx_s]
        conf_s = conf_s[:, :, keep_idx_s]
        # print(joints_s.shape, conf_s.shape)

        # l, t = torch.min(joints_s[:, :, :, 0], dim=2)[0], torch.min(joints_s[:, :, :, 1], dim=2)[0]
        # r, b = torch.max(joints_s[:, :, :, 0], dim=2)[0], torch.max(joints_s[:, :, :, 1], dim=2)[0]
        # boxes_s = torch.stack([l, t, r, b], dim=2)
        # print(joints_s.shape, boxes_s.shape)

        # keep_idx_s = nms(boxes_s[0], conf_s[0], self.nms_thresh)
        keep_idx_s = post_util.pose_nms(
            joints_s[0], conf_s[0], self.nms_thresh,
            oks_scale=self.oks_scale, n_joints=self.n_joints)
        joints_s = joints_s[:, keep_idx_s]
        conf_s = conf_s[:, keep_idx_s]
        # print(joints_s.shape, conf_s.shape)
        return {'joints_s': joints_s, 'conf_s': conf_s}, {}


class MMPEBoxPostProc(object):
    def __init__(self, global_args, post_proc_args):
        self.input_size = (global_args['img_h'], global_args['img_w'])
        self.coord_range = (global_args['coord_h'], global_args['coord_w'])
        self.n_classes = global_args['n_classes']
        self.n_joints = global_args['n_joints']

        self.pi_thresh = post_proc_args['pi_thresh']
        self.conf_thresh = post_proc_args['conf_thresh']
        self.nms_thresh = post_proc_args['nms_thresh']
        self.norm_pose = post_proc_args['norm_pose']
        self.with_box = post_proc_args['with_box']
        self.vis_thresh = post_proc_args['vis_thresh']
        self.oks_scale = post_proc_args['oks_scale']

    def forward(self, out_dict):
        def get_zero_result(n_joints):
            dummy_box_s = torch.zeros(1, 1, n_joints, 2)
            dummy_conf_s = torch.zeros(1, 1)
            return {'joints_s': dummy_box_s, 'conf_s': dummy_conf_s}, {}

        pi_s, prob_s = out_dict['pi'], out_dict['prob']
        mu_s, sig_s = out_dict['mu'], out_dict['sig']
        assert pi_s.shape[0] == 1
        # mu_s = mu_s*sig_s + mu_s    ### ssh; apply RLE
        joints_s = mu_s.transpose(1, 2).clone()
        joints_s = joints_s.view(joints_s.shape[0], joints_s.shape[1], -1, 2)

        if self.norm_pose:
            # joints_s: (1, #comp, #joints + 2, 2)
            center = (joints_s[:, :, 1:2] + joints_s[:, :, 0:1]) * 0.5
            scale = (joints_s[:, :, 1:2] - joints_s[:, :, 0:1])
            joints_s[:, :, 2:] = (joints_s[:, :, 2:] - center) * (scale * 0.5) + center

        joints_s[:, :, :, 0] = joints_s[:, :, :, 0] * (self.input_size[1] / self.coord_range[1])
        joints_s[:, :, :, 1] = joints_s[:, :, :, 1] * (self.input_size[0] / self.coord_range[0])
        # joints_s[:, :, :, 0] = joints_s[:, :, :, 0] * self.input_size[1]
        # joints_s[:, :, :, 1] = joints_s[:, :, :, 1] * self.input_size[0]
        joints_s[:, :, :, 0] = torch.clamp(joints_s[:, :, :, 0], min=0, max=self.input_size[1] - 1)
        joints_s[:, :, :, 1] = torch.clamp(joints_s[:, :, :, 1], min=0, max=self.input_size[0] - 1)
        conf_s = prob_s[:, 0] if prob_s.shape[1] == 1 else prob_s[:, 1]

        if self.pi_thresh is not None:
            nor_pi_s = pi_s / torch.max(pi_s, dim=2, keepdim=True)[0]
            keep_idx_s = torch.nonzero(nor_pi_s[0, 0] > self.pi_thresh).view(-1)
            if len(keep_idx_s) == 0:
                return get_zero_result(self.n_joints)
            joints_s = joints_s[:, keep_idx_s]
            conf_s = conf_s[:, keep_idx_s]

        if self.conf_thresh is not None:
            keep_idx_s = torch.nonzero(conf_s[0] > self.conf_thresh).view(-1)
            if len(keep_idx_s) == 0:
                return get_zero_result(self.n_joints)
            joints_s = joints_s[:, keep_idx_s]
            conf_s = conf_s[:, keep_idx_s]

        # joints_s: (1, #comp, #joints, 2)
        # conf_s:   (1, #comp)
        if self.with_box:
            _boxes_s = joints_s[:, :, :2]
            boxes_s = _boxes_s.view(1, joints_s.shape[1], 4)
            box_conf_s = conf_s
            joints_s = joints_s[:, :, 2:]
            # conf_s = conf_s[:, ].transpose(1, 2)
        else:
            l, t = torch.min(joints_s[:, :, :, 0], dim=2)[0], torch.min(joints_s[:, :, :, 1], dim=2)[0]
            r, b = torch.max(joints_s[:, :, :, 0], dim=2)[0], torch.max(joints_s[:, :, :, 1], dim=2)[0]
            boxes_s = torch.stack([l, t, r, b], dim=2)
            box_conf_s = conf_s

        keep_idx_s = nms(boxes_s[0], box_conf_s[0], self.nms_thresh)
        # keep_idx_s = post_util.pose_nms(
        #     joints_s[0], conf_s[0], self.nms_thresh,
        #     oks_scale=self.oks_scale, n_joints=self.n_joints)
        # a = self.nms_thresh
        # box_shape = boxes_s.shape
        # box_conf_shape = box_conf_s.shape
        # joints_shape = joints_s.shape
        # conf_max = conf_s.max()
        joints_s = joints_s[:, keep_idx_s]
        conf_s = conf_s[:, keep_idx_s]
        return {'joints_s': joints_s, 'conf_s': conf_s}, {}
