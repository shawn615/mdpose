import os
import math
import torch
import numpy as np
from torch.nn import functional as func

epsilon = 1e-12


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def cvt_torch2numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        if tensor.is_cuda:
            tensor = tensor.detach().cpu().numpy()
        else:
            tensor = tensor.detach().numpy()
    elif isinstance(tensor, list) or isinstance(tensor, tuple):
        for i in range(len(tensor)):
            tensor[i] = cvt_torch2numpy(tensor[i])
    elif isinstance(tensor, dict):
        for key in tensor.keys():
            tensor[key] = cvt_torch2numpy(tensor[key])
    return tensor


def cvt_int2onehot(integer, onehot_size):
    if len(integer.shape) > 2:
        integer = integer.squeeze(dim=2)
    onehot = torch.zeros(integer.shape + (onehot_size,)).float().cuda()
    dim = len(onehot.shape) - 1
    onehot.scatter_(dim, integer.unsqueeze(dim=dim), 1)
    return onehot


def clip_joints(joints, size):
    joints[:, :, 0::2] = torch.clamp(joints[:, :, 0::2], min=0, max=size[1] - 1)
    joints[:, :, 1::2] = torch.clamp(joints[:, :, 1::2], min=0, max=size[0] - 1)
    return joints


def clip_boxes_s(boxes_s, size, numpy=False):
    if numpy:
        boxes_s[:, [0, 2]] = np.clip(boxes_s[:, [0, 2]], a_min=0, a_max=size[1] - 1)
        boxes_s[:, [1, 3]] = np.clip(boxes_s[:, [1, 3]], a_min=0, a_max=size[0] - 1)
    else:
        boxes_s[:, [0, 2]] = torch.clamp(boxes_s[:, [0, 2]], min=0, max=size[1] - 1)
        boxes_s[:, [1, 3]] = torch.clamp(boxes_s[:, [1, 3]], min=0, max=size[0] - 1)
    return boxes_s


def sort_boxes_s(boxes_s, confs_s, labels_s=None):
    sorted_confs_s, sorted_idxs = torch.sort(confs_s, dim=0, descending=True)
    if len(sorted_idxs.shape) == 2:
        sorted_idxs = torch.squeeze(sorted_idxs, dim=1)
    sorted_boxes_s = boxes_s[sorted_idxs]

    if labels_s is None:
        return sorted_boxes_s, sorted_confs_s
    else:
        sorted_labels_s = labels_s[sorted_idxs]
        return sorted_boxes_s, sorted_confs_s, sorted_labels_s


def calc_oks_torch(joints_a_s, joints_b_s, visibles_b_s=None, scale_factor=10.0, n_joints=16):
    # joints_a_s:   (#joints_a, 17, 2)
    # joints_b_s:   (#joints_b, 17, 2)
    # visibles_b_s: (#joints_b, 17)
    # boxes_b_s:    (#joints_b, 4), #joints_b == #boxes_b
    # print(joints_a_s.shape, joints_b_s.shape, visibles_b_s.shape)

    joints_a_s = joints_a_s.unsqueeze(1)
    joints_b_s = joints_b_s.unsqueeze(0)
    # joints_a_s:   (#joints_a, 1, 17, 2)
    # joints_b_s:   (1, #joints_b, 17, 2)

    d_sqr = torch.sum((joints_a_s - joints_b_s) ** 2, dim=3).float()
    if n_joints == 16:
        k = np.array([.89, .87, 1.07, .89, .87, 1.07, 1.07, 1.07,
                      .26, .26, .62, .72, .79, .62, .72, .79]) / 5.0
    else:
        k = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72,
                      .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 5.0
    k_sqr = torch.from_numpy(k).float().cuda().view(1, 1, n_joints) ** 2

    s_sqr = get_joints_scale(joints_a_s[0]) * scale_factor
    s_sqr = torch.stack([s_sqr.unsqueeze(0)] * n_joints, dim=2).float()
    # d_sqr:        (#joints_a, #joints_b,  17)
    # k_sqr:        (1,         1,          17)
    # s_sqr:        (1,         #joints_b,  17)

    if visibles_b_s is None:
        visibles_b_s, visibles_sum_b_s = 1.0, n_joints
        # visibles_b_s, visibles_sum_b_s = 1.0, 17.0
    else:
        visibles_b_s = torch.stack([visibles_b_s] * joints_a_s.shape[0], dim=0)
        visibles_sum_b_s = torch.sum(visibles_b_s, dim=2)
    oks_joints_s = torch.exp(-d_sqr / (2 * k_sqr * (s_sqr + epsilon))) * visibles_b_s
    oks_s = torch.sum(oks_joints_s, dim=2) / visibles_sum_b_s
    # oks_joints_s: (#joints_a, #joints_b, 17)
    # visibles_b_s: (#joints_a, #joints_b, 17)
    # oks_s:        (#joints_a, #joints_b)
    return oks_s, oks_joints_s


def calc_jaccard_torch(boxes_a, boxes_b):
    # from https://github.com/luuuyi/RefineDet.PyTorch

    def intersect_torch(boxes_a, boxes_b):
        """ We resize both tensors to [A,B,2] without new malloc:
        [A,2] -> [A,1,2] -> [A,B,2]
        [B,2] -> [1,B,2] -> [A,B,2]
        Then we compute the area of intersect between box_a and box_b.
        Args:
          boxes_a: (tensor) bounding boxes, Shape: [A,4]. 여기서 A는 truths의 bounding box
          boxes_b: (tensor) bounding boxes, Shape: [B,4]. 여기서 B는 anchor들의 bounding box
        Return:
          (tensor) intersection area, Shape: [A,B].
        """
        A = boxes_a.size(0)
        B = boxes_b.size(0)
        max_xy = torch.min(boxes_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                           boxes_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(boxes_a[:, :2].unsqueeze(1).expand(A, B, 2),
                           boxes_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]

    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from anchorbox layers, Shape: [num_anchors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect_torch(boxes_a, boxes_b)
    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]) *
              (boxes_a[:, 3] - boxes_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((boxes_b[:, 2] - boxes_b[:, 0]) *
              (boxes_b[:, 3] - boxes_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def calc_jaccard_numpy(boxes_a, box_b):
    # From: https://github.com/amdegroot/ssd.pytorch
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        boxes_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    def intersect_numpy(_boxes_a, _boxes_b):
        max_xy = np.minimum(_boxes_a[:, 2:4], _boxes_b[2:4])
        min_xy = np.maximum(_boxes_a[:, 0:2], _boxes_b[0:2])
        _inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
        return _inter[:, 0] * _inter[:, 1]

    inter = intersect_numpy(boxes_a, box_b)
    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]) *
              (boxes_a[:, 3] - boxes_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def multivariate_gaussian_pdf(x, mu, sig):
    # make |mu|=K copies of y, subtract mu, divide by sigma
    # -0.5 *
    x_minus_mu = (x - mu).permute(3, 0, 1, 2).reshape(-1, 1, mu.shape[2])
    sig = sig.squeeze().permute(2, 0, 1).repeat(2, 1, 1)
    result = torch.bmm(x_minus_mu, sig)
    result = torch.bmm(result, x_minus_mu.permute(0, 2, 1))
    result = torch.exp(-0.5 * result).view(1, x.shape[1], 1, mu.shape[-1])
    trace = torch.exp(sig.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).view(1, x.shape[1], 1, mu.shape[-1])) ** 2
    trace = math.sqrt((2.0 * math.pi) ** 4 * trace)
    result = result / trace

    return result


def lognormal_pdf(x, mu, sig):
    # make |mu|=K copies of y, subtract mu, divide by sigma
    result = (torch.log(x) - mu) / sig
    result = -0.5 * (result * result)
    result = torch.exp(result) / (x * sig * math.sqrt(2.0 * math.pi))
    return result


def gaussian_pdf(x, mu, sig):
    # make |mu|=K copies of y, subtract mu, divide by sigma
    result = (x - mu) / sig
    result = -0.5 * (result * result)
    result = torch.exp(result) / (sig * math.sqrt(2.0 * math.pi))
    return result


def cauchy_pdf(x, loc, sc):
    dist = ((x - loc) / sc) ** 2
    result = 1 / (math.pi * sc * (dist + 1))
    return result


def laplace_pdf(x, loc, sc):
    dist = -(abs(x - loc) / sc)
    result = torch.exp(dist) / (2 * sc)
    # result = torch.exp(dist) / (weight * sc)
    # result = torch.exp(dist) / (3 * sc)
    return result


def asymmetric_laplace_pdf(x, loc, sc, asym):
    asym2 = torch.where(x < loc, 1 / asym, asym)
    dist = -(abs(x - loc) * sc * asym2)
    result = torch.exp(dist) * sc / (asym + 1 / asym)
    # result = torch.exp(dist) / (weight * sc)
    # result = torch.exp(dist) / (3 * sc)
    return result


def logistic_pdf(x, loc, sc):
    z = torch.exp(-(x - loc) / sc)
    result = z / (sc * ((1 + z) ** 2))
    return result


def gumble_pdf(x, loc, sc):
    z = (x - loc) / sc
    result = torch.exp(-(z + torch.exp(-z))) / sc
    return result


def moc_pdf(mu, sig, pi, points, sum_gauss=True, scale_factor=1.0, points_vis=None):
    mu, sig, pi = mu.unsqueeze(dim=1), sig.unsqueeze(dim=1), pi.unsqueeze(dim=1)
    points = points.unsqueeze(dim=3)
    # mu, sig shape: (batch, 1, 4, n_gauss), xywh
    # pi shape: (batch, 1, 1, n_gauss)
    # points shape: (batch, n_points, 4, 1), xywh

    result = cauchy_pdf(points, mu, sig) * scale_factor
    # result = gaussian_pdf(points, mu, sig) * scale_factor
    if points_vis is not None:
        _points_vis = torch.stack([points_vis] * result.shape[3], dim=3)
        result = torch.where(
            _points_vis == 0, torch.ones(result.shape).cuda(), result)
        # points_vis = -(points_vis.unsqueeze(dim=3) - 1)
        # result *= points_vis
        # print('gausses:', result.shape)
        # print('vis:', points_vis.shape)
    result = torch.prod(result + epsilon, dim=2, keepdim=True)
    result = pi * result
    if sum_gauss:
        result = torch.sum(result, dim=3)
    return result


def mm_pdf(mu, sig, pi, points, sum_comp=True):
    mu, sig, pi = mu.unsqueeze(dim=1), sig.unsqueeze(dim=1), pi.unsqueeze(dim=1)
    points = points.unsqueeze(dim=3)
    # mu, sig shape:    (batch, 1, 4, n_gauss), xywh
    # pi shape:         (batch, 1, 1, n_gauss)
    # points shape:     (batch, n_points, 4, 1), xywh

    result = gaussian_pdf(points, mu, sig)
    # result.shape:     (batch, n_points, 4, n_gauss)
    result = torch.prod(result + epsilon, dim=2, keepdim=True)
    # result.shape:     (batch, n_points, 1, n_gauss)
    if pi is not None:
        result = pi * result
        # result.shape: (batch, n_points, 1, n_gauss)
    if sum_comp:
        result = torch.sum(result, dim=3)
        # result.shape: (batch, n_points, 1)
    return result


def mm_pdf_s(mu_s, sig_s, pi_s, points_s, points_vis_s=None, sum_comp=True):
    try:
        ones = mm_pdf_s.ones[:points_s.shape[0]]
    except:
        mm_pdf_s.ones = torch.ones(300, 4, mu_s.shape[1]).cuda()
        ones = mm_pdf_s.ones[:points_s.shape[0]]

    mu, sig, pi = mu_s.unsqueeze(dim=0), sig_s.unsqueeze(dim=0), pi_s.unsqueeze(dim=0)
    points_s, points_vis_s = points_s.unsqueeze(dim=2), points_vis_s.unsqueeze(2)
    # mu, sig shape:    (1, 4, n_gauss)
    # pi shape:         (1, 1, n_gauss)
    # points shape:     (n_points, 4, 1)
    # points_vis shape: (n_points, 4, 1)

    result = cauchy_pdf(points_s, mu_s, sig_s)
    # result.shape:     (n_points, 4, n_gauss)
    if points_vis_s is not None:
        _points_vis = torch.cat([points_vis_s] * result.shape[2], dim=2)
        result = torch.where(_points_vis == 0, ones, result)
    result = torch.prod(result, dim=1, keepdim=True)
    # result.shape:     (n_points, 1, n_gauss)
    if pi is not None:
        result = (pi * result)[:, 0]
        # result.shape: (n_points, n_gauss)
    if sum_comp:
        result = torch.sum(result, dim=1)
        # result.shape: (n_points)
    return result


def category_pmf(clsprobs, onehot_labels):
    clsprobs = clsprobs.unsqueeze(dim=1)
    onehot_labels = onehot_labels.unsqueeze(dim=3)
    cat_probs = torch.prod(clsprobs ** onehot_labels, dim=2, keepdim=True)
    return cat_probs


def bernoulli_pmf(cls_probs, labels):
    # cls_probs:    (batch, #joints, #gauss)
    # labels:       (batch, #sample, #joints)
    cls_probs = cls_probs.unsqueeze(dim=1)
    # labels = labels.unsqueeze(dim=3)
    ber_probs = (cls_probs ** labels) * ((1 - cls_probs) ** (1 - labels))
    return ber_probs


def sample_coords_from_mog(mu, sig, pi, n_samples):
    # mu.shape: (batch_size, 4, #comp)
    # sig.shape: (batch_size, 4, #comp)
    # pi.shape: (batch_size, 1, #comp)
    # print(mu.shape, sig.shape, pi.shape, n_samples)
    _mu = cvt_torch2numpy(mu)
    _sig = cvt_torch2numpy(sig)
    _pi = cvt_torch2numpy(pi).reshape((pi.shape[0], -1))
    n_gauss = _pi.shape[1]

    gen_coords = list()
    for mu_s, sig_s, pi_s in zip(_mu, _sig, _pi):
        gauss_nums = np.random.choice(n_gauss, size=n_samples, p=pi_s)
        normal_noises = np.random.randn(n_samples * mu.shape[1]).reshape((mu.shape[1], n_samples))
        gen_coords_s = mu_s[:, gauss_nums] + normal_noises * sig_s[:, gauss_nums]
        gen_coords.append(np.expand_dims(gen_coords_s, axis=0))

    gen_coords = np.concatenate(gen_coords, axis=0)
    gen_coords = torch.from_numpy(gen_coords).float().cuda()
    gen_coords = gen_coords.transpose(1, 2)
    return gen_coords


def sample_coords_from_cat(value, prob, n_samples):
    np_value = cvt_torch2numpy(value)
    np_prob = cvt_torch2numpy(prob)

    gen_coords = list()
    for np_value_s, np_prob_s in zip(np_value, np_prob):
        value_idxes_s = np.random.choice(np_prob_s.shape[1], size=n_samples, p=np_prob_s[0])
        gen_coords.append(np_value_s[:, value_idxes_s])

    gen_coords = np.stack(gen_coords, axis=0)
    gen_coords = torch.from_numpy(gen_coords).float().cuda()
    gen_coords = gen_coords.transpose(1, 2)
    return gen_coords


def create_coord_map(coord_map_size, coord_range):
    # gauss_w: 4 --> ((0, 1, 2, 3), ...)
    x_map = np.array(list(range(coord_map_size[1])) * coord_map_size[0]).astype(np.float32)
    y_map = np.array(list(range(coord_map_size[0])) * coord_map_size[1]).astype(np.float32)

    x_map = x_map.reshape((1, 1, coord_map_size[0], coord_map_size[1]))
    y_map = y_map.reshape((1, 1, coord_map_size[1], coord_map_size[0]))
    y_map = y_map.transpose((0, 1, 3, 2))

    # coord_w: 100 --> unit_intv_w: 25
    unit_intv_w = coord_range[1] / coord_map_size[1]    # 0.25, 0.5, 1.0, 2.0, 4.0
    unit_intv_h = coord_range[0] / coord_map_size[0]    # 0.25, 0.5, 1.0, 2.0, 4.0

    # (0, 1, 2, 3) * 25 + 12.5 == (12.5, 37.5, 62.5, 87.5)
    x_map = x_map * unit_intv_w + unit_intv_w / 2   # 0.25: 0.125, 0.375, 0.625, 0.875, ...
    y_map = y_map * unit_intv_h + unit_intv_h / 2
    return np.concatenate((x_map, y_map), axis=1)


def get_joints_scale(joints_s):
    scales_s = torch.zeros(joints_s.shape[0]).cuda()
    # joints_s:     (#joints, 17, 2)
    # visibles_s:   (#joints, 17)
    # scale_s:      (#joints)

    for i, joint_s in enumerate(joints_s):
        w = torch.max(joint_s[:, 0]) - torch.min(joint_s[:, 0])
        h = torch.max(joint_s[:, 1]) - torch.min(joint_s[:, 1])
        scales_s[i] = w * h
    return scales_s
