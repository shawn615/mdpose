import torch
from . import util as lib_util


def pose_nms(joints_s, conf_s, thresh, oks_scale=10.0, n_joints=17):
    # https://github.com/facebookresearch/posewarper/blob/master/lib/nms/nms.py
    # greedily select joints with high confidence and overlap with current maximum <= thresh
    # joints_s: (#joints, 17, 2)
    # conf_s:   (#joints)
    # scale_s:  (#joints)

    order = conf_s.argsort(descending=True)
    keep_idx_s = list()
    while order.shape[0] > 0:
        i = order[0]
        keep_idx_s.append(i)

        sim_pair_s = lib_util.calc_oks_torch(
            joints_s[i:i + 1], joints_s[order[1:]],
            scale_factor=oks_scale, n_joints=n_joints)
        # idx_s = torch.nonzero(sim_pair_s[0] <= thresh).view(-1) # too many pred keypoints, lower nms threshold?
        idx_s = torch.nonzero(sim_pair_s[0] <= thresh).reshape(-1)
        order = order[idx_s + 1]
    return keep_idx_s


# def oks_nms(kpts_db, thresh, sigmas=None, in_vis_thre=None):
#     """
#     greedily select boxes with high confidence and overlap with current maximum <= thresh
#     rule out overlap >= thresh, overlap = oks
#     :param kpts_db
#     :param thresh: retain overlap < thresh
#     :return: indexes to keep
#     """
#     if len(kpts_db) == 0:
#         return []
#
#     scores = np.array([kpts_db[i]['score'] for i in range(len(kpts_db))])
#     kpts = np.array([kpts_db[i]['keypoints'].flatten() for i in range(len(kpts_db))])
#     areas = np.array([kpts_db[i]['area'] for i in range(len(kpts_db))])
#
#     order = scores.argsort()[::-1]
#
#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#
#         oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]], sigmas, in_vis_thre)
#
#         inds = np.where(oks_ovr <= thresh)[0]
#         order = order[inds + 1]
#
#     return keep
