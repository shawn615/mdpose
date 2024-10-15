import torch
from . import util as lib_util
import torch.nn as nn


class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()

        self.prior = prior
        self.register_buffer('mask', mask)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(mask))])

    def _init(self):
        for m in self.t:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)
        for m in self.s:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)

    def forward_p(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def backward_p(self, x):
        # n_components = x.shape[3]
        x = x.permute(3, 2, 1, 0).reshape(-1, 2)
        # log_det_J, z = x.new_zeros(x.shape[0]), x
        log_det_J, z = x.new_zeros(x.shape), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            # log_det_J -= s.sum(dim=1)
            log_det_J -= s
        return z, log_det_J

    def log_prob(self, x):
        DEVICE = x.device
        if self.prior.loc.device != DEVICE:
            self.prior.loc = self.prior.loc.to(DEVICE)
            self.prior.scale_tril = self.prior.scale_tril.to(DEVICE)
            self.prior._unbroadcasted_scale_tril = self.prior._unbroadcasted_scale_tril.to(DEVICE)
            self.prior.covariance_matrix = self.prior.covariance_matrix.to(DEVICE)
            self.prior.precision_matrix = self.prior.precision_matrix.to(DEVICE)

        z, logp = self.backward_p(x)
        # return self.prior.log_prob(z) + logp
        return lib_util.gaussian_pdf(0, z, 1) * torch.exp(logp)
        # return z * torch.exp(logp)

    def prob(self, x):
        return torch.exp(self.log_prob(x))

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        x = self.forward_p(z)
        return x

    def forward(self, x):
        return self.log_prob(x)


def calc_prob_nll_s(pi_s, mu_s, sig_s, prob_s, roi_s, roi_label_s, scale_factor=1.0):
    gauss_lh_s = lib_util.moc_pdf(
        mu_s, sig_s, pi_s, roi_s, sum_gauss=False,
        scale_factor=scale_factor, points_vis=None)[0, :, 0]
    cat_prob_s = lib_util.category_pmf(prob_s, roi_label_s.float())[0, :, 0]
    # print('-', mu_s.shape, sig_s.shape, pi_s.shape, roi_s.shape, gauss_lh_s.shape, cat_prob_s.shape)
    prob_lhs_s = torch.sum(gauss_lh_s * cat_prob_s, dim=1)
    prob_nll_s = -torch.log(prob_lhs_s + lib_util.epsilon)
    return prob_nll_s


# def calc_prob_nll_s(pi_s, mu_s, sig_s, prob_s, roi_s, roi_label_s, scale_factor=1.0):
#     gauss_lh_s = lib_util.moc_pdf(
#         mu_s, sig_s, pi_s, roi_s, sum_gauss=False,
#         scale_factor=scale_factor, points_vis=None)[0]
#     ber_prob_s = lib_util.bernoulli_pmf(prob_s, roi_label_s.float())[0]
#     prob_lhs_s = torch.sum(gauss_lh_s * ber_prob_s, dim=2)
#     prob_nll_s = torch.sum(-torch.log(prob_lhs_s + lib_util.epsilon), dim=1)
#     return prob_nll_s


def calc_center_nll_s(pi_s, center_mu_s, center_sig_s, center_s):
    center_lh_s = lib_util.moc_pdf(center_mu_s, center_sig_s, pi_s, center_s, sum_gauss=True)[0, :, 0]
    # print(center_lh_s.shape)
    # center_lh_s:  (#center)
    center_nll_s = -torch.log(center_lh_s + lib_util.epsilon)
    return center_nll_s


def calc_center_joints_nll_s(pi_s, center_mu_s, center_sig_s, joints_mu_s, joints_sig_s, center_s, joints_s,
                             pi_thresh=0.0001, n_joints=16):
    # joints_mu_s, joints_sig_s:    (1, 34, #gauss), (1, 34, #gauss)
    # joints_s:                     (1, #people, 17, 3)
    joints_mu_s = joints_mu_s.view(1, n_joints, 2, -1).transpose(1, 2)
    joints_sig_s = joints_sig_s.view(1, n_joints, 2, -1).transpose(1, 2)
    joints_s, visible_s = joints_s[:, :, :, :2], joints_s[:, :, :, 2]
    # center_mu_s, joints_mu_s:     (1, 2, #gauss), (1, 2, 17, #gauss)
    # center_sig_s, joints_sig_s:   (1, 2, #gauss), (1, 2, 17, #gauss)
    # center_s, joints_s:           (1, #people, 2), (1, #people, 17, 2)
    # visible_s:                    (1, #people, 17)

    br_center_mu_s = torch.stack([center_mu_s] * n_joints, dim=2)
    br_center_sig_s = torch.stack([center_sig_s] * n_joints, dim=2)
    br_center_s = torch.stack([center_s] * n_joints, dim=2)
    vectors_mu_s = torch.cat([br_center_mu_s, joints_mu_s], dim=1).transpose(2, 3)
    vectors_sig_s = torch.cat([br_center_sig_s, joints_sig_s], dim=1).transpose(2, 3)
    vectors_s = torch.cat([br_center_s, joints_s], dim=3).transpose(2, 3)
    # vectors_mu_s, vectors_sig_s:  (1, 4, #gauss, 17), (1, 4, #gauss, 17)
    # vectors_s:                    (1, #people, 4, 17)

    if pi_thresh is not None:
        nor_pi_s = pi_s / torch.max(pi_s)
        keep_idx_s = torch.nonzero(nor_pi_s[0, 0] >= pi_thresh).view(-1)
        if pi_s.shape[2] > len(keep_idx_s):
            vectors_mu_s = vectors_mu_s[:, :, keep_idx_s, :]
            vectors_sig_s = vectors_sig_s[:, :, keep_idx_s, :]
            pi_s = pi_s[:, :, keep_idx_s]

    uns_pi_s = pi_s.unsqueeze(3)
    joint_lh_s = lib_util.moc_pdf(vectors_mu_s, vectors_sig_s, uns_pi_s, vectors_s, sum_gauss=True)[0, :, 0]
    # joint_lh_s:                   (#people, 17)

    joint_lh_s, visible_s = joint_lh_s.view(-1), visible_s.view(-1)
    keep_idx_s = torch.nonzero(visible_s).view(-1)
    joint_lh_s = joint_lh_s[keep_idx_s]
    joint_nll_s = -torch.log(joint_lh_s + lib_util.epsilon)
    return joint_nll_s


def calc_center_prob_nll_s(pi_s, center_mu_s, center_sig_s, prob_s, roi_center_s, roi_label_s):
    gauss_lh_s = lib_util.mm_pdf(center_mu_s, center_sig_s, pi_s, roi_center_s, sum_comp=False)[0, :, 0]
    cat_prob_s = lib_util.category_pmf(prob_s, roi_label_s.float())[0, :, 0]
    prob_lhs_s = torch.sum(gauss_lh_s * cat_prob_s, dim=1)
    prob_nll_s = -torch.log(prob_lhs_s + lib_util.epsilon)
    return prob_nll_s


