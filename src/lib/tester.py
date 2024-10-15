import os
import abc
import cv2
import tqdm
import numpy as np
from lib import util
from lib.external.dataset import factory
from . import tester_util


def get_tester_dict():
    return {
        'image': ImageTester,
        'mpii_quant': MPIIQuantTester,
        'coco_quant': COCOQuantTester,

        'pose_ae_image': PoseAEImageTester,
    }


class TesterABC(abc.ABC):
    def __init__(self, global_args, tester_args):
        self.global_args = global_args
        self.tester_args = tester_args

    @ abc.abstractmethod
    def run(self, framework, data_loader, result_dir):
        pass


class ImageTester(TesterABC):
    def __init__(self, global_args, tester_args):
        super(ImageTester, self).__init__(global_args, tester_args)
        self.n_images = tester_args['n_images']
        self.max_joints = tester_args['max_joints']
        self.conf_thresh = tester_args['conf_thresh']
        self.draw_mode = tester_args['draw_mode']

    def run(self, framework, data_loader, result_dir):
        assert data_loader.batch_size == 1
        pre_proc = data_loader.dataset.pre_proc
        util.make_dir(result_dir)

        for i, data_dict in enumerate(data_loader):
            if i >= self.n_images:
                break

            _, res_dict = framework.infer_forward(data_dict)
            pr_joints_s = util.cvt_torch2numpy(res_dict['joints_s'])[0]
            # pr_conf_s = util.cvt_torch2numpy(res_dict['conf_s'])[0]

            data_dict = pre_proc.inv_transform_batch(data_dict)
            img_s = data_dict['img'][0]
            # print(img_s.shape)
            gt_joints_s = data_dict['joints'][0]

            sort_idx = 0
            gt_img_path = os.path.join(result_dir, '%03d_%d_%s.png' % (i, sort_idx, 'gt'))
            gt_img_s = tester_util.draw_joints(img_s, gt_joints_s, self.max_joints)
            cv2.imwrite(gt_img_path, gt_img_s[:, :, ::-1])
            # print(gt_img_s.shape)

            # # ssh; remove non_vis keypoints
            # joints_vis = gt_joints_s[:, :, 2]
            # for k in range(len(pr_joints_s)):
            #     try:
            #         pr_joints_s[k] *= joints_vis[k].reshape(17, 1)
            #     except:
            #         pass

            sort_idx += 1
            pred_img_path = os.path.join(result_dir, '%03d_%d_%s.png' % (i, sort_idx, 'pred'))
            pr_img_s = tester_util.draw_joints(img_s, pr_joints_s, self.max_joints)
            cv2.imwrite(pred_img_path, pr_img_s[:, :, ::-1])

            data_dict.clear()
            del data_dict


class MPIIQuantTester(TesterABC):
    def __init__(self, global_args, tester_args):
        super(MPIIQuantTester, self).__init__(global_args, tester_args)
        self.annotation_path = tester_args['annotation_path']

    def run(self, framework, data_loader, result_dir):
        assert data_loader.batch_size == 1
        pre_proc = data_loader.dataset.pre_proc
        util.make_dir(result_dir)

        final_joints = []
        final_scores = []
        gt_joints = []

        # num_samples = data_loader.dataset.__len__()
        sample_pbar = tqdm.tqdm(data_loader)
        for i, data_dict in enumerate(sample_pbar):

            _, res_dict = framework.infer_forward(data_dict)
            pr_joints_s = util.cvt_torch2numpy(res_dict['joints_s'])[0]
            pr_conf_s = util.cvt_torch2numpy(res_dict['conf_s'])[0]

            # data_dict = pre_proc.inv_transform_batch(data_dict)
            # gt_joints_s = data_dict['keypoints'][0, :, 1:, :2]

            # ori_img = cv2.imread(data_loader.dataset.roidb[i]['image'])
            # h, w, _ = ori_img.shape
            h, w = util.cvt_torch2numpy(data_dict['img_size'].float())[0]
            w_pre, h_pre = pre_proc.input_size
            pr_joints_s = np.transpose(pr_joints_s, (2, 0, 1))
            pr_joints_s[0] *= (w / w_pre)
            pr_joints_s[1] *= (h / h_pre)
            pr_joints_s = np.transpose(pr_joints_s, (1, 2, 0))

            # gt_joints_s = np.transpose(gt_joints_s, (2, 0, 1))
            # gt_joints_s[0] *= (w / w_pre)
            # gt_joints_s[1] *= (h / h_pre)
            # gt_joints_s = np.transpose(gt_joints_s, (1, 2, 0))

            final_joints.append(pr_joints_s)
            final_scores.append(pr_conf_s)
            gt_joints.append(data_loader.dataset.roidb[i]['joints'])

            data_dict.clear()
            del data_dict

        results = tester_util.mpii_eval_multi(
            final_scores, final_joints, gt_joints, annotation_file=self.annotation_path)

        ap_file = os.path.join(result_dir, 'ap.txt')
        with open(ap_file, 'w') as file:
            file.write(str(results))


class COCOQuantTester(TesterABC):
    def __init__(self, global_args, tester_args):
        super(COCOQuantTester, self).__init__(global_args, tester_args)
        self.annotation_path = tester_args['annotation_path']
        self.global_args = global_args

    def run(self, framework, data_loader, result_dir):
        assert data_loader.batch_size == 1
        pre_proc = data_loader.dataset.pre_proc
        util.make_dir(result_dir)

        # final_joints = []
        final_scores = []
        final_center = []
        final_scale = []
        final_img_id = []
        final_keypoints = []

        sample_pbar = tqdm.tqdm(data_loader)
        for i, data_dict in enumerate(sample_pbar):
            tmp, res_dict = framework.infer_forward(data_dict)
            pr_joints_s = util.cvt_torch2numpy(res_dict['joints_s'])[0]
            pr_conf_s = util.cvt_torch2numpy(res_dict['conf_s'])[0]

            # ori_img = cv2.imread(data_loader.dataset.roidb[i]['image'])
            # h, w, _ = ori_img.shape
            h, w = util.cvt_torch2numpy(data_dict['img_size'].float())[0]
            h_pre, w_pre = pre_proc.input_size  # 320, 320

            pr_joints_s = np.transpose(pr_joints_s, (2, 0, 1))
            pr_joints_s[0] *= (w / w_pre)
            pr_joints_s[1] *= (h / h_pre)
            pr_joints_s = np.transpose(pr_joints_s, (1, 2, 0))

            # # ssh; remove non_vis keypoints
            # data_dict = pre_proc.inv_transform_batch(data_dict)
            # joints_vis = data_dict['joints'][0][:, :, 2]
            # for k in range(len(pr_joints_s)):
            #     try:
            #         pr_joints_s[k] *= joints_vis[k].reshape(17, 1)
            #     except:
            #         pass

            for cnt_joints in range(len(pr_joints_s)):
                # print(pr_joints_s.shape, pr_conf_s.shape)
                keypoints_scores = np.stack([pr_conf_s[cnt_joints]] * pr_joints_s[cnt_joints].shape[0], axis=0)
                keypoints_scores = np.expand_dims(keypoints_scores, axis=1)
                # print(keypoints_scores.shape)
                final_keypoints.append(np.concatenate([pr_joints_s[cnt_joints], keypoints_scores], axis=1))
                final_scores.append(pr_conf_s[cnt_joints])

                center, scale = self.joint_to_center_scale(pr_joints_s[cnt_joints])
                final_scale.append(scale)
                final_center.append(center)

                final_img_id.append(data_loader.dataset.roidb[i]['img_id'])
                # final_keypoints.append(np.concatenate(
                #     (pr_joints_s[cnt_joints], np.expand_dims(pr_conf_s[cnt_joints], axis=1)), axis=1))
            data_dict.clear()
            del data_dict

        results = tester_util.coco_eval(
            np.asarray(final_img_id), final_keypoints, final_scores,  final_center, final_scale,
            self.annotation_path, result_dir)
        ap_file = os.path.join(result_dir, 'ap.txt')
        with open(ap_file, 'w') as file:
            file.write(str(results))

        json_path = os.path.join(result_dir, 'keypoints_val_results.json')
        if os.path.exists(json_path):
            os.remove(json_path)

    def joint_to_center_scale(self, joints):
        n_joints, _ = joints.shape

        xmin, xmax = min(joints[:, 0]), max(joints[:, 0])
        ymin, ymax = min(joints[:, 1]), max(joints[:, 1])
        w, h = xmax - xmin,  ymax - ymin

        center = np.zeros((2), dtype=np.float32)
        center[0] = xmin + w * 0.5
        center[1] = ymin + h * 0.5
        scale = np.array([w * 1.0 / 200, h * 1.0 / 200], dtype=np.float32)

        if center[0] != -1:
            scale = scale * 1.25
        return center, scale


class PoseAEImageTester(TesterABC):
    def __init__(self, global_args, tester_args):
        super(PoseAEImageTester, self).__init__(global_args, tester_args)
        self.n_images = tester_args['n_images']

    def run(self, framework, data_loader, result_dir):
        pre_proc = data_loader.dataset.pre_proc
        util.make_dir(result_dir)

        for i, data_dict in enumerate(data_loader):
            if i >= self.n_images:
                break
            _, res_dict = framework.infer_forward(data_dict)

            data_dict = pre_proc.inv_transform_batch(data_dict)
            crop_img = data_dict['crop_img']
            gt_crop_joints = data_dict['crop_joints']
            rc_crop_joints = res_dict['rc_crop_joints']
            rc_crop_joints = util.cvt_torch2numpy(rc_crop_joints)

            for j, (crop_img_s, gt_crop_joints_s, rc_crop_joints_s) \
                    in enumerate(zip(crop_img, gt_crop_joints, rc_crop_joints)):
                # print('tester:', crop_img_s.shape, gt_crop_joints_s.shape, rc_crop_joints_s.shape)
                gt_img_s = tester_util.draw_joints(crop_img_s, np.expand_dims(gt_crop_joints_s, axis=0))
                rc_img_s = tester_util.draw_joints(crop_img_s, np.expand_dims(rc_crop_joints_s, axis=0))

                gt_img_path = os.path.join(result_dir, '%03d_%03d_0_gt.png' % (i, j))
                rc_img_path = os.path.join(result_dir, '%03d_%03d_1_rc.png' % (i, j))
                cv2.imwrite(gt_img_path, gt_img_s)
                cv2.imwrite(rc_img_path, rc_img_s)
