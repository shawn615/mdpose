import abc
import numpy as np
import torch
from . import util as lib_util
from . import pre_proc_util as pre_util


def get_pre_proc_dict():
    return {
        'base': PreProcBase,
        'augm': PreProcAugm,

        'crop_base': CropPreProcBase,
        'crop_augm': CropPreProcAugm,
    }


class PreProcABC(abc.ABC):
    def __init__(self, global_args, pre_proc_args):
        self.n_joints = global_args['n_joints']
        self.n_classes = global_args['n_classes']
        self.input_size = (global_args['img_h'], global_args['img_w'])
        self.coord_range = (global_args['coord_h'], global_args['coord_w'])
        self.max_boxes = pre_proc_args['max_boxes']
        self.is_coco = global_args['is_coco']

        dummy_boxes, dummy_joints = \
            self.__create_dummy_boxes_with_keypoints__(self.max_boxes, self.max_boxes)
        self.dummy_boxes = dummy_boxes
        self.dummy_joints = dummy_joints

        self.collate_fn = None

    def __create_dummy_boxes_with_keypoints__(self, _n_dummies, _n_dummies_key):
        boxes = list()
        keypoints = list()
        for _ in range(_n_dummies):
            boxes.append(np.array([0, 0, 0, 0]))
        for _ in range(_n_dummies_key):
            keypoints.append(np.zeros((self.n_joints + 1, 3)))
        return np.array(boxes), np.array(keypoints)

    def __fill__(self, sample_dict):
        n_boxes = sample_dict['boxes'].shape[0]
        n_dummies = self.max_boxes - n_boxes

        n_keypoints = sample_dict['keypoints'].shape[0]
        n_dummies_key = self.max_boxes - n_keypoints

        if n_dummies > 0:
            dummy_boxes = self.dummy_boxes[:n_dummies].copy()
            dummpy_keypoints = self.dummy_joints[:n_dummies_key].copy()
            sample_dict['boxes'] = np.concatenate((sample_dict['boxes'], dummy_boxes), axis=0)
            sample_dict['boxes'] = sample_dict['boxes'].astype(np.float32)

            sample_dict['keypoints'] = np.concatenate((sample_dict['keypoints'], dummpy_keypoints), axis=0)
            sample_dict['keypoints'] = sample_dict['keypoints'].astype(np.float32)

        else:
            sample_dict['boxes'] = sample_dict['boxes'][:self.max_boxes]
            sample_dict['boxes'] = sample_dict['boxes'].astype(np.float32)
        return sample_dict

    @ abc.abstractmethod
    def __augment__(self, sample_dict):
        pass

    @ abc.abstractmethod
    def transform(self, sample_dict):
        pass

    @ abc.abstractmethod
    def inv_transform_batch(self, data_dict):
        pass

    @ abc.abstractmethod
    def process(self, sample_dict):
        pass


class PreProcBase(PreProcABC):
    def __init__(self, global_args, pre_proc_args):
        super(PreProcBase, self).__init__(global_args, pre_proc_args)
        self.rgb_mean = np.array(pre_proc_args['rgb_mean']).astype(np.float32).reshape(3, 1, 1)
        self.rgb_std = np.array(pre_proc_args['rgb_std']).astype(np.float32).reshape(3, 1, 1)
        self.norm_pose = pre_proc_args['norm_pose']

    def __augment__(self, sample_dict):
        return sample_dict

    def transform(self, sample_dict):
        s_dict = sample_dict
        s_dict['img'] = np.transpose(s_dict['img'], axes=(2, 0, 1)).astype(dtype=np.float32) / 255.0
        s_dict['img'] = (s_dict['img'] - self.rgb_mean) / self.rgb_std
        # s_dict['heatmap'] = np.transpose(s_dict['heatmap'], axes=(2, 0, 1)).astype(dtype=np.float32) / 255.0

        s_dict['boxes'][:, [0, 2]] *= (self.coord_range[1] / self.input_size[1])
        s_dict['boxes'][:, [1, 3]] *= (self.coord_range[0] / self.input_size[0])
        s_dict['keypoints'][:, :, 0] *= (self.coord_range[1] / self.input_size[1])
        s_dict['keypoints'][:, :, 1] *= (self.coord_range[0] / self.input_size[0])

        if self.norm_pose:
            # boxes:    (n_objects, 4)
            # joints:   (n_objects, n_joints, 3)
            boxes_ct = np.expand_dims((s_dict['boxes'][:, 0:2] + s_dict['boxes'][:, 2:4]) * 0.5, axis=1)
            boxes_wh = np.expand_dims(s_dict['boxes'][:, 2:4] - s_dict['boxes'][:, 0:2], axis=1)
            # s_dict['keypoints'][:, :, 0:2] = (s_dict['keypoints'][:, :, 0:2] - boxes_ct) * 0.1 + boxes_ct
            s_dict['keypoints'][:, :, 0:2] = ((s_dict['keypoints'][:, :, 0:2] - boxes_ct) / (boxes_wh * 0.5)) + boxes_ct
        return s_dict

    def inv_transform_batch(self, data_dict):
        d_dict = lib_util.cvt_torch2numpy(data_dict)
        d_dict['img'] = d_dict['img'] * self.rgb_std + self.rgb_mean
        d_dict['img'] = (np.transpose(d_dict['img'], axes=(0, 2, 3, 1)) * 255.0).astype(dtype=np.uint8)
        # d_dict['heatmap'] = (np.transpose(d_dict['heatmap'], axes=(0, 2, 3, 1)) * 255.0).astype(dtype=np.uint8)

        if self.norm_pose:
            # print(d_dict['boxes'].shape, d_dict['joints'].shape)
            boxes_ct = np.expand_dims((d_dict['boxes'][:, :, 0:2] + d_dict['boxes'][:, :, 2:4]) * 0.5, axis=2)
            boxes_wh = np.expand_dims(d_dict['boxes'][:, :, 2:4] - d_dict['boxes'][:, :, 0:2], axis=2)
            # d_dict['joints'][:, :, :, 0:2] = (d_dict['joints'][:, :, :, 0:2] - boxes_ct) * 10 + boxes_ct
            d_dict['joints'][:, :, :, 0:2] = (d_dict['joints'][:, :, :, 0:2] - boxes_ct) * (boxes_wh * 0.5) + boxes_ct

        d_dict['boxes'][:, :, [0, 2]] *= (self.input_size[1] / self.coord_range[1])
        d_dict['boxes'][:, :, [1, 3]] *= (self.input_size[0] / self.coord_range[0])
        d_dict['keypoints'][:, :, :, 0] *= (self.input_size[1] / self.coord_range[1])
        d_dict['keypoints'][:, :, :, 1] *= (self.input_size[0] / self.coord_range[0])
        d_dict['joints'][:, :, :, 0] *= (self.input_size[1] / self.coord_range[1])
        d_dict['joints'][:, :, :, 1] *= (self.input_size[0] / self.coord_range[0])
        return d_dict

    def process(self, sample_dict):
        sample_dict['img'] = np.array(sample_dict['img']).astype(np.float32)
        sample_dict['boxes'] = np.array(sample_dict['boxes']).astype(np.float32)
        sample_dict['labels'] = np.array(sample_dict['labels']).astype(np.float32)
        sample_dict['heatmap'] = None # np.array(sample_dict['heatmap']).astype(np.float32)
        sample_dict['img_id'] = np.array(sample_dict['img_id'])

        sample_dict['keypoints'] = np.array(sample_dict['keypoints']).astype(np.float32)
        sample_dict['keypoints_vis'] = np.array(sample_dict['keypoints_vis']).astype(np.float32)
        sample_dict['keypoints'][:, :, 2] = sample_dict['keypoints_vis'][:, :, 0]

        center_vis = np.ones([sample_dict['keypoints'].shape[0], 1])
        center = np.expand_dims(np.concatenate([sample_dict['center'], center_vis], axis=1), axis=1)
        sample_dict['keypoints'] = np.concatenate([center, sample_dict['keypoints']], axis=1)

        s_dict = self.__augment__(sample_dict)
        img_size = np.array(s_dict['img'].shape)[:2]
        s_dict['img'], s_dict['boxes'], s_dict['keypoints'], s_dict['heatmap'] = \
            pre_util.resize(s_dict['img'], s_dict['boxes'], s_dict['keypoints'], s_dict['heatmap'], self.input_size)
        s_dict['boxes'] = lib_util.clip_boxes_s(s_dict['boxes'], self.input_size, numpy=True)

        n_boxes, n_people = s_dict['boxes'].shape[0], s_dict['keypoints'].shape[0]
        s_dict.update({'n_boxes': np.array(n_boxes), 'n_people': np.array(n_people), 'img_size': np.array(img_size)})
        s_dict = self.transform(s_dict)
        s_dict = self.__fill__(s_dict)

        s_dict['center'] = s_dict['keypoints'][:, 0, :2]
        s_dict['joints'] = s_dict['keypoints'][:, 1:]

        s_dict.pop('heatmap', None)
        del s_dict['scales']
        del s_dict['labels']
        del s_dict['keypoints_vis']
        return s_dict


class PreProcAugm(PreProcBase):
    def transform(self, sample_dict):
        sample_dict = super(PreProcAugm, self).transform(sample_dict)
        # if self.crop:
        #     crop_img, crop_joints = sample_dict['crop_img'], sample_dict['crop_joints']
        #     crop_img = crop_img.astype(dtype=np.float32) / 255.0
        #     crop_img = crop_img.transpose(0, 3, 1, 2)
        #     crop_img = (crop_img - self.crop_rgb_mean) / self.crop_rgb_std
        #
        #     crop_joints[:, :, 0] *= (self.coord_range[1] / self.input_size[1])
        #     crop_joints[:, :, 1] *= (self.coord_range[0] / self.input_size[0])
        #     sample_dict.update({'crop_img': crop_img, 'crop_joints': crop_joints})
        return sample_dict

    def inv_transform_batch(self, data_dict):
        data_dict = super(PreProcAugm, self).inv_transform_batch(data_dict)
        # if self.crop:
        #     data_dict['crop_img'] = data_dict['crop_img'] * self.crop_rgb_std + self.crop_rgb_mean
        #     data_dict['crop_img'] = (
        #             np.transpose(data_dict['crop_img'], axes=(0, 2, 3, 1)) * 255.0).astype(dtype=np.uint8)
        #
        #     data_dict['crop_joints'][:, :, :, 0] *= (self.input_size[1] / self.coord_range[1])
        #     data_dict['crop_joints'][:, :, :, 1] *= (self.input_size[0] / self.coord_range[0])
        return data_dict

    def __augment__(self, sample_dict):
        img = np.array(sample_dict['img'])
        boxes = np.array(sample_dict['boxes'])
        labels = np.array(sample_dict['labels'])
        heatmap = None # np.array(sample_dict['heatmap'])

        keypoints = np.array(sample_dict['keypoints'])
        keypoints_vis = np.array(sample_dict['keypoints_vis'])
        center = np.array(sample_dict['center'])
        scale = np.array(sample_dict['scales'])

        img_id = np.array(sample_dict['img_id'])

        # rotation by jihye
        img, boxes, keypoints, heatmap = pre_util.rotation(img, boxes, keypoints, heatmap)
        img, boxes, keypoints, heatmap = pre_util.expand(img, boxes, keypoints, heatmap)
        img, boxes, labels, keypoints, heatmap = pre_util.rand_crop(img, boxes, labels, keypoints, heatmap)
        img, boxes, keypoints, heatmap = pre_util.rand_flip(img, boxes, keypoints, heatmap, self.is_coco)

        sample_dict['img'] = img
        sample_dict['boxes'] = boxes
        sample_dict['labels'] = labels
        sample_dict['keypoints'] = keypoints
        sample_dict['heatmap'] = heatmap

        return sample_dict


class CropPreProcBase(PreProcABC):
    def __init__(self, global_args, pre_proc_args):
        super(CropPreProcBase, self).__init__(global_args, pre_proc_args)
        self.input_size = (global_args['img_h'], global_args['img_w'])
        self.rgb_mean = np.array(pre_proc_args['rgb_mean']).astype(np.float32).reshape(1, 3, 1, 1)
        self.rgb_std = np.array(pre_proc_args['rgb_std']).astype(np.float32).reshape(1, 3, 1, 1)
        self.with_img = pre_proc_args['with_img']
        self.collate_fn = self.__collate__

    def __collate__(self, sample_dict_list):
        data_dict = {'crop_joints': [], 'crop_joints_vis': []}
        if self.with_img:
            data_dict['crop_img'] = []

        for sample_dict in sample_dict_list:
            if self.with_img:
                data_dict['crop_img'].append(sample_dict['crop_img'])
            data_dict['crop_joints'].append(sample_dict['crop_joints'])
            data_dict['crop_joints_vis'].append(sample_dict['crop_joints_vis'])

        if self.with_img:
            data_dict['crop_img'] = torch.from_numpy(np.concatenate(data_dict['crop_img'], axis=0))
        data_dict['crop_joints'] = torch.from_numpy(np.concatenate(data_dict['crop_joints'], axis=0))
        data_dict['crop_joints_vis'] = torch.from_numpy(np.concatenate(data_dict['crop_joints_vis'], axis=0))
        return data_dict

    def __augment__(self, sample_dict):
        return sample_dict

    def transform(self, sample_dict):
        if self.with_img:
            sample_dict['crop_img'] = np.transpose(sample_dict['crop_img'], axes=(0, 3, 1, 2)) / 255.0
            sample_dict['crop_img'] = (sample_dict['crop_img'] - self.rgb_mean) / self.rgb_std
        sample_dict['crop_joints'][:, :, 0] *= (self.coord_range[1] / self.input_size[1])
        sample_dict['crop_joints'][:, :, 1] *= (self.coord_range[0] / self.input_size[0])
        return sample_dict

    def inv_transform_batch(self, data_dict):
        data_dict = lib_util.cvt_torch2numpy(data_dict)
        if self.with_img:
            data_dict['crop_img'] = data_dict['crop_img'] * self.rgb_std + self.rgb_mean
            data_dict['crop_img'] = (np.transpose(data_dict['crop_img'], axes=(0, 2, 3, 1)) * 255.0).astype(dtype=np.uint8)
        data_dict['crop_joints'][:, :, 0] *= (self.input_size[1] / self.coord_range[1])
        data_dict['crop_joints'][:, :, 1] *= (self.input_size[0] / self.coord_range[0])
        return data_dict

    def process(self, sample_dict):
        sample_dict['img'] = np.array(sample_dict['img']).astype(np.float32)
        sample_dict['boxes'] = np.array(sample_dict['boxes']).astype(np.float32)
        sample_dict['keypoints'] = np.array(sample_dict['keypoints']).astype(np.float32)[:, :, :2]
        sample_dict['keypoints_vis'] = np.array(sample_dict['keypoints_vis']).astype(np.float32)
        sample_dict['center'] = np.array(sample_dict['center'])
        sample_dict['scale'] = np.array(sample_dict['scales'])

        augm_sample_dict = self.__augment__(sample_dict)
        crop_img, crop_joints = pre_util.crop_img_and_joints(
            augm_sample_dict['img'], augm_sample_dict['boxes'],
            augm_sample_dict['keypoints'], self.input_size, self.with_img)

        if self.with_img:
            augm_sample_dict['crop_img'] = crop_img
        augm_sample_dict['crop_joints'] = crop_joints
        augm_sample_dict['crop_joints_vis'] = augm_sample_dict['keypoints_vis'][:, :, 0]

        augm_sample_dict = self.transform(augm_sample_dict)
        return augm_sample_dict


class CropPreProcAugm(PreProcABC):
    def __augment__(self, sample_dict):
        img = np.array(sample_dict['img'])
        boxes = np.array(sample_dict['boxes'])
        keypoints = np.array(sample_dict['keypoints'])
        sample_dict['img'], sample_dict['boxes'], sample_dict['keypoints'] = \
            pre_util.rand_flip(img, boxes, keypoints)
        return sample_dict
