import os
import abc
import cv2
import random
import numpy as np
from torch.utils.data.dataset import Dataset
from .pre_proc import get_pre_proc_dict
from lib.external.dataset.roidb import combined_roidb


def get_dataset_dict():
    return {
        # 'coco': COCODataset,
        'coco_keypoint': COCOKeypointDataset,
        'mpii': MPIIDataset
    }


class DatasetABC(abc.ABC, Dataset):
    def __init__(self, global_args, dataset_args):
        super(DatasetABC, self).__init__()
        self.global_args = global_args
        self.roots = dataset_args['roots']
        self.types = dataset_args['types']
        pre_proc_class = get_pre_proc_dict()[dataset_args['pre_proc']]
        self.pre_proc = pre_proc_class(global_args, dataset_args['pre_proc_args'])

    @ abc.abstractmethod
    def shuffle(self):
        pass

    @ abc.abstractmethod
    def get_name2number_map(self):
        pass

    @ abc.abstractmethod
    def get_number2name_map(self):
        pass

    @ abc.abstractmethod
    def get_dataset_roots(self):
        pass


class COCOKeypointDataset(DatasetABC):
    def __init__(self, global_args, dataset_args):
        super(COCOKeypointDataset, self).__init__(global_args, dataset_args)

        self.is_coco = global_args['is_coco']
        if self.is_coco:
            imdb_names = "coco_2017_" + dataset_args['types'][0]
            # imdb_names = "coco_2017_" + type.replace('_', '-')
            self.roidb = combined_roidb(imdb_names, self.roots[0])
            self.data_size = len(self.roidb)
        else:
            imdb_names = "mpii_0000_" + dataset_args['types'][0]
            # imdb_names = "coco_2017_" + type.replace('_', '-')
            self.roidb = combined_roidb(imdb_names, self.roots[0])
            self.data_size = len(self.roidb)
            # print("data size: {}".format(self.data_size))

        self.number2name_map = {0: 'background', 1: 'person'}
        self.name2number_map = {'background': 0, 'person': 1}

    def __getitem__(self, index):
        minibatch_db = self.roidb[index]
        img = cv2.imread(minibatch_db['image'])[:, :, ::-1]

        if len(img.shape) == 2:
            img = np.repeat(np.expand_dims(img, axis=2), 3, axis=2)
        boxes = minibatch_db['boxes']
        labels = minibatch_db['gt_classes']
        keypoints = minibatch_db['joints']
        keypoints_vis = minibatch_db['joints_vis']
        center = minibatch_db['centers']
        scales = minibatch_db['scales']
        img_id = minibatch_db['img_id']


        sample_dict = {'img': img, 'boxes': boxes, 'labels': labels, 'heatmap': None,
                       'keypoints': keypoints, 'keypoints_vis': keypoints_vis,
                       'center': center, 'scales': scales, 'img_id': img_id} #, 'headboxes': headboxes}
        sample_dict = self.pre_proc.process(sample_dict)
        return sample_dict

    def __len__(self):
        return len(self.roidb)

    def shuffle(self):
        random.shuffle(self.roidb)

    def get_number2name_map(self):
        return self.number2name_map

    def get_name2number_map(self):
        return self.name2number_map

    def get_dataset_roots(self):
        return self.roots


class MPIIDataset(DatasetABC):
    def __init__(self, global_args, dataset_args):
        super(MPIIDataset, self).__init__(global_args, dataset_args)

        imdb_names = "mpii_" + dataset_args['types'][0]
        # imdb_names = "coco_2017_" + type.replace('_', '-')
        self.roidb = combined_roidb(imdb_names, self.roots[0])
        self.data_size = len(self.roidb)

        self.number2name_map = {0: 'background', 1: 'person'}
        self.name2number_map = {'background': 0, 'person': 1}

    def __getitem__(self, index):
        minibatch_db = self.roidb[index]
        img = cv2.imread(minibatch_db['image'])[:, :, ::-1]

        if len(img.shape) == 2:
            img = np.repeat(np.expand_dims(img, axis=2), 3, axis=2)
        boxes = minibatch_db['boxes']
        labels = minibatch_db['gt_classes']
        keypoints = minibatch_db['joints']
        keypoints_vis = minibatch_db['joints_vis']
        center = minibatch_db['centers']
        scales = minibatch_db['scales']
        img_id = minibatch_db['img_id']

        sample_dict = {'img': img, 'boxes': boxes, 'labels': labels, 'heatmap': None,
                       'keypoints': keypoints, 'keypoints_vis': keypoints_vis,
                       'center': center, 'scales': scales, 'img_id': img_id} #, 'headboxes': headboxes}
        sample_dict = self.pre_proc.process(sample_dict)
        return sample_dict

    def __len__(self):
        return len(self.roidb)

    def shuffle(self):
        random.shuffle(self.roidb)

    def get_number2name_map(self):
        return self.number2name_map

    def get_name2number_map(self):
        return self.name2number_map

    def get_dataset_roots(self):
        return self.roots

