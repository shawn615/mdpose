# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import os
import numpy as np
import scipy.sparse
import pickle
import json
import uuid
# COCO API
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .imdb import imdb
from lib.external.dataset import dataset_util as ds_util
import shutil


class coco(imdb):
    def __init__(self, image_set, year, data_path):
        if 'coco' in data_path:
            imdb.__init__(self, 'coco_' + year + '_' + image_set)
            self.num_joints = 17
        else:
            imdb.__init__(self, 'mpii_' + year + '_' + image_set)
            self.num_joints = 16
        # COCO specific config options
        self.config = {'use_salt': True, 'cleanup': True}
        # name, paths
        self._year = year
        self._image_set = image_set
        # self._data_path = osp.join(cfg.DATA_DIR, 'coco')
        # load COCO API, classes, class <-> id mappings
        self._data_path = data_path

        self.image_width = 192
        self.image_height = 256
        # self.num_joints = 17 # 17
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200

        self._COCO = COCO(self._get_ann_file())
        cats = self._COCO.loadCats(self._COCO.getCatIds())
        self._classes = tuple(['__background__'] + [c['name'] for c in cats])
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self._class_to_coco_cat_id = dict(list(zip([c['name'] for c in cats], self._COCO.getCatIds())))
        self._image_index = self._load_image_set_index()
        self._image_index_tmp = self._load_image_set_index()

        # Default to roidb handler
        self.set_proposal_method('gt')
        self.competition_mode(False)

        # Some image sets are "views" (i.e. subsets) into others.
        # For example, minival2014 is a random 5000 image subset of val2014.
        # This mapping tells us where the view's images and proposals come from.
        self._view_map = {
            'minival2014': 'val2014',    # 5k val2014 subset
            'valminusminival2014': 'val2014',    # val2014 \setminus minival2014
            'test-dev2015': 'test2015',
            'valminuscapval2014': 'val2014',
            'capval2014': 'val2014',
            'captest2014': 'val2014',
            'test-dev2017': 'test2017',
            'train2017': 'trainval2014',
            'val2017': 'minival2014'
        }
        coco_name = image_set + year    # e.g., "val2014"
        self._data_name = (self._view_map[coco_name]
                           if coco_name in self._view_map else coco_name)
        # Dataset splits that have ground-truth annotations (test splits
        # do not have gt annotations)
        self._gt_splits = ('train', 'val', 'minival')

    def set_data_path(self, data_path):
        self._data_path = data_path

    def _get_ann_file(self):
        if 'coco' in self._data_path:
            prefix = 'person_keypoints' \
                if 'test' not in self._image_set else 'image_info'
            print(os.path.join(self._data_path, 'annotations', prefix + '_' + self._image_set +  self._year +'.json'))
            return os.path.join(self._data_path, 'annotations', prefix + '_' + self._image_set +  self._year +'.json')
        else:
            print(os.path.join(self._data_path, 'annot2', self._image_set+'.json'))
            return os.path.join(self._data_path, 'annot2', self._image_set+'.json')

    def _load_image_set_index(self):
        """
        Load image ids.
        """
        image_ids = self._COCO.getImgIds()
        return image_ids

    def _get_widths(self):
        anns = self._COCO.loadImgs(self._image_index)
        widths = [ann['width'] for ann in anns]
        return widths

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self._image_index[i]

    def image_path_from_index(self, index, coco2014=False):
        """
        Construct an image path from the image's "index" identifier.
        """
        # Example image path for index=119993:
        #   images/train2014/COCO_train2014_000000119993.jpg

        file_name = ''
        if coco2014:
            file_name += ('COCO_' + self._data_name + '_' +
                          str(index).zfill(12) + '.jpg')
            image_path = osp.join(self._data_path, 'images',
                                  self._data_name, file_name)
            if not osp.exists(image_path):
                file_name = ''
                file_name += ('COCO_' + 'train2014' + '_' +
                              str(index).zfill(12) + '.jpg')
                image_path = osp.join(self._data_path, 'images',
                                      self._data_name, file_name)
                if not osp.exists(image_path):
                    file_name = ''
                    file_name += ('COCO_' + 'val2014' + '_' +
                                  str(index).zfill(12) + '.jpg')
        else:
            file_name += (str(index).zfill(12) + '.jpg')
        image_path = osp.join(self._data_path, 'images',
                              self._data_name, file_name)
        # print("file_name : {}".format(file_name))
        # print("image_path: {}".format(image_path))

        # mpii dataset
        if not osp.exists(image_path):
            file_name = ''
            file_name += (str(index).zfill(12) + '.jpg')
            image_path = osp.join(self._data_path, 'images', self._data_name[:-4], file_name)

        assert osp.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')

        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                load_pkl_dict = pickle.load(fid)
            self._image_index = load_pkl_dict['img_idx']
            roidb = load_pkl_dict['roidb']

            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_coco_annotation(index)
                    for index in self._image_index]
        ## remove None image
        gt_roidb = list(filter((None).__ne__, gt_roidb))
        self._image_index = self._image_index_tmp

        pkl_dict = {'roidb': gt_roidb, 'img_idx': self._image_index}
        with open(cache_file, 'wb') as fid:
            pickle.dump(pkl_dict, fid, pickle.HIGHEST_PROTOCOL)
            # pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def _load_coco_annotation(self, index):
        """
        Loads COCO bounding-box instance annotations. Crowd instances are
        handled by marking their overlaps (with all categories) to -1. This
        overlap value means that crowd "instances" are excluded from training.
        """
        im_ann = self._COCO.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self._COCO.getAnnIds(imgIds=index, iscrowd=None)
        objs = self._COCO.loadAnns(annIds)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        if num_objs == 0:
            if index in self._image_index_tmp:
                self._image_index_tmp.remove(index)
            return

        boxes = [] #np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = [] #np.zeros((num_objs), dtype=np.int32)
        overlaps = [] #np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = [] #np.zeros((num_objs), dtype=np.float32)

        joints = [] #np.zeros((num_objs, self.num_joints, 3), dtype=np.float32)
        joints_vis = [] #np.zeros((num_objs, self.num_joints, 3), dtype=np.uint16)
        centers = []
        scales = []
        headboxes = []

        # Lookup table to map from COCO category ids to our internal class
        # indices
        coco_cat_id_to_class_ind = dict([(self._class_to_coco_cat_id[cls],
                                          self._class_to_ind[cls])
                                         for cls in self._classes[1:]])

        for ix, obj in enumerate(objs):
            cls = coco_cat_id_to_class_ind[obj['category_id']]
            if max(obj['keypoints']) == 0 and self._image_set == 'train':
                continue
            tmp_clean_box = obj['clean_bbox'][:4].copy()
            c, s = self._box2cs([tmp_clean_box[0], tmp_clean_box[1], tmp_clean_box[2] - tmp_clean_box[0], tmp_clean_box[3]-tmp_clean_box[1]])
            centers.append(c)
            scales.append(s)

            boxes.append(obj['clean_bbox'])
            gt_classes.append(cls)
            seg_areas.append(obj['area'])
            if obj['iscrowd']:
                # Set overlap to -1 for all classes for crowd objects
                # so they will be excluded during training
                overlaps.append(-1.0)
            else:
                overlaps.append(1.0)

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
            for ipt in range(self.num_joints):
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0

            joints.append(joints_3d)
            joints_vis.append(joints_3d_vis)

        if len(boxes) == 0:
            if index in self._image_index_tmp:
                self._image_index_tmp.remove(index)
            return

        boxes = np.asarray(boxes, dtype=np.uint16)
        overlaps = np.asarray(overlaps, dtype=np.float32)
        ds_util.validate_boxes(boxes, width=width, height=height)
        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'width': width,
                'height': height,
                'boxes': boxes,
                'gt_classes': np.asarray(gt_classes, dtype=np.int32),
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': np.asarray(seg_areas, dtype=np.float32),
                'joints': np.asarray(joints),
                'joints_vis': np.asarray(joints_vis),
                'centers': np.asarray(centers),
                'scales': np.asarray(scales),
                # 'headboxes': np.asarray(headboxes)
                 }

    def _get_widths(self):
        return [r['width'] for r in self.roidb]

    def append_flipped_images(self):
        num_images = self.num_images
        widths = self._get_widths()
        for i in range(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'width': widths[i],
                     'height': self.roidb[i]['height'],
                     'boxes': boxes,
                     'gt_classes': self.roidb[i]['gt_classes'],
                     'gt_overlaps': self.roidb[i]['gt_overlaps'],
                     'flipped': True,
                     'seg_areas': self.roidb[i]['seg_areas']}

            self.roidb.append(entry)
        self._image_index = self._image_index * 2

    def _get_box_file(self, index, coco2014=False):
        # first 14 chars / first 22 chars / all chars + .mat
        # COCO_val2014_0/COCO_val2014_000000447/COCO_val2014_000000447991.mat
        file_name = ''
        if coco2014:
            file_name += ('COCO_' + self._data_name + '_' + str(index).zfill(12) + '.mat')
        else:
            file_name += (str(index).zfill(12) + '.jpg')
        return osp.join(file_name[:14], file_name[:22], file_name)

    def _print_detection_eval_metrics(self, coco_eval):
        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95

        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        ap_default = np.mean(precision[precision > -1])
        print(('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
               '~~~~').format(IoU_lo_thresh, IoU_hi_thresh))
        print('{:.1f}'.format(100 * ap_default))
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            # minus 1 because of __background__
            precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
            ap = np.mean(precision[precision > -1])
            print('{:.1f}'.format(100 * ap))

        print('~~~~ Summary metrics ~~~~')
        coco_eval.summarize()

    def _do_detection_eval(self, res_file, output_dir):
        ann_type = 'bbox'
        coco_dt = self._COCO.loadRes(res_file)
        coco_eval = COCOeval(self._COCO, coco_dt)
        coco_eval.params.useSegm = (ann_type == 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        self._print_detection_eval_metrics(coco_eval)
        # eval_file = osp.join(output_dir, 'detection_results.pkl')
        # with open(eval_file, 'wb') as fid:
        #   pickle.dump(coco_eval, fid, pickle.HIGHEST_PROTOCOL)
        # print('Wrote COCO eval results to: {}'.format(eval_file))

    def _coco_results_one_category(self, boxes, cat_id):
        results = []
        for im_ind, index in enumerate(self.image_index):
            dets = boxes[im_ind].astype(np.float)
            # if dets == []:
            #   continue
            if len(boxes[im_ind]) == 0:
                continue
            scores = dets[:, -1]
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs + 1
            hs = dets[:, 3] - ys + 1
            results.extend(
                [{'image_id': index,
                  'category_id': cat_id,
                  'bbox': [xs[k], ys[k], ws[k], hs[k]],
                  'score': scores[k]} for k in range(dets.shape[0])])
        return results

    def _write_coco_results_file(self, all_boxes, res_file):
        # [{"image_id": 42,
        #   "category_id": 18,
        #   "bbox": [258.15,41.29,348.26,243.78],
        #   "score": 0.236}, ...]
        results = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind, self.num_classes - 1))
            coco_cat_id = self._class_to_coco_cat_id[cls]
            results.extend(self._coco_results_one_category(all_boxes[cls_ind], coco_cat_id))
        print('Writing results json to {}'.format(res_file))
        with open(res_file, 'w') as fid:
            json.dump(results, fid)

    def evaluate_detections(self, all_boxes, output_dir):
        res_file = osp.join(output_dir, ('detections_' + self._image_set + self._year + '_results'))
        if self.config['use_salt']:
            res_file += '_{}'.format(str(uuid.uuid4()))
        res_file += '.json'
        self._write_coco_results_file(all_boxes, res_file)
        # Only do evaluation on non-test sets
        if self._image_set.find('test') == -1:
            self._do_detection_eval(res_file, output_dir)
        # Optionally cleanup results json file
        if self.config['cleanup']:
            os.remove(res_file)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True