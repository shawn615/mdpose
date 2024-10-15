import cv2
import random
import numpy as np
from . import util as lib_util

cv2.setNumThreads(0)

def resize(img, boxes, keypoints, heatmap, size):
    shape = img.shape
    img = cv2.resize(img, (size[1], size[0]))
    if heatmap is not None:
        heatmap = cv2.resize(heatmap, (size[1], size[0]))

    boxes[:, [0, 2]] *= (size[1] / shape[1])
    boxes[:, [1, 3]] *= (size[0] / shape[0])

    keypoints[:, :, 0] *= (size[1] / shape[1])
    keypoints[:, :, 1] *= (size[0] / shape[0])
    return img, boxes, keypoints, heatmap


def rand_brightness(img, lower=-32, upper=32):
    if random.randint(0, 1) == 0:
        delta = random.uniform(lower, upper)
        img += delta
    return img


def rand_contrast(img, lower=0.5, upper=1.5):
    if random.randint(0, 1) == 0:
        alpha = random.uniform(lower, upper)
        img *= alpha
    return img


def expand(img, boxes, keypoints, heatmap, ratio_range=(1, 2)): ## (1,4)
    if np.random.randint(2):
        return img, boxes, keypoints, heatmap
    else:
        h, w, c = img.shape
        ratio = np.random.uniform(*ratio_range)

        left = int(np.random.uniform(0, w * ratio - w))
        top = int(np.random.uniform(0, h * ratio - h))

        expand_h, expand_w = int(h * ratio), int(w * ratio)
        expand_img = np.zeros((expand_h, expand_w, c), dtype=img.dtype)
        expand_heatmap = np.zeros((expand_h, expand_w, 17), dtype=img.dtype)
        expand_img[:, :, :] = np.mean(img, axis=(0, 1))
        expand_img[top:top + h, left:left + w] = img
        img = expand_img
        if heatmap is not None:
            expand_heatmap[top:top + h, left:left + w] = heatmap
            heatmap = expand_heatmap

        boxes = boxes.copy()
        boxes[:, 0:2] += (left, top)
        boxes[:, 2:4] += (left, top)
        keypoints[:, :, :2] += (left, top)
        return img, boxes, keypoints, heatmap


def rand_crop(
        img, boxes, labels, keypoints, heatmap, min_scale=0.3,
        iou_opts=(0.0, 0.1, 0.3, 0.7, 0.9, 1.0)):

    h, w, _ = img.shape
    while True:
        min_iou = np.random.choice(iou_opts)
        if min_iou >= 1.0:
            return img, boxes, labels, keypoints, heatmap
        else:
            max_iou = float('inf')

            _w = int(np.random.uniform(min_scale * w, w))
            _h = int(np.random.uniform(min_scale * h, h))
            if _h/_w < 0.5 or h/w > 2:
                continue
            left = int(np.random.uniform(w - _w))
            top = int(np.random.uniform(h - _h))

            rect = np.array([left, top, left+_w, top+_h])
            if len(boxes):
                overlap = lib_util.calc_jaccard_numpy(boxes, rect)
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                croped_img = img[rect[1]:rect[3], rect[0]:rect[2], :]
                if heatmap is not None:
                    croped_heatmap = heatmap[rect[1]:rect[3], rect[0]:rect[2], :]
                else:
                    croped_heatmap = None
                centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2.0

                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                mask = m1 * m2

                if not mask.any():
                    continue

                cur_boxes = boxes[mask, :].copy()
                cur_boxes[:, 0:2] = np.maximum(cur_boxes[:, 0:2], rect[0:2])
                cur_boxes[:, 0:2] -= rect[0:2]
                cur_boxes[:, 2:4] = np.minimum(cur_boxes[:, 2:4], rect[2:4])
                cur_boxes[:, 2:4] -= rect[0:2]
                cur_labels = labels[mask]

                cur_keypoints = keypoints[mask]
                v1 = (rect[0] <= cur_keypoints[:, :, 0]) * (rect[1] <= cur_keypoints[:, :, 1])
                v2 = (rect[2] > cur_keypoints[:, :, 0]) * (rect[3] > cur_keypoints[:, :, 1])
                vis = (v1 * v2).astype(np.int)
                cur_keypoints[:, :, 2] *= vis
                cur_keypoints[:, :, :2] -= rect[:2]
                return croped_img, cur_boxes, cur_labels, cur_keypoints, croped_heatmap
            else:
                return img, boxes, labels, keypoints, heatmap


def rand_flip(img, boxes, keypoints, heatmap, is_coco):
    if is_coco:
        flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]  # coco
    else:
        flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]   # mpii

    _, w, _ = img.shape
    if np.random.randint(2):
        img = img[:, ::-1]
        boxes[:, 0::2] = w - boxes[:, 2::-2]
        keypoints[:, :, 0] = w - keypoints[:, :, 0]

        # Change left-right parts
        # for i in range(len(keypoints)):
        keypoints = np.transpose(keypoints, (1, 2, 0))
        for pair in flip_pairs:
            # if np.random.randint(2):    # apply part-flipping -ssh
            keypoints[pair[0]+1, :, :], keypoints[pair[1]+1, :, :] = \
                keypoints[pair[1]+1, :, :], keypoints[pair[0]+1, :, :].copy()
            if heatmap is not None:
                heatmap[:, :, pair[0]], heatmap[:, :, pair[1]] = \
                    heatmap[:, :, pair[1]].copy(), heatmap[:, :, pair[0]].copy()
        keypoints = np.transpose(keypoints, (2, 0, 1))

    return img, boxes, keypoints, heatmap


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints*joints_vis, joints_vis


def rotation(img, boxes, coords, heatmap):
    rf = 30.0
    # rf = 45.0
    r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
        if random.random() <= 0.6 else 0
    h, w, _ = img.shape
    c = np.asarray([w/2.0, h/2.0])
    image_size = [w, h]
    trans = get_affine_transform(c, np.asarray([w/200, w/200]), r, image_size)
    img = cv2.warpAffine(img, trans, (int(image_size[0]), int(image_size[1])), flags=cv2.INTER_LINEAR)  # cause of dataloader deadlock
    if heatmap is not None:
        heatmap = cv2.warpAffine(heatmap, trans, (int(image_size[0]), int(image_size[1])), flags=cv2.INTER_LINEAR)

    for i in range(len(boxes)):
        boxes[i, 0:2] = affine_transform(boxes[i, 0:2], trans)
        boxes[i, 2:4] = affine_transform(boxes[i, 2:4], trans)

    for i in range(len(coords)):
        for j in range(len(coords[i])):
            if coords[i, j, 0] > 0.0:
                coords[i, j, 0:2] = affine_transform(coords[i, j, 0:2], trans)

    return img, boxes, coords, heatmap


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)

    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img,
                             trans,
                             (int(output_size[0]), int(output_size[1])),
                             flags=cv2.INTER_LINEAR)

    return dst_img


def crop_img_and_joints(img, boxes, joints, image_size, with_img):
    # print(img.shape, boxes.shape, joints.shape, image_size)
    crop_imgs = list()
    crop_joints = list()

    for box, joint in zip(boxes, joints):
        crop_wh = box[2:4] - box[:2]
        crop_joint = joint - box[:2]
        crop_joint *= (np.array(image_size) / crop_wh)
        crop_joints.append(crop_joint)

        if with_img:
            crop_img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            crop_img = cv2.resize(crop_img, image_size)
            crop_imgs.append(crop_img)

    crop_joints = np.stack(crop_joints)
    if with_img:
        crop_imgs = np.stack(crop_imgs)
    return crop_imgs, crop_joints


def crop_roi_img_and_keypoints(img, keypoints, keypoints_vis, c, s, image_size=(320, 320), n_joints=16):
    # flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
    flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
    r = 0
    check_flip = False
    rf = 40 ###   ROT_FACTOR: 40
    sf = 0.3 ###  SCALE_FACTOR: 0.3

    all_img = []
    all_joints = []
    all_joints_vis = []

    for idx in range(len(keypoints)):

        s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
            if random.random() <= 0.6 else 0

        joints = keypoints[idx]
        joints_vis = keypoints_vis[idx]

        # if random.random() <= 0.5:
        #     data_numpy = img[:, ::-1, :]
        #     joints, joints_vis = fliplr_joints(
        #         keypoints[idx], keypoints_vis[idx], data_numpy.shape[1], flip_pairs)
        #     c[idx][0] = data_numpy.shape[1] - c[idx][0] - 1

        trans = get_affine_transform(c[idx], s[idx], r, image_size)
        input = cv2.warpAffine(
            img,
            trans,
            (int(image_size[0]), int(image_size[1])),
            flags=cv2.INTER_LINEAR)

        # if transform:
        #     input = transform(input)

        for i in range(n_joints):
            if keypoints_vis[idx][i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
                # is visible: true
                if ((0 <= joints[i, 0]) and (joints[i, 0] < image_size[0])) \
                        and ((0 <= joints[i, 1]) and (joints[i, 1] < image_size[1])):
                    joints_vis[i, :] = 1.0
                    # joints[i, 2] = 1.0

        all_img.append(input)
        all_joints.append(joints)
        all_joints_vis.append(joints_vis)
    return np.asarray(all_img), np.asarray(all_joints), np.asarray(all_joints_vis)[:, :, 2]
