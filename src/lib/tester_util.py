import os
import cv2
import json
import numpy as np
from sklearn.metrics import mean_squared_error
from collections import OrderedDict
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO


colors = ((64, 64, 64), (31, 119, 180), (174, 199, 232), (255, 127, 14),
          (255, 187, 120), (44, 160, 44), (152, 223, 138), (214, 39, 40),
          (255, 152, 150), (148, 103, 189), (197, 176, 213), (140, 86, 75),
          (196, 156, 148), (227, 119, 194), (247, 182, 210), (127, 127, 127),
          (199, 199, 199), (188, 189, 34), (219, 219, 141), (23, 190, 207),
          (158, 218, 229), (180, 119, 31))

# COCO Output Format
# Nose – 0, Right Shoulder – 1, Right Elbow – 2, Right Wrist – 3,
# Left Shoulder – 4, Left Elbow – 5, Left Wrist – 6, Right Hip – 7,
# Right Knee – 8, Right Ankle – 9, Left Hip – 10, Left Knee – 11,
# Left Ankle – 12, Right Eye – 13, Left Eye – 14, Right Ear – 15,
# Left Ear – 16
# Nose – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4,
# Left Shoulder – 5, Left Elbow – 6, Left Wrist – 7, Right Hip – 8,
# Right Knee – 9, Right Ankle – 10, Left Hip – 11, Left Knee – 12,
# Left Ankle – 13, Right Eye – 14, Left Eye – 15, Right Ear – 16,
# Left Ear – 17
coco_joint_idx_pairs = ((0, 13), (0, 14), (13, 15), (14, 16),
                        (1, 2), (2, 3), (4, 5), (5, 6),
                        (7, 8), (8, 9), (10, 11), (11, 12))

# MPII Output Format
# Head – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4,
# Left Shoulder – 5, Left Elbow – 6, Left Wrist – 7, Right Hip – 8,
# Right Knee – 9, Right Ankle – 10, Left Hip – 11, Left Knee – 12,
# Left Ankle – 13, Chest – 14, Background – 15
mpii_joint_idx_pairs = ()


def draw_joints(img_s, joints_s, max_joints=100, mode='coco'):
    # joint_idx_pairs = coco_joint_idx_pairs if mode == 'coco' else mpii_joint_idx_pairs
    n_joints = joints_s.shape[1] # 17 if mode == 'coco' else 16
    joints_img_s = img_s.copy()
    n_draw_joints = 0

    for i in range(joints_s.shape[0]):
        if n_draw_joints >= max_joints:
            break

        # color_idx = i % len(colors)
        # for joint_idx_pair in joint_idx_pairs:
        #     cv2.line(joints_img_s, tuple(joints_s[i, joint_idx_pair[0], :2]),
        #              tuple(joints_s[i, joint_idx_pair[1], :2]), colors[color_idx], 2)

        for j in range(n_joints):
            cv2.circle(joints_img_s, tuple(joints_s[i, j, :2]), 3, colors[j], 2)
        n_draw_joints += 1
    return joints_img_s


def draw_boxes(img_s, boxes_s, confs_s=None, labels_s=None,
               class_map=None, conf_thresh=0.0, max_boxes=100):

    box_img_s = img_s.copy()
    n_draw_boxes = 0
    n_wrong_boxes = 0
    n_thresh_boxes = 0
    for i, box in enumerate(boxes_s):
        try:
            l, t = int(round(box[0])), int(round(box[1]))
            r, b = int(round(box[2])), int(round(box[3]))
        except IndexError:
            print(boxes_s)
            print(i, box)
            print('IndexError')
            exit()

        if confs_s is not None:
            if conf_thresh > confs_s[i]:
                n_thresh_boxes += 1
                continue
        if (r - l <= 0) or (b - t <= 0):
            n_wrong_boxes += 1
            continue
        if n_draw_boxes >= max_boxes:
            continue

        conf_str = '-' if confs_s is None else '%0.3f' % confs_s[i]
        if labels_s is None:
            lab_str, color = '-', colors[i % len(colors)]
        else:
            lab_i = int(labels_s[i])
            lab_str = str(lab_i) if class_map is None else class_map[lab_i]
            color = colors[lab_i % len(colors)]

        box_img_s = cv2.rectangle(box_img_s, (l, t), (r, b), color, 2)
        l = int(l - 1 if l > 1 else r - 60)
        t = int(t - 8 if t > 8 else b)
        r, b = int(l + 60), int(t + 8)
        box_img_s = cv2.rectangle(box_img_s, (l, t), (r, b), color, cv2.FILLED)
        box_img_s = cv2.putText(box_img_s, '%s %s' % (conf_str, lab_str), (l + 1, t + 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255),
                                1, cv2.LINE_AA)
        n_draw_boxes += 1

    info_text = 'n_draw_b: %d, n_thr_b: %d, n_wrong_b: %d' % \
                (n_draw_boxes, n_thresh_boxes, n_wrong_boxes)
    if confs_s is not None:
        info_text += ', sum_of_conf: %.3f' % (np.sum(confs_s))
    else:
        info_text += ', sum_of_conf: -'

    box_img_s = cv2.rectangle(box_img_s, (0, 0), (350, 11), (0, 0, 0), cv2.FILLED)
    box_img_s = cv2.putText(box_img_s, info_text, (5, 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.25, (255, 255, 255), 1, cv2.LINE_AA)
    return box_img_s


def coco_eval(final_img_id, final_keypoints, final_score, final_center, final_scale, anno_file, res_folder):
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
    res_file = os.path.join(res_folder, 'keypoints_val_results.json')
    cat_results = []

    for img_idx in range(len(final_keypoints)):
        result = {
            'image_id': int(final_img_id[img_idx]),
            'category_id': int(1),
            'keypoints': np.reshape(final_keypoints[img_idx], (-1)).tolist(),
            'score': float(final_score[img_idx]),
            'center': final_center[img_idx][0].tolist(),
            'scale': final_scale[img_idx][0].tolist()
        }
        cat_results.append(result)

    with open(res_file, 'w') as f:
        json.dump(cat_results, f, sort_keys=True, indent=4)
    try:
        json.load(open(res_file))
    except Exception:
        content = []
        with open(res_file, 'r') as f:
            for line in f:
                content.append(line)
        content[-1] = ']'
        with open(res_file, 'w') as f:
            for c in content:
                f.write(c)

    info_str = _do_python_keypoint_eval(res_file, anno_file)
    name_value = OrderedDict(info_str)
    return name_value, name_value['AP']


def _do_python_keypoint_eval(res_file, anno_file):
    coco = COCO(anno_file)
    coco_dt = coco.loadRes(res_file)
    print("len", len(coco.anns))
    print("len_coco_dt", len(coco_dt.anns))
    coco_eval = COCOeval(coco, coco_dt, 'keypoints')
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

    info_str = []
    for ind, name in enumerate(stats_names):
        info_str.append((name, coco_eval.stats[ind]))

    print(info_str)
    return info_str


def PCKh(preds, gts, headboxes_src, list_label_preds):
    headboxes_src = [headboxes['headboxes_src'] for headboxes in headboxes_src] # ssh
    headboxes_src = np.asarray(headboxes_src)
    headsizes = [headboxes_src[:, 2] - headboxes_src[:, 0], headboxes_src[:, 3] - headboxes_src[:, 1]]
    headsizes = np.linalg.norm(headsizes, axis=0)
    headsizes *= 0.6 #SC_BIAS
    threshold = 0.5
    pckh = np.zeros((len(preds), len(gts)), dtype=np.int16)
    pckh_value = np.zeros((len(preds), len(gts), 16))
    TF_label_preds = np.zeros((len(preds), 16))

    for j in range(len(gts)):
        for i in range(len(preds)):

            jnt_visible = 1 - gts[j][:,2]
            uv_error = preds[i] - gts[j][:,:2]
            uv_err = np.linalg.norm(uv_error, axis=1)

            scale = np.multiply(headsizes[j], np.ones((len(uv_err), 1)))
            # scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))   # ssh
            scaled_uv_err = np.divide(uv_err, np.transpose(scale, (1,0))) #np.divide(uv_err, scale)
            scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
            jnt_count = np.sum(jnt_visible, axis=0)
            less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                              jnt_visible)
            pckh[i][j] = int(100 - np.average(np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)))
            pckh_value[i][j] = less_than_threshold

    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(pckh) ## col_ind is the index of gt for preds

    for gt_idx in range(len(col_ind)):
        TF_label_preds[gt_idx] = pckh_value[gt_idx][col_ind[gt_idx]]

    for gt_idx in range(len(preds)):
        list_label_preds.append(TF_label_preds[gt_idx])

    # from hungarian_algorithm import algorithm
    # algorithm.find_matching(pckh, matching_type='max', return_type='list')

    return TF_label_preds


def mpii_eval_multi(final_scores, final_joints, gt_joints, annotation_file='./'):
    head = 9
    lsho = 13
    lelb = 14
    lwri = 15
    lhip = 3
    lkne = 4
    lank = 5

    rsho = 12
    relb = 11
    rwri = 10
    rkne = 1
    rank = 0
    rhip = 2

    pelvis = 6
    thorax = 7
    neck = 8

    gt = json.load(open(annotation_file))
    TF_label_preds = []
    for idx_img in range(len(final_joints)):
        # print(final_joints[idx_img].shape)
        # print(gt_joints[idx_img].shape)
        # print(gt['images'][idx_img]['annotations'])
        # print(TF_label_preds, '\n')
        PCKh(final_joints[idx_img], gt_joints[idx_img], gt['images'][idx_img]['annotations'], TF_label_preds)
    preds_scores = np.concatenate(final_scores)
    TF_label_preds = np.transpose(TF_label_preds, (1, 0))

    from sklearn.metrics import average_precision_score
    ap = []
    for i in range(16):
        y_true = TF_label_preds[i]
        y_scores = preds_scores
        ap.append(average_precision_score(y_true, y_scores))

    name_value = [
        ('Head', '%.2f' % (100 * ap[head])),
        ('Shoulder', '%.2f' % (100 * (0.5 * (ap[lsho] + ap[rsho])))),
        ('Elbow', '%.2f' % (100 * (0.5 * (ap[lelb] + ap[relb])))),
        ('Wrist', '%.2f' % (100 * (0.5 * (ap[lwri] + ap[rwri])))),
        ('Hip',  '%.2f' % (100 * (0.5 * (ap[lhip] + ap[rhip])))),
        ('Knee',  '%.2f' % (100 * (0.5 * (ap[lkne] + ap[rkne])))),
        ('Ankle',  '%.2f' % (100 * (0.5 * (ap[lank] + ap[rank])))),
        ('Pelvis', '%.2f' % (100 * ap[pelvis])),
        ('Thorax', '%.2f' % (100 * ap[thorax])),
        ('Neck', '%.2f' % (100 * ap[neck])),
        ('Mean',  '%.2f' % (100 * (np.sum(ap)) / 16.0))
    ]
    print(name_value)
    # exit()
    return name_value


def mpii_eval(gt_dir, val_dir, val_to_coco_dir, preds):
    import json
    SC_BIAS = 0.6
    threshold = 0.5
    # preds = preds[:][:][0:2]# + 1.0
    # preds = np.asarray(preds)[:,:,0:2]
    gt_dict = json.load(open(gt_dir))
    val_dict = json.load(open(val_dir))
    val_to_coco = json.load(open(val_to_coco_dir))

    jnt_missing = gt_dict['jnt_missing']
    pos_gt_src = gt_dict['pos_gt_src']
    headboxes_src = gt_dict['headboxes_src']

    # pos_pred_src = np.transpose(preds, [1, 2, 0])

    pick_preds = []
    for idx in range(len(val_dict)):
        origin_file_name = val_dict[idx]['image']
        idx_img = val_to_coco['origin_filenames'].index(origin_file_name)
        min = 1000000000000000 #np.inf
        pick_idx_person = 0
        if len(preds[idx_img]) == 0:
            pick_preds.append(np.zeros((16, 2)))
            continue
        for idx_person in range(len(preds[idx_img])):
            if min > mean_squared_error(np.asarray(val_dict[idx_img]['center']),preds[idx_img][idx_person][6]):
                pick_idx_person = idx_person
                # min = np.linalg.norm(np.asarray(val_dict[idx_img]['center']) - preds[idx_img][idx_person][6])
                min = mean_squared_error(np.asarray(val_dict[idx_img]['center']),preds[idx_img][idx_person][6])
        pick_preds.append(preds[idx_img][pick_idx_person])
        # pick_preds.append(np.asarray(val_dict[idx]['joints']))

    pos_pred_src = np.transpose(pick_preds, (1, 2, 0))

    # pos_gt = []
    # head_gt = []
    # pos_vis_gt = []
    # for i in range(len(gt_dict['images'])):
    #     img_joints = []
    #     img_heads = []
    #     joints_vis = []
    #     for j in range(len(gt_dict['images'][i]['annotations'])):
    #         joints = np.reshape(np.asarray(gt_dict['images'][i]['annotations'][j]['keypoints']), (16, -1))
    #         img_joints.append(joints[:,:2])
    #         joints_vis.append(joints[:,2])
    #         img_heads.append(np.asarray(gt_dict['images'][i]['annotations'][j]['headboxes_src']))
    #         pos_gt.append(img_joints)
    #     head_gt.append(img_heads)
    #     pos_vis_gt.append(joints_vis)

    head = 9
    lsho = 13
    lelb = 14
    lwri = 15
    lhip = 3
    lkne = 4
    lank = 5

    rsho = 12
    relb = 11
    rwri = 10
    rkne = 1
    rank = 0
    rhip = 2


    jnt_visible = 1 - np.asarray(jnt_missing)
    uv_error = pos_pred_src - pos_gt_src
    uv_err = np.linalg.norm(uv_error, axis=1)
    headboxes_src = np.asarray(headboxes_src)
    headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
    # img_heads = np.asarray(img_heads)
    # headsizes = img_heads[:, 2] - img_heads[:, 0]
    headsizes = np.linalg.norm(headsizes, axis=0)
    headsizes *= SC_BIAS
    scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
    scaled_uv_err = np.divide(uv_err, scale)
    scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
    jnt_count = np.sum(jnt_visible, axis=1)
    less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                      jnt_visible)
    PCKh = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)

    # jnt_visible = 1 - np.asarray(jnt_missing)
    # uv_error = pos_pred_src - pos_gt_src
    # uv_err = np.linalg.norm(uv_error, axis=1)
    # headboxes_src = np.asarray(headboxes_src)
    # # headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
    # img_heads = np.asarray(img_heads)
    # headsizes = img_heads[:, 2] - img_heads[:, 0]
    # headsizes = np.linalg.norm(headsizes, axis=0)
    # headsizes *= SC_BIAS
    # scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
    # scaled_uv_err = np.divide(uv_err, scale)
    # scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
    # jnt_count = np.sum(jnt_visible, axis=1)
    # less_than_threshold = np.multiply((scaled_uv_err <= threshold),
    #                                   jnt_visible)
    # PCKh = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)

    # save
    rng = np.arange(0, 0.5 + 0.01, 0.01)
    pckAll = np.zeros((len(rng), 16))

    for r in range(len(rng)):
        threshold = rng[r]
        less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                          jnt_visible)
        pckAll[r, :] = np.divide(100. * np.sum(less_than_threshold, axis=1),
                                 jnt_count)

    PCKh = np.ma.array(PCKh, mask=False)
    PCKh.mask[6:8] = True

    jnt_count = np.ma.array(jnt_count, mask=False)
    jnt_count.mask[6:8] = True
    jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

    name_value = [
        ('Head', PCKh[head]),
        ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
        ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
        ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
        ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
        ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
        ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
        ('Mean', np.sum(PCKh * jnt_ratio)),
        ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
    ]

    print(name_value)
    # name_value = OrderedDict(name_value)
    return 0 #name_value, name_value['Mean']
