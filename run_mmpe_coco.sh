#!/usr/bin/env bash

# MPII joints
# 0: r ankle, 1: r knee,  2: r hip,     3: l hip,     4: l knee,      5: l ankle,     6: pelvis,    7: thorax,
# 8: neck,    9: head,    10: r wrist,  11: r elbow,  12: r shoulder, 13: l shoulder, 14: l elbow,  15: l wrist

python3 ./src/run.py \
--bash_file="./run_mmpe_coco.sh" \
--result_dir="./result/`(date "+%Y%m%d-%H%M%S")`_coco_17joints_biasrevised_laplace_lrelutoswish_size800_fcposesetting_lrdecayrevised" \
\
--load_dir="None" \
\
--global_args="{
    'n_classes': 2, 'batch_size': 16, 'max_batch_size': 32,
    'img_h': 800, 'img_w': 800, 'coord_h': 5, 'coord_w': 5,
    'n_joints': 17, 'is_coco': True, 'top_k': 100,
}" \
--training_args="{
    'init_iter': 0, 'max_iter': 270001, 'print_intv': 10,
#    'max_grad': 7, 'lr_decay_schd': {100000: 0.1, 170000: 0.1},
    'max_grad': 7, 'lr_decay_schd': {180000: 0.1, 260000: 0.1},
    'devices': [0, 1], 'sync_bnorm': True, 'port': 12355,
    'write_log': True,
}" \
\
--test_iters="[
    5000, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000, 220000, 240000, 260000, 270000]" \
--snapshot_iters="[5000, 50000, 100000, 130000, 150000, 170000, 190000, 210000, 230000, 250000, 270000]" \
\
--framework_info="{
    'framework': 'mmpe', 'framework_args': {},
}" \
--network_info="{
    'network': 'mmpe_test',
    'network_args': {
        'pretrained': True, 'backbone': 'res50fpn', 'fmap_ch': 256,
#        'dynamic_ch': 4389,
        'dynamic_ch': 25093,
        'xy_limit_factor': 1.0, 'std_factor': 0.05, 'max_batch_size': 32,
    },
}" \
--loss_func_info="{
    'loss_func': 'mmpe',
    'loss_func_args': {
        'lw_dict': {'joints_nll': 1.0, 'prob_nll': 0.0},
        'n_group_joints': 2, 'n_train_groups': 8, 'with_box': False,
#        'n_group_joints': 8, 'n_train_groups': 2, 'with_box': False,
        'joint_sampling': True, 'n_samples': 1,
        'oks_scale': 1.0, 'sim_thresh': 0.5,
        'joint_weight': [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            # 1.2, 1.2, 1.0, 1.0, 1.2, 1.2, 1.0, 0.8,
            # 0.8, 0.8, 1.0, 1.0, 0.8, 0.8, 1.0, 1.0,
        ],
    },
}" \
--post_proc_info="{
    'post_proc': 'mmpe_box',
    'post_proc_args': {
        'with_box': False, 'norm_pose': False,
        'pi_thresh': 0.0001, 'conf_thresh': 0.0001,
        'oks_scale': 1000, 'nms_thresh': 0.7,
        'vis_thresh': 0.5,
    },
}" \
--optimizer_info="{
    'optimizer': 'SGD',
    'optimizer_args': {
        'lr': 0.01, 'momentum': 0.9, 'weight_decay': 0.00005,
    },
#    'optimizer': 'Adam',
#    'optimizer_args': {
#        'lr': 0.001, 'weight_decay': 0.00005,
#    },
#    'optimizer': 'AdamW',
#    'optimizer_args': {
#        'lr': 0.001, 'weight_decay': 0.01,
#    },
}" \
--train_data_loader_info="{
    'dataset': 'coco_keypoint',
    'dataset_args': {
        'roots': ['./data/coco2017'],
        'types': ['train'],
        'pre_proc': 'augm', 'pre_proc_args': {
            'max_boxes': 300, 'norm_pose': False,
            'rgb_mean': [0.485, 0.456, 0.406],
            'rgb_std': [0.229, 0.224, 0.225],
            'with_img': True, 'crop': False,
        },
    },
    'shuffle': True, 'num_workers': 8, 'batch_size': 16,
}" \
--test_data_loader_info="{
    'dataset': 'coco_keypoint',
    'dataset_args': {
        'roots': ['./data/coco2017'],
        'types': ['val'],
        'pre_proc': 'base', 'pre_proc_args': {
            'max_boxes': 300, 'norm_pose': False,
            'rgb_mean': [0.485, 0.456, 0.406],
            'rgb_std': [0.229, 0.224, 0.225],
            'with_img': True, 'crop': False
        },
    },
    'shuffle': False, 'num_workers': 2, 'batch_size': 1,
}" \
--tester_info_list="[{
    'tester': 'image',
    'tester_args': {'n_images': 50, 'conf_thresh': 0.0001, 'max_joints': 20, 'draw_mode': 'coco'},
}, {
    'tester': 'coco_quant',
    'tester_args': {'annotation_path': './data/coco2017/annotations/person_keypoints_val2017.json'}
}]" \

