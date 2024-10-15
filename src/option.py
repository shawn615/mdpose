import os
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
# from torch.utils.data import DataLoader
from lib.network import get_network_dict
from lib.post_proc import get_post_proc_dict
from lib.loss_func import get_loss_func_dict
from lib.framework import get_framework_dict
from lib.dataset import get_dataset_dict
from lib.tester import get_tester_dict


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bash_file', type=str)
    parser.add_argument('--result_dir', type=str)
    parser.add_argument('--load_dir', type=str, default=None)

    parser.add_argument('--global_args', type=str)
    parser.add_argument('--framework_info', type=str)
    parser.add_argument('--network_info', type=str)
    parser.add_argument('--post_proc_info', type=str)
    parser.add_argument('--loss_func_info', type=str)
    parser.add_argument('--optimizer_info', type=str)

    parser.add_argument('--train_data_loader_info', type=str)
    parser.add_argument('--test_data_loader_info', type=str)
    parser.add_argument('--tester_info_list', type=str)
    parser.add_argument('--training_args', type=str)

    parser.add_argument('--snapshot_iters', type=str, default='')
    parser.add_argument('--test_iters', type=str, default='')

    args = parser.parse_args()
    args.global_args = cvt_str2python_data(args.global_args)
    args.framework_info = cvt_str2python_data(args.framework_info)
    args.network_info = cvt_str2python_data(args.network_info)
    args.post_proc_info = cvt_str2python_data(args.post_proc_info)
    args.loss_func_info = cvt_str2python_data(args.loss_func_info)
    args.optimizer_info = cvt_str2python_data(args.optimizer_info)

    args.train_data_loader_info = cvt_str2python_data(args.train_data_loader_info)
    args.test_data_loader_info = cvt_str2python_data(args.test_data_loader_info)
    args.tester_info_list = cvt_str2python_data(args.tester_info_list)

    args.training_args = cvt_str2python_data(args.training_args)
    args.snapshot_iters = cvt_str2python_data(args.snapshot_iters)
    args.test_iters = cvt_str2python_data(args.test_iters)

    args.result_dict_dict = dict()
    args.result_dir_dict = dict()
    args.result_dir_dict['root'] = args.result_dir
    args.result_dir_dict['src'] = os.path.join(args.result_dir, 'src')
    args.result_dir_dict['log'] = os.path.join(args.result_dir, 'log')
    args.result_dir_dict['test'] = os.path.join(args.result_dir, 'test')
    args.result_dir_dict['snapshot'] = os.path.join(args.result_dir, 'snapshot')
    return args


def cvt_str2python_data(arg_str):
    if isinstance(arg_str, str):
        python_data = yaml.full_load(arg_str)
    else:
        python_data = arg_str

    if isinstance(python_data, dict):
        for key, value in python_data.items():
            if value == 'None':
                python_data[key] = None
            elif isinstance(value, dict) or isinstance(value, list):
                python_data[key] = cvt_str2python_data(value)

    elif isinstance(python_data, list):
        for i, value in enumerate(python_data):
            if value == 'None':
                python_data[i] = None
            elif isinstance(value, dict) or isinstance(value, list):
                python_data[i] = cvt_str2python_data(value)
    return python_data


def create_loss_func(global_args, loss_func_info):
    loss_func_cls = get_loss_func_dict()[loss_func_info['loss_func']]
    return loss_func_cls(global_args, loss_func_info['loss_func_args'])


def create_network(global_args, network_info, loss_func):
    network_args = network_info['network_args']
    network_cls = get_network_dict()[network_info['network']]
    return network_cls(global_args, network_args, loss_func)


def create_post_proc(global_args, post_proc_info):
    post_proc_args = post_proc_info['post_proc_args']
    post_proc_cls = get_post_proc_dict()[post_proc_info['post_proc']]
    return post_proc_cls(global_args, post_proc_args)


def create_framework(global_args, framework_info, network, post_proc, world_size):
    framework_key = framework_info['framework']
    framework_args = framework_info['framework_args']
    framework_cls = get_framework_dict()[framework_key]
    return framework_cls(global_args, framework_args, network, post_proc, world_size)


def create_optimizer(optimizer_info, network):
    optimizer_dict = {
        'SGD': torch.optim.SGD,
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW,
    }

    optimizer_args = optimizer_info['optimizer_args']
    optimizer_args.update({'params': network.parameters()})
    optimizer = optimizer_dict[optimizer_info['optimizer']]
    return optimizer(**optimizer_args)


def create_dataset(global_args, data_loader_info):
    dataset_key = data_loader_info['dataset']
    dataset_args = data_loader_info['dataset_args']
    dataset = get_dataset_dict()[dataset_key](global_args, dataset_args)
    return dataset


def create_data_loader(dataset, data_loader_info, rank=0, world_size=1, with_sampler=False):
    batch_size = data_loader_info['batch_size']
    num_workers = data_loader_info['num_workers']
    shuffle = data_loader_info['shuffle']

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=shuffle, num_replicas=world_size, rank=rank) if with_sampler else None
    if sampler is not None:
        shuffle = False

    # print(dataset)
    # print(sampler)
    data_loader = DataLoader(
        dataset=dataset, batch_size=int(batch_size / world_size),
        shuffle=shuffle, num_workers=num_workers, sampler=sampler)
    return data_loader
# def create_data_loader(global_args, data_loader_info):
#     dataset_key = data_loader_info['dataset']
#     dataset_args = data_loader_info['dataset_args']
#
#     batch_size = data_loader_info['batch_size'] \
#         if 'batch_size' in data_loader_info.keys() else global_args['batch_size']
#     shuffle = data_loader_info['shuffle']
#     num_workers = data_loader_info['num_workers']
#
#     dataset = get_dataset_dict()[dataset_key](global_args, dataset_args)
#     data_loader = DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers)
#     if dataset.pre_proc.collate_fn is not None:
#         data_loader.collate_fn = dataset.pre_proc.collate_fn
#     return data_loader


def create_tester(global_args, tester_info):
    tester_class = get_tester_dict()[tester_info['tester']]
    return tester_class(global_args, tester_info['tester_args'])
