import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from lib import util as lib_util
import option
import util
import numpy as np
import random

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # torch.multiprocessing.set_sharing_strategy('file_system')

    print('[RUN] parse arguments')
    args = option.parse_options()

    print('[RUN] make result directories')
    lib_util.make_dir(args.result_dir_dict['root'])
    lib_util.make_dir(args.result_dir_dict['src'])
    util.copy_file(args.bash_file, args.result_dir)
    util.copy_dir('./src', args.result_dir_dict['src'])
    for name in args.result_dir_dict.keys():
        lib_util.make_dir(args.result_dir_dict[name])

    world_size = len(args.training_args['devices'])
    if world_size > 1:
        mp.spawn(ddp_train, args=(world_size, args,), nprocs=world_size, join=True)
    else:
        ddp_train(0, 1, args)


def ddp_train(rank, world_size, args):
    print('rank:', rank, 'world_size:', world_size)
    if world_size > 1:
        setup_ddp(args.training_args, rank, world_size=world_size)

    device_num = args.training_args['devices'][rank]
    with torch.cuda.device(device_num):
        loss_func = option.create_loss_func(args.global_args, args.loss_func_info)
        network = option.create_network(args.global_args, args.network_info, loss_func)
        network.build()
        network.cuda()
        network = nn.SyncBatchNorm.convert_sync_batchnorm(network) if args.training_args['sync_bnorm'] else network
        network_ddp = nn.parallel.DistributedDataParallel(
            network, device_ids=[device_num]) if world_size > 1 else network
        # network_ddp = nn.parallel.DistributedDataParallel(
        #     network, find_unused_parameters=True, device_ids=[device_num]) if world_size > 1 else network

        post_proc = option.create_post_proc(args.global_args, args.post_proc_info)
        framework = option.create_framework(args.global_args, args.framework_info, network, post_proc, 1)
        framework_ddp = option.create_framework(args.global_args, args.framework_info, network_ddp, post_proc, world_size)

        optimizer = option.create_optimizer(args.optimizer_info, network)
        load_snapshot(args.load_dir, network, optimizer)
        print('[OPTIMIZER] learning rate:', optimizer.param_groups[0]['lr'])

        train_dataset = option.create_dataset(args.global_args, args.train_data_loader_info)
        test_dataset = option.create_dataset(args.global_args, args.test_data_loader_info)
        train_data_loader = option.create_data_loader(
            train_dataset, args.train_data_loader_info, rank=rank, world_size=world_size, with_sampler=world_size > 1)
        test_data_loader = option.create_data_loader(test_dataset, args.test_data_loader_info)

        train_logger = SummaryWriter(args.result_dir_dict['log']) \
            if ((rank == 0) and args.training_args['write_log']) else None
        tester_dict = dict() if rank == 0 else None
        if tester_dict is not None:
            for tester_info in args.tester_info_list:
                tester_dict[tester_info['tester']] = option.create_tester(args.global_args, tester_info)

        n_batches = train_data_loader.__len__()
        global_step = args.training_args['init_iter']

        while True:
            # test_dir = os.path.join(args.result_dir_dict['test'], '%07d' % global_step)
            # run_testers(tester_dict, framework, test_data_loader, test_dir)
            # break
            start_time = time.time()
            for train_data_dict in train_data_loader:
                batch_time = time.time() - start_time

                if rank == 0 and (global_step in args.snapshot_iters):
                    snapshot_dir = os.path.join(args.result_dir_dict['snapshot'], '%07d' % global_step)
                    save_snapshot(framework.network, optimizer, snapshot_dir)

                if rank == 0 and (global_step in args.test_iters):
                    test_dir = os.path.join(args.result_dir_dict['test'], '%07d' % global_step)
                    run_testers(tester_dict, framework, test_data_loader, test_dir)

                if args.training_args['max_iter'] <= global_step:
                    break

                if global_step in args.training_args['lr_decay_schd'].keys():
                    decay_learning_rate(optimizer, args.training_args['lr_decay_schd'][global_step])

                # print("About to train one step")
                train_loss_dict, value_dict, train_time = \
                    train_network_one_step(args, framework_ddp, optimizer, train_data_dict)
                # print("Train one step done")

                if rank == 0 and (global_step % args.training_args['print_intv'] == 0):
                    iter_str = '[TRAINING] %d/%d:' % (global_step, args.training_args['max_iter'])
                    info_str = 'n_batches: %d, batch_time: %0.3f, train_time: %0.3f' % \
                               (n_batches, batch_time, train_time)
                    loss_str = util.cvt_dict2str(train_loss_dict)
                    print_str = iter_str + '\n- ' + info_str + '\n- ' + loss_str + '\n'

                    if train_logger is not None:
                        for key, value in train_loss_dict.items():
                            train_logger.add_scalar(key, value, global_step)

                    if len(value_dict.keys()) > 0:
                        value_str = util.cvt_dict2str(value_dict)
                        print_str += '- ' + value_str + '\n'

                        if train_logger is not None:
                            for key, value in value_dict.items():
                                train_logger.add_scalar(key, value, global_step)
                    print(print_str)

                train_loss_dict.clear()
                train_data_dict.clear()
                del train_loss_dict, train_data_dict
                global_step += 1

                start_time = time.time()
            if args.training_args['max_iter'] <= global_step:
                break
        if world_size > 1:
            cleanup_ddp()


def setup_ddp(training_args, rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(training_args['port'])
    dist.init_process_group('nccl', init_method='env://', rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


def train_network_one_step(args, framework, optimizer, train_data_dict):
    start_time = time.time()
    train_loss_dict, train_value_dict = framework.train_forward(train_data_dict)
    util.update_network(framework.network, optimizer, train_loss_dict, args.training_args['max_grad'])
    train_time = time.time() - start_time
    return train_loss_dict, train_value_dict, train_time


def run_testers(tester_dict, framework, test_data_loader, test_dir):
    lib_util.make_dir(test_dir)
    for key, tester in tester_dict.items():
        # test_data_loader = create_data_loader(test_dataset, args.test_data_loader_info)
        tester_dir = os.path.join(test_dir, key)
        tester.run(framework, test_data_loader, tester_dir)
        # test_data_loader.stop()
        # del test_data_loader
        print('[TEST] %s: %s' % (key, tester_dir))
    print('')


def save_snapshot(network, optimizer, save_dir):
    lib_util.make_dir(save_dir)
    network_path = os.path.join(save_dir, 'network.pth')
    optimizer_path = os.path.join(save_dir, 'optimizer.pth')
    network.save(network_path)
    torch.save(optimizer.state_dict(), optimizer_path)
    print('[OPTIMIZER] save: %s' % optimizer_path)
    print('')


def load_snapshot(load_dir, network, optimizer):
    if (load_dir is not None) and os.path.exists(load_dir):
        network_path = os.path.join(load_dir, 'network.pth')
        network.load(network_path)

        optimizer_path = os.path.join(load_dir, 'optimizer.pth')
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location='cpu'))
            print('[OPTIMIZER] load: %s' % optimizer_path)


def decay_learning_rate(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        old_lr = param_group['lr']
        new_lr = old_lr * decay_rate
        param_group['lr'] = new_lr
    print('[OPTIMIZER] learning rate: %f -> %f' % (old_lr, new_lr))


if __name__ == '__main__':
    main()
