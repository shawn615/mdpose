import os
import shutil
import torch.nn as nn
from lib import util as lib_util


def cvt_dict2str(value_dict):
    result_str = ''
    for key, value in value_dict.items():
        result_str += ('%s: %.7f, ' % (key, value))
    result_str = result_str[:-2]
    return result_str.rstrip()


def create_result_dir(result_dir, names=None):
    result_dir_dict = dict()
    lib_util.make_dir(result_dir)
    for name in names:
        dir_path = os.path.join(result_dir, name)
        lib_util.make_dir(dir_path)
        result_dir_dict[name] = dir_path
    return result_dir_dict


def copy_file(src_path, dst_dir):
    src_file = src_path.split('/')[-1]
    dst_path = os.path.join(dst_dir, src_file)
    shutil.copyfile(src_path, dst_path)


def copy_dir(src, dst, symlinks=False, ignore=None):

    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def update_learning_rate(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        old_lr = param_group['lr']
        new_lr = old_lr * decay_rate
        param_group['lr'] = new_lr
    print('[OPTIMIZER] learning rate: %f -> %f' % (old_lr, new_lr))
    print('')


def update_network(network, optimizer, loss_dict, max_grad=None):
    sum_loss = 0
    for _, loss in loss_dict.items():
        sum_loss += loss
    optimizer.zero_grad()
    sum_loss.backward()
    if max_grad is not None:
        nn.utils.clip_grad_norm_(network.parameters(), max_grad)
    optimizer.step()

'''
    def update_networks(optimizer, loss_dict, net_dict, max_gradient=None):
        # global iter_num
        sum_loss = 0
        for _, loss in loss_dict.items():
            sum_loss += loss
        optimizer.zero_grad()
        sum_loss.backward()
        if max_gradient is not None:
            # max_grad, mean_grad = 0, 0
            for _, net in net_dict.items():
                # for param in net.parameters():
                # max_grad = torch.max(torch.norm(param.grad))
                # mean_grad = torch.mean(torch.norm(param.grad))
                # print(iter_num, max_gradient, max_grad, mean_grad)
                nn.utils.clip_grad_norm_(net.parameters(), max_gradient)
            # iter_num += 1
        optimizer.step()
'''
