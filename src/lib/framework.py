import abc
import torch
import torch.distributed


def get_framework_dict():
    return {
        'mmpe': MMPEFramework,
    }


class FrameworkABC(abc.ABC):
    def __init__(self, global_args, framework_args, network, post_proc, world_size):
        self.global_args = global_args
        self.framework_args = framework_args
        self.network = network
        self.post_proc = post_proc
        self.world_size = world_size

    def train_forward(self, data_dict):
        loss_dict, value_dict = self.forward(data_dict, train=True, grad_enable=True)
        return loss_dict, value_dict

    def infer_forward(self, data_dict):
        output_dict, result_dict = self.forward(data_dict, train=False, grad_enable=False)
        return output_dict, result_dict

    def valid_forward(self, data_dict):
        loss_dict, value_dict = self.forward(data_dict, train=True, grad_enable=False)
        return loss_dict, value_dict

    @ abc.abstractmethod
    def forward(self, data_dict, train=True, grad_enable=True):
        pass

    @ abc.abstractmethod
    def merge_batch_losses(self, loss_dict):
        pass

    @abc.abstractmethod
    def merge_batch_values(self, value_dict):
        pass

# import time

class MMPEFramework(FrameworkABC):
    def forward(self, data_dict, train=True, grad_enable=True):
        self.network.train(train)
        torch.autograd.set_grad_enabled(grad_enable)

        # t1 = time.time()
        image = data_dict['img'].requires_grad_(grad_enable).float().cuda()
        # heatmap = data_dict['heatmap'].requires_grad_(grad_enable).float().cuda()
        # image = torch.cat([image, heatmap], dim=1)

        if train:
            gt_dict = dict()
            # t2 = time.time()
            gt_dict['boxes'] = data_dict['boxes'].cuda()
            gt_dict['joints'] = data_dict['joints'].cuda()
            gt_dict['n_people'] = data_dict['n_people'].long().cuda()
            # t3 = time.time()
            loss_dict, value_dict = self.network.forward(image, gt_dict, loss=True)

            # t4 = time.time()
            loss_dict = self.merge_batch_losses(loss_dict)
            value_dict = self.merge_batch_values(value_dict)
            # t5 = time.time()
            # print(t2-t1)
            # print(t3-t2)
            # print(t4-t3)
            # print(t5-t4)
            # print('')
            return loss_dict, value_dict

        else:
            output_dict = self.network.forward(image, loss=False)
            result_dict, value_dict = self.post_proc.forward(output_dict)
            # value_dict = self.merge_batch_values(value_dict)
            return output_dict, result_dict

    def merge_batch_losses(self, loss_dict):
        batch_loss_dict = dict()
        for key, value in loss_dict.items():
            if len(value) > 0:
                if self.world_size > 1:
                    gather_num = [torch.ones(1).long().cuda() for _ in range(self.world_size)]
                    torch.distributed.all_gather(gather_num, torch.tensor(value.shape[0]).view(1).long().cuda())
                    batch_loss_dict[key] = \
                        self.world_size * torch.sum(value, dim=0) / torch.sum(torch.cat(gather_num, dim=0))
                else:
                    batch_loss_dict[key] = torch.sum(value, dim=0) / value.shape[0]
        return batch_loss_dict

    def merge_batch_values(self, value_dict):
        batch_value_dict = dict()
        for key, value in value_dict.items():
            if len(value) > 0:
                batch_value_dict[key] = torch.sum(value, dim=0) / value.shape[0]
            else:
                batch_value_dict[key] = None
        return batch_value_dict
