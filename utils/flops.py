
import os
import sys
import time
import math

import operator
from functools import reduce
import torch.nn as nn
import torch
import torch.nn.init as init


def cal_param_size(model):
    return sum([i.numel() for i in model.parameters()])


count_ops = 0
def measure_layer(layer, x, multi_add=1):
    delta_ops = 0
    type_name = str(layer)[:str(layer).find('(')].strip()

    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) //
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) //
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w // layer.groups * multi_add

    ### ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = 0
        delta_ops = weight_ops + bias_ops

    global count_ops
    count_ops += delta_ops
    return


def is_leaf(module):
    return sum(1 for x in module.children()) == 0


def should_measure(module):
    if is_leaf(module):
        return True
    return False


def cal_multi_adds(model, shape=(2,3,32,32)):
    global count_ops
    count_ops = 0
    data = torch.zeros(shape).cuda()

    def new_forward(m):
        def lambda_forward(x):
            measure_layer(m, x)
            return m.old_forward(x)
        return lambda_forward

    def modify_forward(model):
        for child in model.children():
            if should_measure(child):
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    model.forward(data)
    restore_forward(model)

    return count_ops


# ---------------------------------------------------------------------
# Implementation of Learning Rate Scheduler
# ---------------------------------------------------------------------

class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    """ Polynomial decay learning rate scheduler.
    """

    def __init__(self, optimizer, epochs, iters_per_epoch, power=0.9, last_epoch=-1):
        self.epochs = epochs
        self.iters_per_epoch = iters_per_epoch
        self.max_iters = 40000 #self.epochs * self.iters_per_epoch
        self.cur_iter = 0
        self.power = power
        self.is_warn = False
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * ((1 - float(self.cur_iter) / self.max_iters) ** self.power) 
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is not None and epoch != 0:
            # update lr after each epoch if epoch is given
            # after each epoch, set epoch += 1 and call this function 
            if not self.is_warn:
                # logger.log_warn('PolynomialLR is designed for updating learning rate after each iteration.\n'
                #                 'However, it will be updated after each epoch now, please be careful.\n')
                print("PolynomialLR is designed for updating learning rate after each iteration. However, it will be updated after each epoch now, please be careful.\n")
                self.is_warn = True

            self.last_epoch = epoch
            assert self.last_epoch <= self.epochs
            self.cur_iter = self.last_epoch * self.iters_per_epoch

        elif epoch is None:
            # update lr after each iteration if epoch is None
            self.cur_iter += 1
            self.last_epoch = math.floor(self.cur_iter / self.iters_per_epoch)

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr




