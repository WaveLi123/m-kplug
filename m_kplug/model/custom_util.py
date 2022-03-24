#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import os
import psutil
import torch.nn as nn

import time
from functools import wraps


def show_memory_info(hint):
    pid = os.getpid()
    p = psutil.Process(pid)

    info = p.memory_full_info()
    memory = info.uss / 1024. / 1024. / 1024.
    print('{} memory used: {} GB, pid: {}'.format(hint, memory, pid))


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Running [%s] consumes %.3f seconds" %
              (function.__name__, (t1 - t0))
              # (function.func_name, str(t1 - t0)) #for python2.x
              )
        return result

    return function_timer


def get_masked_softmax(tensor, mask):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, *, sequence_length).
    Args:
        tensor: The tensor on which the softmax function must be applied along
            the last dimension.
        mask: A mask of the same size as the tensor with 0s in the positions of
            the values that must be masked and 1s everywhere else.
    Returns:
        A tensor of the same size as the inputs containing the result of the
        softmax.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])

    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)

    return result.view(*tensor_shape)


if __name__ == '__main__':
    bs = 2
    hidden_size = 50
    src_len = 6
    tgt_len = 5

    attn_weight = torch.randn((bs, tgt_len, src_len))
    a = torch.ones((bs, src_len))
    a[0][-2:] = 0
    a[-1][-3:] = 0
    src_encoding_mask = a.type(torch.bool)

    print(attn_weight)
    masked_attn_weight = get_masked_softmax(attn_weight, src_encoding_mask)
    print(masked_attn_weight)
