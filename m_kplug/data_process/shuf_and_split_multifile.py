""" 多个大文件的shuffle
"""

import os
import sys
import time
import mmap
from random import shuffle
import numpy as np
from tqdm import tqdm


cur_path = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.dirname(cur_path)
sys.path.insert(0, base_path)

from raw_data_process.aaai_step2_img_to_vec import Img2Vec
from data_process.extract_patch_vector import get_img_patch_file

chunk_size = 1  # 每20行视为一个chunk. 1. 减少shuffle的压力 2.保证一个batch内的数据训练任务相同。

# data_dir = ''
# items = ['bpe', 'meta', 'ocr']  # 预训练任务


cate_name = sys.argv[1]
data_dir = '/data/liweikang/jdsum/{}/'.format(cate_name)
items = ['src', 'tgt', 'sku']  # 摘要任务
in_prefix = sys.argv[2]  # 'bpe'
out_prefix = sys.argv[3]  # 'bpe_batch'


def get_img_patch_files(sku_files):
    img2vec = Img2Vec(cuda=True, model='resnet-101', layer='third_last')
    for sku_file in sku_files:
        get_img_patch_file(sku_file, img2vec)
    sys.stdout.write("The Task Is Finished! All File Num: {}\n".format(len(sku_files)))


def find_lines(data):
    chunk_idx = 0
    for i, char in enumerate(data):
        if char == b'\n':
            chunk_idx += 1
        if chunk_idx >= chunk_size:
            chunk_idx = 0
            yield i


def shuffle_and_split():
    f_inputs = [open(data_dir + '/' + in_prefix + '/train.' + item) for item in items]
    all_mmap = [mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) for f in f_inputs]

    all_line_idx = []
    for item, item_mmap in zip(items, all_mmap):
        t1 = time.time()
        start = 0
        lines = []
        for end in find_lines(item_mmap):
            lines.append((start, end))
            start = end + 1
        t2 = time.time()
        print('load {} {}s'.format(item, int(t2 - t1)))
        all_line_idx.append(lines)

    all_line_idx = list(zip(*all_line_idx))
    shuffle(all_line_idx)
    # lines_per_file = int(1000000 / chunk_size)  # finetune任务，可以分块小一些
    lines_per_file = int(50000 / chunk_size)  # finetune任务，可以分块小一些

    sku_files = []
    for i, line_idx in enumerate(all_line_idx):
        if i % lines_per_file == 0:
            out_dir = os.path.join(data_dir, out_prefix, 'parts')
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            f_outs = [open(os.path.join(out_dir, 'data.{}.{}'.format(int(i / lines_per_file), item)), 'wb') for item in
                      items]
            sku_files.append(os.path.join(out_dir, 'data.{}.{}'.format(int(i / lines_per_file), items[-1])))

        for item_line_idx, item_mmap, f_out in zip(line_idx, all_mmap, f_outs):
            f_out.write(item_mmap[item_line_idx[0]:item_line_idx[1] + 1])

    with open(os.path.join(data_dir, out_prefix, "sku_files"), 'w') as f_out_sku:
        for line in sku_files:
            f_out_sku.write(line + '\n')
    return sku_files


if __name__ == "__main__":
    sku_files = shuffle_and_split()
    print(sku_files)
    get_img_patch_files(sku_files)

