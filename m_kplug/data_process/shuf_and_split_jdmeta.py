"""

input: txt, meta, label

融合ocr数据和短文数据，并

https://stackoverflow.com/questions/24492331/shuffle-a-large-list-of-items-without-loading-in-memory

shuffle时还要带着meta信息。
"""
import time
import mmap
from random import shuffle

# 5000000
chunk_size = 20  # 每20行视为一个chunk。1. 减少shuffle的压力 2.保证一个batch内的数据训练任务相同。

def find_lines(data):
    chunk_idx = 0
    for i, char in enumerate(data):
        if char == b'\n':
            chunk_idx += 1
        if chunk_idx >= chunk_size:
            chunk_idx = 0
            yield i

data_dir = '/workspace/fairseq/data/JD/discovery_meta/'
# data_dir = '/workspace/fairseq/data/JD/discovery_meta/small/'

def shuffle_and_split():
    f1 = open(data_dir + 'bpe/discovery_all.bpe')
    f2 = open(data_dir + 'bpe/discovery_all.meta')
    f3 = open(data_dir + 'bpe/discovery_all.ocr')
    data_txt = mmap.mmap(f1.fileno(), 0, access=mmap.ACCESS_READ)
    data_meta = mmap.mmap(f2.fileno(), 0, access=mmap.ACCESS_READ)
    data_ocr = mmap.mmap(f3.fileno(), 0, access=mmap.ACCESS_READ)

    t1 = time.time()
    start = 0
    lines_txt = []
    for end in find_lines(data_txt):
        lines_txt.append((start, end))
        start = end + 1
    t2 = time.time()
    print('load bpe {}s'.format(int(t2-t1)))

    start = 0
    lines_meta = []
    for end in find_lines(data_meta):
        lines_meta.append((start, end))
        start = end + 1
    t3 = time.time()
    print('load meta {}s'.format(int(t3 - t2)))

    start = 0
    lines_ocr = []
    for end in find_lines(data_ocr):
        lines_ocr.append((start, end))
        start = end + 1
    t4 = time.time()
    print('load ocr {}s'.format(int(t4 - t3)))

    lines = list(zip(lines_txt, lines_meta)) + lines_ocr

    shuffle(lines)

    lines_per_file = int(2000000/chunk_size)   # 1000000行， 大概一个文件

    null_meta = b'0\n' * chunk_size
    for i, (v1, v2) in enumerate(lines):
        if i % lines_per_file == 0:
            f_out_txt = open(data_dir + 'bpe/parts/data.{}.bpe'.format(int(i/lines_per_file)), 'wb')
            f_out_meta = open(data_dir + 'bpe/parts/data.{}.meta'.format(int(i/lines_per_file)), 'wb')
        if isinstance(v1, tuple):
            f_out_txt.write(data_txt[v1[0]:v1[1]+1])
            f_out_meta.write(data_meta[v2[0]:v2[1] + 1])
        else:
            f_out_txt.write(data_ocr[v1:v2+1])
            f_out_meta.write(null_meta)

if __name__ == "__main__":
    shuffle_and_split()




