"""
https://github.com/pytorch/fairseq/blob/master/examples/backtranslation/deduplicate_lines.py

行去重。这里采用的

简单的思路是：每行做md5。

## 加速
这里涉及的加速有
- md5计算快


## 疑问
md5碰撞会怎样？

"""

import argparse
import fileinput
import hashlib
from multiprocessing import Pool
import sys


def get_hashes_and_lines(raw_line):
    hash = hashlib.md5(raw_line).hexdigest()
    return hash, raw_line


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('files', nargs='*', help='input files')
    args = parser.parse_args()

    seen = set()
    with fileinput.input(args.files, mode='rb') as h:
        pool = Pool(args.workers)
        results = pool.imap_unordered(get_hashes_and_lines, h, 1000)
        for i, (hash, raw_line) in enumerate(results):
            if hash not in seen:
                seen.add(hash)
                sys.stdout.buffer.write(raw_line)
            if i % 1000000 == 0:
                print(i, file=sys.stderr, end="", flush=True)
            elif i % 100000 == 0:
                print(".", file=sys.stderr, end="", flush=True)
    print(file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()