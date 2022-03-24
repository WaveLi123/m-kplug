"""
input:
    txt & meta
output:
    bpe & meta_idx



## 小数据
--vocab-bpe ../../vocab/vocab.jd.txt --inputs small/discovery_all --outputs small/bpe/discovery_all --workers 1

python bpe_meta_encoder.py \
--vocab-bpe ../../vocab/vocab.jd.txt \
--inputs small/discovery_all \
--outputs small/bpe/discovery_all \
--workers 60

"""

import argparse
import contextlib
import sys
import os
import json

from collections import Counter
from multiprocessing import Pool

from transformers import BertTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocab-bpe",
        type=str,
        help='path to vocab.bpe',
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=['-'],
        help="input files to filter/encode",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=['-'],
        help="path to save encoded outputs",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep empty lines",
    )
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    assert len(args.inputs) == len(args.outputs), \
        "number of input and output paths should match"

    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input + postfix, "r", encoding="utf-8"))
            if input != "-" else sys.stdin
            for input in args.inputs for postfix in ['.txt', '.meta']
        ]
        outputs = [
            (stack.enter_context(open(output + '.bpe', "w", encoding="utf-8")),
             stack.enter_context(open(output + '.meta', "w", encoding="utf-8")))
            if output != "-" else sys.stdout
            for output in args.outputs
        ]

        encoder = MultiprocessingEncoder(args)
        pool = Pool(args.workers, initializer=encoder.initializer)
        encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 1000)

        stats = Counter()
        for i, (filt, enc_lines) in enumerate(encoded_lines, start=1):
            if filt == "PASS":
                for enc_line, output_h in zip(enc_lines, outputs):
                    print(enc_line[0], file=output_h[0])
                    print(enc_line[1], file=output_h[1])
            else:
                stats["num_filtered_" + filt] += 1
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)


def find_sub_list(sl, l):
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            results.append((ind, ind + sll))
    return results


class MultiprocessingEncoder(object):

    def __init__(self, args):
        self.args = args

    def initializer(self):
        global bpe, cate2id
        bpe = BertTokenizer(self.args.vocab_bpe)
        all_category = [line.strip() for line in open('raw/category')]
        cate2id = {cate: str(i) for i, cate in enumerate(all_category)}

    def encode(self, line):
        global bpe
        subword = bpe._tokenize(line)
        return subword

    def decode(self, tokens):
        global bpe
        return bpe.decode(tokens)

    def get_match_idx_from_tokens(self, tokens, metas):
        meta_tokens = set([meta for meta in metas.split()])
        meta_tokens = [self.encode(token) for token in meta_tokens]
        all_index = []
        for meta_token in meta_tokens:
            all_index += find_sub_list(meta_token, tokens)
        all_index = sorted(all_index, key=lambda kv: kv[0])
        return all_index

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        global cate2id

        enc_lines = []
        lines = [lines]
        for line, meta in lines:
            line = line.strip()
            category = meta.rstrip().split('\t')[0]
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None]
            tokens = self.encode(line)
            if False and '[UNK]' in tokens:
                print(''.join(tokens))
                print(line)
            meta_idx = self.get_match_idx_from_tokens(tokens, meta)
            cateid = cate2id.get(category, '<pad>')
            meta_idx_str = ' '.join([cateid] + [str(i) for k in meta_idx for i in k])
            enc_lines.append((" ".join(tokens), meta_idx_str))
        return ["PASS", enc_lines]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]


if __name__ == "__main__":
    main()
