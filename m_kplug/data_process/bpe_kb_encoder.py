"""
<风格>复古</风格>的旗袍款式

1. 先分词，再标签。

"""

import os
import re
from transformers import BertTokenizer
from collections import defaultdict, Counter

PATTEN_BIO = re.compile('<?.*>?')


def parse_tag_words(subwords):
    """ HC<领型>圆领</领型><风格>拼接</风格>连衣裙 """
    tags = []
    new_subwords = []
    i = 0
    entity = ''
    while i < len(subwords):
        if subwords[i] == '<':
            if entity == '':  # <领型>
                for j in range(i + 1, len(subwords)):
                    if subwords[j] == '>':
                        break
                    entity += subwords[j]
            else:  # </领型>
                for j in range(i + 1, len(subwords)):
                    if subwords[j] == '>':
                        break
                entity = ''
            i = j + 1
            continue

        if entity != '':  # 圆领
            for j in range(i, len(subwords)):
                if subwords[j] == '<':
                    i = j
                    break
                new_subwords.append(subwords[j])
                if j == i:
                    tags.append('B-' + entity)
                else:
                    tags.append('I-' + entity)
            continue

        tags.append('O')
        new_subwords.append(subwords[i])
        i = i + 1

    return new_subwords, tags


def bpe(part='train', export_dict=False):
    all_tags = []
    data_dir = '/workspace/fairseq/data/JD/kb_ner'
    bpe_dir = data_dir + '/bpe'
    bpe = BertTokenizer('/workspace/fairseq/data/vocab/vocab.jd.txt')  # never_split参数对tokenizer不起作用
    f_src = open(os.path.join(bpe_dir, part + '.src'), 'w', encoding='utf-8')
    f_tgt = open(os.path.join(bpe_dir, part + '.tgt'), 'w', encoding='utf-8')
    for line in open(data_dir + '/raw/jdai.jave.fashion.' + part, 'r', encoding='utf-8'):
        cid, sid, sent, tag_sent = line.strip().split('\t')
        subwords = bpe._tokenize(tag_sent)
        subwords, tags = parse_tag_words(subwords)
        f_src.write(' '.join(subwords) + '\n')
        f_tgt.write(' '.join(tags) + '\n')
        all_tags += tags

    if export_dict:
        with open(os.path.join(bpe_dir, 'vocab.tgt.txt'), 'w', encoding='utf-8') as f_out:
            f_out.write('\n'.join([k for k, cnt in Counter(all_tags).items()]) + '\n')


def find_all_entity():
    all_tag_sent = []
    for line in open('raw/jdai.jave.fashion.train', 'r', encoding='utf-8'):
        cid, sid, sent, tag_sent = line.strip().split('\t')
        all_tag_sent.append(tag_sent)
    entity_list = re.findall('<.*?>', ''.join(all_tag_sent))
    aa = Counter(entity_list)
    f_out = open('raw/vocab_bioattr.txt', 'w', encoding='utf-8')
    f_out.write(''.join(['{}\t{}\n'.format(k, cnt) for k, cnt in aa.items()]))


if __name__ == "__main__":
    bpe('train', export_dict=True)
    bpe('valid')
    bpe('test')
