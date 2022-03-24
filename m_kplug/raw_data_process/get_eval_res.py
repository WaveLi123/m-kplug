#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import random
import sys
import codecs

cur_path = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.dirname(cur_path)
sys.path.insert(0, base_path)


def get_uniq_hypo_res(in_file):
    hypo_res = dict()
    uniq_texts = set()

    with codecs.open(in_file, 'r') as f_r:
        eval_lines = f_r.readlines()  # quite limited lins
    random.shuffle(eval_lines)

    for line in eval_lines:
        line_split = str(line).strip().split()
        id = line_split[0]
        text = str("".join(line_split[1:])).strip().replace(",", "，")

        if text not in uniq_texts:
            uniq_texts.add(text)

            hypo_res[id] = text

    return hypo_res


def get_plain_res(in_file, id_tag=None):
    hypo_res = dict()

    with codecs.open(in_file, 'r') as f_r:
        eval_lines = f_r.readlines()  # quite limited lins
    for line in eval_lines:
        line_split = str(line).strip().split()
        id = line_split[0]
        text = str("".join(line_split[1:])).strip().replace(",", "，")
        if id_tag:
            hypo_res["-".join([id_tag, id.split('-')[-1]])] = text
        else:
            hypo_res[id] = text

    return hypo_res


def alignment_model_res(uniq_hypos, k_hypos, t_hypos, id_srcs, id_tgts, out_file):
    with codecs.open(out_file, 'w', encoding='utf-8') as f_w:
        f_w.write("ID,Src,Tgt,MVPNet,K-Plug,TextualCopy\n")
        for id, uniq_text in uniq_hypos.items():
            src = id_srcs[id]
            tgt = id_tgts[id]
            k_text = k_hypos[id]
            t_text = t_hypos[id]

            f_w.write(",".join([id, src, tgt, uniq_text, k_text, t_text]) + '\n')

    sys.stdout.write("\nEval Res has been Dumped.\n")


def extract_eval_res(uniq_file, k_file, t_file, src_file, tgt_file, out_file):
    alignment_model_res(uniq_hypos=get_uniq_hypo_res(uniq_file),
                        k_hypos=get_plain_res(k_file),
                        t_hypos=get_plain_res(t_file),
                        id_srcs=get_plain_res(src_file, id_tag="H"),
                        id_tgts=get_plain_res(tgt_file, id_tag="H"),
                        out_file=out_file,
                        )


if __name__ == '__main__':
    """
    Cases&Bags: python get_eval_res.py /data/liweikang/jdsum/cases_bags/res/report_out/uniq_hypo.txt /data/liweikang/jdsum/cases_bags/res/report_out/k_hypo.txt /data/liweikang/jdsum/cases_bags/res/report_out/t_hypo.txt /data/liweikang/jdsum/cases_bags/res/report_out/uniq_src.txt /data/liweikang/jdsum/cases_bags/res/report_out/uniq_tgt.txt /data/liweikang/jdsum/cases_bags/res/report_out/cases_bags.h_eval
    Home Appliances: python get_eval_res.py /data/liweikang/jdsum/home_appliances/res/report_out/uniq_hypo.txt /data/liweikang/jdsum/home_appliances/res/report_out/k_hypo.txt /data/liweikang/jdsum/home_appliances/res/report_out/t_hypo.txt /data/liweikang/jdsum/home_appliances/res/report_out/uniq_src.txt /data/liweikang/jdsum/home_appliances/res/report_out/uniq_tgt.txt /data/liweikang/jdsum/home_appliances/res/report_out/home_appliances.h_eval
    """
    extract_eval_res(uniq_file=sys.argv[1], k_file=sys.argv[2], t_file=sys.argv[3], src_file=sys.argv[4],
                     tgt_file=sys.argv[5], out_file=sys.argv[6])
