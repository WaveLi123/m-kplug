#!/usr/bin/env python3

'''
Created on Aug 1, 2016
@author: skarumbaiah

Computes Fleiss' Kappa
Joseph L. Fleiss, Measuring Nominal Scale Agreement Among Many Raters, 1971.
'''
import sys
import numpy as np
from scipy.stats import ttest_rel


def get_rates(in_file, rate_num=3):
    info_rates = []
    fluency_rates = []
    for line in open(in_file):
        line_split = line.split('\t')
        cur_info_rate = []
        cur_fluency_rate = []
        for id in line_split[:rate_num]:
            cur_info_rate.append(int(id))
        for id in line_split[rate_num:]:
            cur_fluency_rate.append(int(id))
        info_rates.append(cur_info_rate)
        fluency_rates.append(cur_fluency_rate)
    return info_rates, fluency_rates


def test_ttest():
    a = np.random.randint(1, 5, size=100)
    b = np.random.randint(1, 5, size=100)
    print(a)
    print(b)
    print(ttest_rel(a, b))


def get_ttest_res():
    info_rates, fluency_rates = get_rates(in_file=sys.argv[1])
    print(info_rates)
    info_rates = np.array(info_rates)
    print(info_rates)
    print(info_rates[..., 0])
    print(info_rates[..., 1])
    print(info_rates[..., 2])
    print(ttest_rel(info_rates[..., 0], info_rates[..., 1]))
    print(ttest_rel(info_rates[..., 0], info_rates[..., 2]))
    print(ttest_rel(info_rates[..., 1], info_rates[..., 2]))


if __name__ == '__main__':
    test_ttest()
    get_ttest_res()
