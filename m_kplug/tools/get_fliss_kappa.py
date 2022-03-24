#!/usr/bin/env python3

'''
Created on Aug 1, 2016
@author: skarumbaiah

Computes Fleiss' Kappa
Joseph L. Fleiss, Measuring Nominal Scale Agreement Among Many Raters, 1971.
'''
import sys


def checkInput(rate, n):
    """
    Check correctness of the input matrix
    @param rate - ratings matrix
    @return n - number of raters
    @throws AssertionError
    """
    N = len(rate)
    k = len(rate[0])
    assert all(len(rate[i]) == k for i in range(k)), "Row length != #categories)"
    assert all(isinstance(rate[i][j], int) for i in range(N) for j in range(k)), "Element not integer"
    assert all(sum(row) == n for row in rate), "Sum of ratings != #raters)"


def fleissKappa(rate, n):
    """
    Computes the Kappa value
    @param rate - ratings matrix containing number of ratings for each subject per category
    [size - N X k where N = #subjects and k = #categories]
    @param n - number of raters
    @return fleiss' kappa
    """

    N = len(rate)
    k = len(rate[0])
    print("#raters = ", n, ", #subjects = ", N, ", #categories = ", k)
    checkInput(rate, n)

    # mean of the extent to which raters agree for the ith subject
    PA = sum([(sum([i ** 2 for i in row]) - n) / (n * (n - 1)) for row in rate]) / N
    print("PA = ", PA)

    # mean of squares of proportion of all assignments which were to jth category
    PE = sum([j ** 2 for j in [sum([rows[i] for rows in rate]) / (N * n) for i in range(k)]])
    print("PE =", PE)

    kappa = -float("inf")
    try:
        kappa = (PA - PE) / (1 - PE)
        kappa = float("{:.3f}".format(kappa))
    except ZeroDivisionError:
        print("Expected agreement = 1")

    print("Fleiss' Kappa =", kappa)

    return kappa


def get_rates(in_file, metric_um=4, rate_num=3):
    info_rates = []
    fluency_rates = []
    for line in open(in_file):
        line_split = line.split('\t')
        cur_info_rate = [0] * metric_um
        cur_fluency_rate = [0] * metric_um
        for id in line_split[:rate_num]:
            cur_info_rate[int(id) - 1] += 1
        for id in line_split[rate_num:]:
            cur_fluency_rate[int(id) - 1] += 1
        info_rates.append(cur_info_rate)
        fluency_rates.append(cur_fluency_rate)
    return info_rates, fluency_rates


def test_fleiss():
    rate = \
        [
            [18, 7, 7, 18],
            [29, 6, 6, 9],
            [20, 7, 11, 12],
        ]
    kappa = fleissKappa(rate, 4)
    print(kappa)


def get_fleiss_res():
    info_rates, fluency_rates = get_rates(in_file=sys.argv[1])
    print(info_rates)
    kappa = fleissKappa(info_rates, 3)
    print(kappa)

    print(fluency_rates)
    kappa = fleissKappa(fluency_rates, 3)
    print(kappa)


if __name__ == '__main__':
    get_fleiss_res()
    # test_fleiss()
