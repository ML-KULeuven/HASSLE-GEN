#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 16:36:56 2020

@author: mohit
"""
import numpy as np
from .local_search import learn_weighted_max_sat


def example1():
    # Example
    #      A \/ B
    # 1.0: A
    #
    # --:  A, B   sat  opt (1)
    # --:  A,!B   sat  opt (1)
    # --: !A, B   sat !opt (0)
    # --: !A,!B  !sat !opt (0)
    #
    #  A:  A, B   sat  opt (1)
    #  A:  A,!B   sat  opt (1)
    #
    # !A: !A, B   sat  opt (0)
    # !A: !A,!B  !sat  opt (0)
    #
    #  B:  A, B   sat  opt (1)
    #  B: !A, B   sat !opt (0)
    #
    # !B:  A,!B   sat  opt (1)
    # !B: !A,!B  !sat !opt (0)

    data = np.array(
        [
            [True, True],
            [True, False],
            [False, True],
            [False, False],
            [True, True],
            [True, False],
            [False, True],
            [False, False],
            [True, True],
            [False, True],
            [True, False],
            [False, False],
        ]
    )

    labels = np.array(
        [True, True, False, False, True, True, True, False, True, False, True, False]
    )
    contexts = [set(), set(), set(), set(), {1}, {1}, {-1}, {-1}, {2}, {2}, {-2}, {-2}]
    # learn_weighted_max_sat(2,2, data, labels, contexts, 12, 0.1, 5, 1)


def example2():
    # Example
    #      !A \/ !B \/ !C
    # 1.0: A
    # 0.5: B \/ !C
    #
    # pos  A, B,!C  A
    # neg  A,!B, C  A suboptimal
    # neg  A, B, C  A infeasible
    #
    # pos !A, B,!C !A
    # neg !A,!B, C !A suboptimal
    #
    # pos  A, B,!C  B
    # neg !A, B, C  B suboptimal
    # neg  A, B, C  B infeasible
    #
    # pos  A,!B,!C !B
    # neg !A,!B, C !B suboptimal
    #
    # pos  A,!B, C  C
    # neg !A, B, C  C suboptimal
    # neg  A, B, C  C infeasible
    #
    # pos  A, B,!C !C
    # neg !A,!B,!C !C suboptimal
    #
    # pos !A,!B,!C  !A,!B
    # pos  A,!B,!C  !B,!C
    # pos  !A,B,C  B,C

    data = np.array(
        [
            [True, True, False],
            [True, False, True],
            [True, True, True],
            [False, True, False],
            [False, False, True],
            [True, True, False],
            [False, True, True],
            [True, True, True],
            [True, False, False],
            [False, False, True],
            [True, False, True],
            [False, True, True],
            [True, True, True],
            [True, True, False],
            [False, False, False],
            [False, False, False],
            [True, False, False],
            [False, True, True],
        ]
    )

    labels = np.array(
        [
            True,
            False,
            False,
            True,
            False,
            True,
            False,
            False,
            True,
            False,
            True,
            False,
            False,
            True,
            False,
            True,
            True,
            True,
        ]
    )

    contexts = [
        {1},
        {1},
        {1},
        {-1},
        {-1},
        {2},
        {2},
        {2},
        {-2},
        {-2},
        {3},
        {3},
        {3},
        {-3},
        {-3},
        {-1, -2},
        {-2, -3},
        {2, 3},
    ]


#    a, b, c, scores, best_scores = learn_weighted_max_sat(
#        3, data, labels, contexts, "walk_sat", 18,cutoff_time=60
#    )
#    print(a)
#    plt.plot(range(len(best_scores)), best_scores, "r-", label="Walk_SAT")
#
#    a, b, c, scores, best_scores = learn_weighted_max_sat(
#        3, data, labels, contexts, "novelty", 18,cutoff_time=60
#    )
#    print(a)
#    plt.plot(range(len(best_scores)), best_scores, "g-", label="Novelty")
#
##    a, b, c, scores, best_scores = learn_weighted_max_sat(
##        3, data, labels, contexts, "novelty_plus", 18
##    )
##    print(a)
##    plt.plot(range(len(best_scores)), best_scores, "b-", label="Novelty+")
##
##    a, b, c, scores, best_scores = learn_weighted_max_sat(
##        3, data, labels, contexts, "adaptive_novelty_plus", 18
##    )
##    print(a)
##    plt.plot(range(len(best_scores)), best_scores, "y-", label="Adaptive_Novelty+")
#
#    plt.legend(loc="lower right")
#
#    plt.show()


if __name__ == "__main__":
    example2()
