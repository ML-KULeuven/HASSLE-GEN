# noinspection PyUnresolvedReferences
import logging
from typing import List

import numpy as np

from .type_def import MaxSatModel, Clause, suppress_stdout
from gurobipy import Model, GRB, quicksum


logger = logging.getLogger(__name__)


def learn_weighted_max_sat_MILP(
    m: int, data: np.ndarray, labels: np.ndarray, contexts: List[Clause], cutoff
) -> MaxSatModel:
    """
    Learn a weighted MaxSAT model from examples. Contexts and clauses are set-encoded, i.e., they are represented by
    sets containing positive or negative integers in the range -n-1 to n+1. If a set contains an positive integer i, the i-1th
     Boolean feature is set to True, if it contains a negative integer -i, the i-1th Boolean feature is set to False.
    :param m:
        The number of clauses in the MaxSAT model to learn
    :param data:
        A Boolean s x n (number examples x number Boolean variables) numpy array in which every row is an example and
        every column a Boolean feature (True = 1 or False = 0)
    :param labels:
        A Boolean numpy array (length s) where the kth entry contains the label (1 or 0) of the kth example
    :param contexts:
        A list of s set-encoded contexts.
    :return:
        A list of weights and clauses. Every entry of the list is a tuple containing as first element None for hard
        constraints (clauses) or a floating point number for soft constraints, and as second element a set-encoded clause.
    """

    w_max_value = 1
    s = data.shape[0]
    n = data.shape[1]
    big_m = 2 * (m + 1)
    epsilon = 10 ** (-2)

    context_pool = dict()
    unique_contexts = []
    context_indices = []
    for context in contexts:
        key = frozenset(context)
        if key not in context_pool:
            context_pool[key] = len(context_pool)
            unique_contexts.append(context)
        context_indices.append(context_pool[key])
    context_counts = len(context_pool)

    contexts_with_positive = set()
    for k in range(s):
        if labels[k]:
            contexts_with_positive.add(context_indices[k])
    skip_positive_contexts = True

    logger.debug("Learn wMaxSAT")
    logger.debug(f"w_max: {w_max_value}")
    logger.debug(f"s: {s}")
    logger.debug(f"n: {n}")
    logger.debug(f"m: {m}")

    with suppress_stdout():
        mod = Model("LearnMaxSat")

    mod.setParam("OutputFlag", False)

    # Constraint decision variables
    c_j = [mod.addVar(vtype=GRB.BINARY, name=f"c_{j})") for j in range(m)]
    a_jl = [
        [mod.addVar(vtype=GRB.BINARY, name=f"a_[{j}, {l}]") for l in range(2 * n)]
        for j in range(m)
    ]

    # Weights decision variables
    w_j = [
        mod.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=w_max_value, name=f"w_{j})")
        for j in range(m)
    ]
    wz_j = [mod.addVar(vtype=GRB.BINARY, name=f"wz_{j})") for j in range(m)]

    # Auxiliary decision variabnles
    gamma_context = [
        mod.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=m + 1, name=f"gamma_{context})")
        for context in range(context_counts)
    ]

    # Coverage
    cov_jk = [
        [mod.addVar(vtype=GRB.BINARY, name=f"cov_[{j}, {k}])") for k in range(s)]
        for j in range(m)
    ]
    covp_jk = [
        [mod.addVar(vtype=GRB.BINARY, name=f"covp_[{j}, {k}])") for k in range(s)]
        for j in range(m)
    ]
    cov_k = [mod.addVar(vtype=GRB.BINARY, name=f"cov_[{k}])") for k in range(s)]

    # Values
    opt_k = [mod.addVar(vtype=GRB.BINARY, name=f"opt_[{k}])") for k in range(s)]
    w_jk = [
        [
            mod.addVar(
                vtype=GRB.CONTINUOUS, lb=0, ub=w_max_value, name=f"w_[{j}, {k}])"
            )
            for k in range(s)
        ]
        for j in range(m)
    ]

    # Extra variables
    cov_o = [
        mod.addVar(vtype=GRB.BINARY, name=f"cov_{o}") for o in range(context_counts)
    ]
    ccov_o = [
        mod.addVar(vtype=GRB.BINARY, name=f"ccov_{o}") for o in range(context_counts)
    ]
    cov_oj = [
        [mod.addVar(vtype=GRB.BINARY, name=f"cov_[{o}, {j}]") for j in range(m)]
        for o in range(context_counts)
    ]
    covp_oj = [
        [mod.addVar(vtype=GRB.BINARY, name=f"covp_[{o}, {j}]") for j in range(m)]
        for o in range(context_counts)
    ]
    cov_ojl = [
        [
            [
                mod.addVar(vtype=GRB.BINARY, name=f"cov_[{o}, {j}, {l}]")
                for l in range(2 * n)
            ]
            for j in range(m)
        ]
        for o in range(context_counts)
    ]
    xp_ol = [
        [mod.addVar(vtype=GRB.BINARY, name=f"xp_[{o},{l}]") for l in range(n)]
        for o in range(context_counts)
    ]
    w_oj = [
        [
            mod.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"w_[{o}, {j}]")
            for j in range(m)
        ]
        for o in range(context_counts)
    ]

    mod.setObjective(
        quicksum([gamma_context[context] for context in range(context_counts)]),
        GRB.MAXIMIZE,
    )

    # Constraints on x_prime
    for o in range(context_counts):
        if not (skip_positive_contexts and o in contexts_with_positive):
            context = unique_contexts[o]
            for index in context:
                if index < 0:
                    mod.addConstr(ccov_o[o] <= (1 - xp_ol[o][abs(index) - 1]))
                else:
                    mod.addConstr(ccov_o[o] <= xp_ol[o][index - 1])

            mod.addConstr(
                ccov_o[o]
                >= quicksum(
                    xp_ol[o][index - 1] if index > 0 else (1 - xp_ol[o][abs(index) - 1])
                    for index in context
                )
                - len(context)
                + 1
            )

    for o in range(context_counts):
        if not (skip_positive_contexts and o in contexts_with_positive):
            for j in range(m):
                for l in range(2 * n):
                    mod.addConstr(cov_ojl[o][j][l] <= a_jl[j][l])
                    if l < n:
                        x_covered = xp_ol[o][l]
                    else:
                        x_covered = 1 - xp_ol[o][l - n]
                    mod.addConstr(cov_ojl[o][j][l] <= x_covered)
                    mod.addConstr(cov_ojl[o][j][l] >= a_jl[j][l] + x_covered - 1)

    for o in range(context_counts):
        if not (skip_positive_contexts and o in contexts_with_positive):
            for j in range(m):
                for l in range(2 * n):
                    mod.addConstr(cov_oj[o][j] >= cov_ojl[o][j][l])
                mod.addConstr(
                    cov_oj[o][j] <= quicksum(cov_ojl[o][j][l] for l in range(2 * n))
                )

    for o in range(context_counts):
        if not (skip_positive_contexts and o in contexts_with_positive):
            for j in range(m):
                mod.addConstr(covp_oj[o][j] >= cov_oj[o][j])
                mod.addConstr(covp_oj[o][j] >= (1 - c_j[j]))
                mod.addConstr(covp_oj[o][j] <= cov_oj[o][j] + (1 - c_j[j]))

    for o in range(context_counts):
        if not (skip_positive_contexts and o in contexts_with_positive):
            mod.addConstr(cov_o[o] <= ccov_o[o])
            for j in range(m):
                mod.addConstr(cov_o[o] <= covp_oj[o][j])
            mod.addConstr(
                cov_o[o] >= quicksum(covp_oj[o][j] for j in range(m)) + ccov_o[o] - m
            )

    for o in range(context_counts):
        if not (skip_positive_contexts and o in contexts_with_positive):
            for j in range(m):
                mod.addConstr(w_oj[o][j] <= cov_oj[o][j])
                mod.addConstr(w_oj[o][j] <= (1 - c_j[j]))
                mod.addConstr(w_oj[o][j] <= w_j[j] + (1 - cov_oj[o][j]) + c_j[j])
                mod.addConstr(w_oj[o][j] >= w_j[j] - (1 - cov_oj[o][j]) - c_j[j])

    for o in range(context_counts):
        if not (skip_positive_contexts and o in contexts_with_positive):
            mod.addConstr(gamma_context[o] <= quicksum([w_oj[o][j] for j in range(m)]))
            mod.addConstr(
                gamma_context[o] <= cov_o[o] * big_m
            )  # TODO Consider whether we need it? What about opt_k in infeasible contexts?

        # mod.addConstr(
        #     gamma_context[o]
        #     >= quicksum([w_oj[o][j] for j in range(m)]) + epsilon - big_m * opt_o[o]
        # )

    # Positive / negative constraints with contexts
    for k in range(s):
        if not (
            skip_positive_contexts and context_indices[k] in contexts_with_positive
        ):
            mod.addConstr(cov_k[k] <= cov_o[context_indices[k]])

            if labels[k]:
                mod.addConstr(cov_o[context_indices[k]] + cov_k[k] + opt_k[k] == 3)
            else:
                mod.addConstr(cov_o[context_indices[k]] + cov_k[k] + opt_k[k] <= 2)
        else:
            if labels[k]:
                mod.addConstr(cov_k[k] + opt_k[k] == 2)
            else:
                mod.addConstr(cov_k[k] + opt_k[k] <= 1)

    # Constraints for weights
    for j in range(m):
        mod.addConstr(
            w_j[j] <= (1 - wz_j[j])
        )  # TODO What about setting w > 2epsilon for soft constraints?
        mod.addConstr(w_j[j] >= 3 * epsilon - wz_j[j])

    for j in range(m):
        for k in range(s):
            mod.addConstr(
                w_jk[j][k] <= big_m * cov_jk[j][k],
                name=f"w_[{j}, {k}] <= M * cov_[{j}, {k}]",
            )

    for j in range(m):
        for k in range(s):
            mod.addConstr(
                w_jk[j][k] <= big_m * (1 - c_j[j]),
                name=f"w_[{j}, {k}] <= M * (1 - c_{j})",
            )

    for j in range(m):
        for k in range(s):
            mod.addConstr(
                w_jk[j][k] <= w_j[j] + big_m * (1 - cov_jk[j][k]) + big_m * c_j[j],
                name=f"w_[{j}, {k}] <= w_{j} + M * (1 - cov_[{j}, {k}]) + M * c_{j}",
            )

    for j in range(m):
        for k in range(s):
            mod.addConstr(
                w_jk[j][k] >= w_j[j] - big_m * (1 - cov_jk[j][k]) - big_m * c_j[j],
                name=f"w_[{j}, {k}] >= w_{j} - M * (1 - cov_[{j}, {k}]) - M * c_{j}",
            )

    # Constraints for gamma
    for k in range(s):
        mod.addConstr(
            gamma_context[context_indices[k]]
            <= quicksum([w_jk[j][k] for j in range(m)]) + big_m * (1 - opt_k[k]),
            name=f"gamma_{context_indices[k]} <= SUM_j w_[j, {k}] + M * (1 - opt_{k})",
        )

    for k in range(s):
        mod.addConstr(
            gamma_context[context_indices[k]]
            >= quicksum([w_jk[j][k] for j in range(m)]) + epsilon - big_m * opt_k[k],
            name=f"gamma_{context_indices[k]} >= SUM_j w_[j, {k}] + epsilon - M * opt_{k}",
        )

    # Constraints for coverage
    for j in range(m):
        for k in range(s):
            mod.addConstr(
                covp_jk[j][k] >= 1 - c_j[j], name=f"covp_[{j}, {k}] >= 1 - c_{j}"
            )

    for j in range(m):
        for k in range(s):
            mod.addConstr(
                covp_jk[j][k] >= cov_jk[j][k], name=f"covp_[{j}, {k}] >= cov_[{j}, {k}]"
            )

    for j in range(m):
        for k in range(s):
            mod.addConstr(
                covp_jk[j][k] <= cov_jk[j][k] + (1 - c_j[j]),
                name=f"covp_[{j}, {k}] <= cov_[{j}, {k}] + (1 - c_{j})",
            )

    def x_kl(k, l):
        if l < n:
            return 1 if data[k][l] else 0
        else:
            return 0 if data[k][l - n] else 1

    for j in range(m):
        for k in range(s):
            for l in range(2 * n):
                mod.addConstr(
                    cov_jk[j][k] >= a_jl[j][l] * x_kl(k, l),
                    name=f"cov_[{j}, {k}] >= a_[{j}, {l}] * {x_kl(k, l)}",
                )

    for j in range(m):
        for k in range(s):
            mod.addConstr(
                cov_jk[j][k]
                <= quicksum([a_jl[j][l] * x_kl(k, l) for l in range(2 * n)]),
                name=f"cov_[{j}, {k}] <= SUM_l a_[{j}, l] * x_[{k}, l)",
            )

    for j in range(m):
        for l in range(n):
            mod.addConstr(
                a_jl[j][l] + a_jl[j][n + l] <= 1,
                name=f"a_[{j}, {l}] + a_[{j}, {l + n}] <= 1",
            )

    for j in range(m):
        for k in range(s):
            mod.addConstr(cov_k[k] <= covp_jk[j][k], name=f"cov_{k} <= covp_[{j}, {k}]")

    for k in range(s):
        mod.addConstr(
            cov_k[k] >= quicksum([covp_jk[j][k] for j in range(m)]) - (m - 1),
            name=f"cov_{k} >= SUM_j covp_[j, {k}] - (m - 1)",
        )

    # mod.addConstr(c_j[0] >= 1, name="Forcing the first constraint to be hard")

    # Positive / negative constraints
    # for k in range(s):
    #     if labels[k]:
    #         mod.addConstr(cov_k[k] + opt_k[k] >= 2, name=f"cov_{k} + opt_{k} >= 2")
    #     else:
    #         mod.addConstr(cov_k[k] + opt_k[k] <= 1, name=f"cov_{k} + opt_{k} <= 1")
    if cutoff > 0:
        mod.Params.timeLimit = cutoff

    mod.optimize()

    if mod.status == GRB.Status.OPTIMAL:

        def char(_i):
            return (" " if _i < n else "!") + "abcdefghijklmnopqrstuvwxyz"[
                abs(_i % n)
            ].capitalize()

        def char_feature(_i, val):
            return (" " if val else "!") + "abcdefghijklmnopqrstuvwxyz"[
                abs(_i)
            ].capitalize()

        logger.info("Learning results")
        for k in range(s):
            logger.info(
                " : ".join(
                    [
                        (" sat" if cov_k[k].x else "!sat"),
                        (" opt" if opt_k[k].x else "!opt"),
                        " + ".join([f"{w_jk[j][k].x}" for j in range(m)])
                        + f" = {sum(w_jk[j][k].x for j in range(m))}",
                        f"gamma_{context_indices[k]} {gamma_context[context_indices[k]].x}",
                        ("pos " if labels[k] else "neg ")
                        + ",".join(char_feature(i, data[k][i]) for i in range(n)),
                        ", ".join(
                            (" sat'" if covp_jk[j][k].x else "!sat'") for j in range(m)
                        ),
                    ]
                )
            )

        for o in range(context_counts):
            logger.info(f"Context {o}: {unique_contexts[o]}")
            logger.info(
                "  ".join(
                    [f"xpo[{l}] {xp_ol[o][l].x}" for l in range(n)]
                    + [f"gamma {gamma_context[o].x}"]
                    + [f"wo[{j}] {w_oj[o][j].x} ({wz_j[j].x})" for j in range(m)]
                )
            )

        return [
            (
                None if c_j[j].x else w_j[j].x,
                {
                    l + 1 if l < n else -(l - n + 1)
                    for l in range(2 * n)
                    if a_jl[j][l].x
                },
            )
            for j in range(m)
        ]
    else:
        pass


#
# def label_instance(model: MaxSatModel, instance: Instance, context: Context) -> bool:
#    value = get_value(model, instance)
#    if value is None:
#        return False
#    best_instance = solve_weighted_max_sat(len(instance), model, context, 1)
#    best_value = get_value(model, best_instance)
#    logger.debug(f"Best instance: {best_value} - {best_instance}")
#    return value >= best_value
