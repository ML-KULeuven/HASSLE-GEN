#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 10:44:15 2020

@author: mohit
"""
from .pysat_solver import solve_weighted_max_sat
import numpy as np


def hc_sat_ex(maxsat, instance, context, rng):
    lst = []
    for i, (w, clause) in enumerate(maxsat.maxSatModel()):
        covered = any(
            not instance[abs(i) - 1] if i < 0 else instance[i - 1] for i in clause
        )
        if w is None and covered:
            lst.append(i)
    if not lst:
        return -1
    return rng.choice(lst)


def hc_not_sat_ex(maxsat, instance, context, rng):
    lst = []
    for i, (w, clause) in enumerate(maxsat.maxSatModel()):
        covered = any(
            not instance[abs(i) - 1] if i < 0 else instance[i - 1] for i in clause
        )
        if w is None and not covered:
            lst.append(i)
    if not lst:
        return -1
    return rng.choice(lst)


def sc_sat_ex_not_opt(model, instance, context, rng, conjunctive_contexts=0):
    opt, cost = solve_weighted_max_sat(model.n, model.maxSatModel(), context, 1, conjunctive_contexts=conjunctive_contexts)
    lst = []
    for i, (w, clause) in enumerate(model.maxSatModel()):
        ex_covered = any(
            not instance[abs(i) - 1] if i < 0 else instance[i - 1] for i in clause
        )
        opt_covered = any(not opt[abs(i) - 1] if i < 0 else opt[i - 1] for i in clause)
        if w is not None and ex_covered and not opt_covered:
            lst.append(i)
    if not lst:
        return -1
    return rng.choice(lst)


def sc_sat_opt_not_ex(model, instance, context, rng, conjunctive_contexts=0):
    opt, cost = solve_weighted_max_sat(model.n, model.maxSatModel(), context, 1, conjunctive_contexts=conjunctive_contexts)
    lst = []
    for i, (w, clause) in enumerate(model.maxSatModel()):
        ex_covered = any(
            not instance[abs(i) - 1] if i < 0 else instance[i - 1] for i in clause
        )
        opt_covered = any(not opt[abs(i) - 1] if i < 0 else opt[i - 1] for i in clause)
        if w is not None and opt_covered and not ex_covered:
            lst.append(i)
    if not lst:
        return -1
    return rng.choice(lst)


def sc_not_sat_any(model, instance, context, rng, conjunctive_contexts=0):
    opt, cost = solve_weighted_max_sat(model.n, model.maxSatModel(), context, 1, conjunctive_contexts=conjunctive_contexts)
    lst = []
    for i, (w, clause) in enumerate(model.maxSatModel()):
        ex_covered = any(
            not instance[abs(i) - 1] if i < 0 else instance[i - 1] for i in clause
        )
        opt_covered = any(not opt[abs(i) - 1] if i < 0 else opt[i - 1] for i in clause)
        if w is not None and not opt_covered and not ex_covered:
            lst.append(i)
    if not lst:
        return -1
    return rng.choice(lst)


def sc_sat_both(model, instance, context, rng, conjunctive_contexts=0):
    opt, cost = solve_weighted_max_sat(model.n, model.maxSatModel(), context, 1, conjunctive_contexts=conjunctive_contexts)
    lst = []
    for i, (w, clause) in enumerate(model.maxSatModel()):
        ex_covered = any(
            not instance[abs(i) - 1] if i < 0 else instance[i - 1] for i in clause
        )
        opt_covered = any(not opt[abs(i) - 1] if i < 0 else opt[i - 1] for i in clause)
        if w is not None and opt_covered and ex_covered:
            lst.append(i)
    if not lst:
        return -1
    return rng.choice(lst)


def sc_not_sat_ex(model, instance, context, rng):
    lst = []
    for i, (w, clause) in enumerate(model.maxSatModel()):
        ex_covered = any(
            not instance[abs(i) - 1] if i < 0 else instance[i - 1] for i in clause
        )
        if w is not None and not ex_covered:
            lst.append(i)
    if not lst:
        return -1
    return rng.choice(lst)


def sc_sat_ex(model, instance, context, rng):
    lst = []
    for i, (w, clause) in enumerate(model.maxSatModel()):
        ex_covered = any(
            not instance[abs(i) - 1] if i < 0 else instance[i - 1] for i in clause
        )
        if w is not None and ex_covered:
            lst.append(i)
    if not lst:
        return -1
    return rng.choice(lst)


def remove_literal(model, clause_index, literals):
    neighbours = []
    for literal in literals:
        j = abs(literal) - 1
        if model.l[clause_index][j] == np.sign(literal):
            neighbour = model.deep_copy()
            neighbour.l[clause_index][j] = 0
            if any(neighbour.l[clause_index]):
                neighbours.append(neighbour)
    return neighbours


def add_literal(model, clause_index, literals, clause_len):
    neighbours = []
    for literal in literals:
        j = abs(literal) - 1
        if model.l[clause_index][j] == 0:
            neighbour = model.deep_copy()
            neighbour.l[clause_index][j] = int(np.sign(literal))
            if (
                any(neighbour.l[clause_index])
                and num_literals_in_clause(neighbour.l[clause_index]) <= clause_len
            ):
                neighbours.append(neighbour)
    return neighbours


def instance_to_literals(instance):
    literals = set()
    for i, elem in enumerate(instance):
        if elem:
            literals.add(i + 1)
        else:
            literals.add(-i - 1)
    return literals


def num_literals_in_clause(clause):
    return sum(map(abs, clause))


def neighbours_inf(model, instance, context, clause_len, rng):
    i = hc_not_sat_ex(model, instance, context, rng)
    neighbours = []
    neighbour = model.deep_copy()
    neighbour.c[i] = 1 - neighbour.c[i]
    neighbours.append(neighbour)
    for j in range(model.n):
        values = [-1, 0, 1]
        values.remove(model.l[i][j])
        for val in values:
            neighbour = model.deep_copy()
            neighbour.l[i][j] = val
            if (
                any(neighbour.l[i])
                and num_literals_in_clause(neighbour.l[i]) <= clause_len
            ):
                neighbours.append(neighbour)

    return neighbours


def neighbours_sub(model, instance, context, clause_len, rng, w, conjunctive_contexts=0):
    sol, cost = solve_weighted_max_sat(model.n, model.maxSatModel(), context, 1, conjunctive_contexts=conjunctive_contexts)
    opt_literals = instance_to_literals(sol)
    exp_literals = instance_to_literals(instance)
    neighbours = []

    index = hc_sat_ex(model, instance, context, rng)
    if index >= 0:
        neighbours.extend(remove_literal(model, index, opt_literals))

    index = sc_sat_ex_not_opt(model, instance, context, rng, conjunctive_contexts=conjunctive_contexts)
    if index >= 0:
        neighbour = model.deep_copy()
        neighbour.c[index] = 1 - neighbour.c[index]
        neighbours.append(neighbour)

        if w == 1:
            tmp_w = model.w[index]
            if tmp_w < 0.999 and model.c[index] == 0:
                neighbour = model.deep_copy()
                neighbour.w[index] = (tmp_w + 1) / 2
                neighbours.append(neighbour)

    index = sc_sat_opt_not_ex(model, instance, context, rng, conjunctive_contexts=conjunctive_contexts)
    if index >= 0:
        neighbours.extend(remove_literal(model, index, opt_literals))
        neighbours.extend(add_literal(model, index, exp_literals, clause_len))

        if w == 1:
            tmp_w = model.w[index]
            if tmp_w > 0.001 and model.c[index] == 0:
                neighbour = model.deep_copy()
                neighbour.w[index] = tmp_w / 2
                neighbours.append(neighbour)

    index = sc_not_sat_any(model, instance, context, rng, conjunctive_contexts=conjunctive_contexts)
    if index >= 0:
        neighbours.extend(
            add_literal(model, index, exp_literals - opt_literals, clause_len)
        )

    index = sc_sat_both(model, instance, context, rng, conjunctive_contexts=conjunctive_contexts)
    if index >= 0:
        neighbours.extend(remove_literal(model, index, opt_literals - exp_literals))

    return neighbours


def neighbours_pos(model, instance, context, rng, w):
    exp_literals = instance_to_literals(instance)

    neighbours = []

    index = hc_sat_ex(model, instance, context, rng)
    if index >= 0:
        neighbours.extend(remove_literal(model, index, exp_literals))

    index = sc_not_sat_ex(model, instance, context, rng)
    if index >= 0:
        neighbour = model.deep_copy()
        neighbour.c[index] = 1 - neighbour.c[index]
        neighbours.append(neighbour)

        if w == 1:
            tmp_w = model.w[index]
            if tmp_w < 0.999 and model.c[index] == 0:
                neighbour = model.deep_copy()
                neighbour.w[index] = (tmp_w + 1) / 2
                neighbours.append(neighbour)

    index = sc_sat_ex(model, instance, context, rng)
    if index >= 0:
        neighbours.extend(remove_literal(model, index, exp_literals))

        if w == 1:
            tmp_w = model.w[index]
            if tmp_w > 0.001 and model.c[index] == 0:
                neighbour = model.deep_copy()
                neighbour.w[index] = tmp_w / 2
                neighbours.append(neighbour)

    return neighbours


def neighbours_pos_inf(model, instance, context, rng, w):
    exp_literals = instance_to_literals(instance)

    neighbours = []

    index = hc_sat_ex(model, instance, context, rng)
    if index >= 0:
        neighbours.extend(remove_literal(model, index, exp_literals))

    return neighbours


def neighbours_pos_sub(model, instance, context, rng, w):
    exp_literals = instance_to_literals(instance)

    neighbours = []

    index = sc_not_sat_ex(model, instance, context, rng)
    if index >= 0:
        neighbour = model.deep_copy()
        neighbour.c[index] = 1 - neighbour.c[index]
        neighbours.append(neighbour)

        if w == 1:
            tmp_w = model.w[index]
            if tmp_w < 0.999 and model.c[index] == 0:
                neighbour = model.deep_copy()
                neighbour.w[index] = (tmp_w + 1) / 2
                neighbours.append(neighbour)

    index = sc_sat_ex(model, instance, context, rng)
    if index >= 0:
        neighbours.extend(remove_literal(model, index, exp_literals))

        if w == 1:
            tmp_w = model.w[index]
            if tmp_w > 0.001 and model.c[index] == 0:
                neighbour = model.deep_copy()
                neighbour.w[index] = tmp_w / 2
                neighbours.append(neighbour)

    return neighbours
