#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 12:49:04 2019

@author: mohit
"""

from typing import Optional
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
import numpy as np
import csv
import copy

from .type_def import MaxSatModel, Clause, Instance, Context


def solve_weighted_max_sat(
    n: int, model: MaxSatModel, context: Clause, num_sol, prev_sol=[], conjunctive_contexts=False
):
    """
    Solves a MaxSatModel and tries to return num_sol optimal solutions
    """
    c = WCNF()
    c.nv = n
    for w, clause in model:
        #c.append(list(map(int, list(clause))), weight=w)
        if w != 0 and len(clause) > 0:
            c.append(list(map(int, list(clause))), weight=w)

    if context and len(context) > 0:
        if not conjunctive_contexts:
            #c.append(list(map(int, list(context))), weight=None)
            c.append(list(map(int, list(context))))
        else:
            # Conjunctive context
            for i in list(context):
                c.append([int(i)])
    s = RC2(c)
    sol = []
    cst = -1

    for m in s.enumerate():
        # while len(m) < n:
        #     m.append(len(m) + 1)

        if cst < 0:
            cst = s.cost
        if s.cost > cst or len(sol) >= num_sol:
            break
        m = [v > 0 for v in m]
        if m not in prev_sol:
            sol.append(m)
        if len(sol) >= num_sol:
            break
    if num_sol == 1 and sol:
        return sol[0], cst
    return sol, cst


def solve_weighted_max_sat_file(
    wcnf_file, context: Clause, num_sol, prev_sol=[]
) -> Optional[Instance]:
    """
    Solves a MaxSatModel file and tries to return num_sol optimal solutions
    """
    wcnf = WCNF(wcnf_file)
    if len(context) > 0:
        wcnf.append(list(map(int, list(context))), weight=None)
    s = RC2(wcnf)
    model = []
    cst = -1
    for m in s.enumerate():
        if cst < 0:
            cst = s.cost
        if s.cost > cst or len(model) >= num_sol:
            break
        m = [v > 0 for v in m]
        if np.array(m) not in prev_sol:
            model.append(np.array(m))
    return model


def get_value(model: MaxSatModel, instance: Instance, context=None, conjunctive_contexts=False) -> Optional[float]:
    """
    Returns the weighted value of an instance
    """
    model = copy.deepcopy(model)
    if context is not None and len(context) > 0:
        if not conjunctive_contexts:
            model.append((None, context))
        else:
            for i in context:
                model.append((None, {i}))
    value = 0
    for weight, clause in model:
        try:
            covered = len(clause) > 0 and any(not instance[abs(i) - 1] if i < 0 else instance[i - 1] for i in clause)
        except:
            print(f"Instance is: {instance}")
            print(f"Clause is: {clause}")
            raise Exception("Error")
        if weight is None:
            if not covered:
                return None
        else:
            if covered:
                value += weight
    return value


def get_cost(model: MaxSatModel, instance: Instance) -> Optional[float]:
    """
    Returns the weighted value of an instance
    """
    value = 0
    for weight, clause in model:
        covered = any(
            not instance[abs(i) - 1] if i < 0 else instance[i - 1] for i in clause
        )
        if weight is None:
            if not covered:
                return None
        else:
            if not covered:
                value += weight
    return value


def label_instance(model: MaxSatModel, instance: Instance, context: Context) -> bool:
    value = get_value(model, instance, context)
    if value is None:
        return False
    best_instance, cst = solve_weighted_max_sat(len(instance), model, context, 1)
    if cst < 0:
        return False
    best_value = get_value(model, best_instance, context)
    return value == best_value


def label_instance_with_cache(model: MaxSatModel, instance: Instance, context: Context,
                              cached_best_value=None, value_of_instance=-1, conjunctive_contexts=0):

    if value_of_instance == -1:
        value = get_value(model, instance, context, conjunctive_contexts=conjunctive_contexts)
    else:
        value = value_of_instance

    if value is None:
        return (-1, None)

    if cached_best_value is not None and cached_best_value != 0:
        # Do not use cached_best_value when it is equal to 0, because the cached_best_value---when provided---results
        # from inference on a algebraic decision diagram, in which no distinction can be made between an infeasible
        # instance and a feasible instance which satisfies 0 weight in soft constraints. This can lead to a bug in
        # evaluation, because depending on whether the model is simply infeasible or its optimal value is 0, a different
        # label might have to be given for an instance with value 0
        best_value = cached_best_value
    else:
        if cached_best_value == 0 and value_of_instance != -1:
            # Do not rely on algebraic_decision_diagram when cached_best_value is 0, for the reasoning given above
            value = get_value(model, instance, context, conjunctive_contexts=conjunctive_contexts)
            if value is None:
                return (-1, None)
        best_instance, cst = solve_weighted_max_sat(len(instance), model, context, 1, conjunctive_contexts=conjunctive_contexts)
        if cst < 0:
            return (-1, None)
        best_value = get_value(model, best_instance, context, conjunctive_contexts=conjunctive_contexts)
    if round(value, 10) == round(best_value, 10):
        return (1, best_value)
    else:
        return (0, best_value)


def is_infeasible(model: MaxSatModel, instance: Instance, context: Context, conjunctive_contexts:int=0) -> bool:
    value = get_value(model, instance, context, conjunctive_contexts=conjunctive_contexts)
    if value is None:
        return True
    return False


def is_suboptimal(model: MaxSatModel, instance: Instance, context: Context, conjunctive_contexts:int=0) -> bool:
    value = get_value(model, instance, context, conjunctive_contexts=conjunctive_contexts)
    if value is None:
        return False
    best_instance, cst = solve_weighted_max_sat(len(instance), model, context, 1, conjunctive_contexts=conjunctive_contexts)
    best_value = get_value(model, best_instance, context, conjunctive_contexts=conjunctive_contexts)
    return not value == best_value


def represent_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def fix_var(cnf_file, output_path, var):
    output_file = open(output_path, "w")
    filewriter = csv.writer(output_file, delimiter=" ")
    final_lines = []
    final_vars = set()
    num_soft = 1
    with open(cnf_file) as fp:
        for line in fp:
            words = line.strip().split(" ")
            if words[0] == "p":
                final_lines.append(words)
                continue
            elif words[0] == "c":
                continue
            literals = list(map(int, words[1:]))
            sat = 0
            for l in literals:
                if l in var:
                    sat = 1
                    break
            if sat == 0:
                tmp = [l for i, l in enumerate(literals) if -1 * l not in var]
                if len(tmp) <= 1:
                    continue
                if int(words[0]) <= 1:
                    num_soft += 1
                    final_lines.append([words[0]])
                else:
                    final_lines.append([num_soft])
                final_lines[len(final_lines) - 1].extend(tmp)
                final_vars = final_vars.union(
                    list(map(abs, final_lines[len(final_lines) - 1][1:-1]))
                )

    final_lines[0][4] = num_soft
    final_lines[0][3] = len(final_lines) - 1
    final_lines[0][2] = len(final_vars)
    final_vars = sorted(final_vars)
    for i, line in enumerate(final_lines):
        if i > 0:
            for j, l in enumerate(line[1:-1]):
                line[j + 1] = np.sign(l) * (final_vars.index(abs(l)) + 1)
        filewriter.writerow(line)
    output_file.close()


def simplify(cnf_file, output_file, num_var_not_fixed, rng):
    sol = solve_weighted_max_sat_file(cnf_file, [], 1)
    int_sol = []
    for i, val in enumerate(sol[0]):
        if val:
            int_sol.append(i + 1)
        else:
            int_sol.append(-(i + 1))
    sol = rng.choice(int_sol, size=len(int_sol) - num_var_not_fixed, replace=False)
    output_file += ".wcnf"
    fix_var(cnf_file, output_file, sol)
    wcnf = WCNF(output_file)
    wcnf.to_file(output_file)
    return cnf_to_model(output_file, rng)


def cnf_to_model(cnf_file, rng):
    model = []
    n = 0
    k = 0
    with open(cnf_file) as fp:
        line = fp.readline()
        while line:
            line_as_list = line.strip().split()
            if line_as_list[0] == "p":
                n = int(line_as_list[2])
                k = int(line_as_list[3])
            elif len(line_as_list) > 1 and represent_int(line_as_list[0]):
                model.append((int(line_as_list[0]), set(map(int, line_as_list[1:-1]))))

            line = fp.readline()
    if model:
        return model, n, k
    return None


def model_to_cnf(model, param, param_str, path):
    cnf_file = open(path + param_str + ".cnf", "w")
    filewriter = csv.writer(cnf_file, delimiter=" ")
    filewriter.writerow(
        [
            "p",
            "wcnf",
            param["n"],
            param["num_hard"] + param["num_soft"],
            param["num_soft"] + 1,
        ]
    )
    for weight, clause in model:
        if weight is None:
            tmp = [param["num_soft"] + 1]
            tmp.extend(list(clause))
            filewriter.writerow(tmp)
        else:
            tmp = [weight]
            tmp.extend(clause)
            filewriter.writerow(tmp)
    cnf_file.close()
