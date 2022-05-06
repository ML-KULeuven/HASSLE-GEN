import numpy as np
import logging
import pickle
import os
import itertools as it
import copy

from hassle_sls.pysat_solver import (
    solve_weighted_max_sat,
    label_instance,
    is_infeasible,
    is_suboptimal,
    get_value
)
from hassle_sls.type_def import MaxSatModel, Context
from pysat.examples.fm import FM
from pysat.formula import WCNF
from scipy.special import binom
from typing import List


logger = logging.getLogger(__name__)
_MIN_WEIGHT, _MAX_WEIGHT = 1, 101


def generate_models(n, max_clause_length, num_hard, num_soft, model_seed, rng, use_new_generation_method):
    param = f"_n_{n}_max_clause_length_{max_clause_length}_num_hard_{num_hard}_num_soft_{num_soft}_model_seed_{model_seed}_use_new_generation_method_{use_new_generation_method}"
    if os.path.exists("pickles/target_models/" + param + ".pickle"):
        true_model = pickle.load(
            open("pickles/target_models/" + param + ".pickle", "rb")
        )["true_model"]
        return true_model, param
    true_model = generate_model(n, max_clause_length, num_hard, num_soft, use_new_generation_method, rng)
    return true_model, param


def generate_contexts_and_data(
    n, model, num_context, num_pos, num_neg, neg_type, param, rng, context_seed, conjunctive_contexts=0
):
    param += f"_num_context_{num_context}_num_pos_{num_pos}_num_neg_{num_neg}_neg_type_{neg_type}_context_seed_{context_seed}_conjunctive_contexts_{conjunctive_contexts}"
    if os.path.exists("pickles/contexts_and_data/" + param + ".pickle"):
        return param
    pickle_var = {}
    pickle_var["contexts"] = []
    pickle_var["data"] = []
    pickle_var["labels"] = []
    if num_context == 0:
        data, labels = random_data(
            n, model, set(), num_pos, num_neg, neg_type, rng
        )
        pickle_var["contexts"].extend([set()] * len(data))
        pickle_var["data"].extend(data)
        pickle_var["labels"].extend(labels)
    else:
        sol = solve_weighted_max_sat(n, model, None, 1, conjunctive_contexts=conjunctive_contexts)[0]
        opt_val = get_value(model, sol, None, conjunctive_contexts=conjunctive_contexts)

        contexts = [None]
        context = None
        attempts = 0
        while len(contexts) != num_context + 1 and attempts < 100:
            attempts = attempts + 1
            generate_new_context_attempts = 0
            context = random_context(n, rng)
            while context in contexts:
                context = random_context(n, rng)
                generate_new_context_attempts += 1
                if generate_new_context_attempts > 1000:
                    raise Exception("Cannot generate as many unique contexts as requested")

            sol, cst = solve_weighted_max_sat(n, model, context, 1, conjunctive_contexts=conjunctive_contexts)
            # Only add the context if it actually affects the maximal attainable value in the target model
            if sol and opt_val != get_value(model, sol, None, conjunctive_contexts=conjunctive_contexts):
                data, labels = random_data(
                    n, model, context, num_pos, num_neg, neg_type, rng, conjunctive_contexts
                )
                # Only add context when we were able to generate the specified number of instances per context in that context
                if len(data) == num_pos + num_neg:
                    attempts = 0
                    contexts.append(context)
                    pickle_var["contexts"].extend([context] * len(data))
                    pickle_var["data"].extend(data)
                    pickle_var["labels"].extend(labels)
    if len(pickle_var["data"]) == num_context * (num_pos + num_neg):
        return param, pickle_var
    else:
        num_data_created = len(pickle_var["data"])
        print(f"Failed to create requested amount of data. Was only able to create {num_data_created}")
        return None, None


def generate_model(num_vars, clause_length, num_hard, num_soft, use_new_generation_method, rng):
    """
    Generates a new target MAX-SAT model. Can use the old method or the new method, which differ in that the old method
    implicitly gives a strong preference to long clauses, while the new method generates clauses of which the length
    varies more strongly.
    :param num_vars: The number of variables
    :param clause_length: The maximum clause length
    :param num_hard: The number of hard constraints
    :param num_soft: The number of soft constraints
    :param use_new_generation_method: A Boolean that denotes whether to use the new generation method
    :param rng: A numpy RandomState
    :return: The generated MAX-SAT model
    """
    if use_new_generation_method:
        # New method
        return list(sample_models_new(1, num_vars, clause_length, num_hard, num_soft, rng))[0]
    else:
        # Old method
        return list(sample_models(1, num_vars, clause_length, num_hard, num_soft, rng))[0]


def _generate_all_clauses_up_to_length(num_vars, length):
    flip_or_dont = lambda v: -(v - num_vars) if v > num_vars else v

    lits = range(1, 2 * num_vars + 1)
    clauses = set(
        [
            tuple(set(map(flip_or_dont, clause)))
            for clause in it.combinations_with_replacement(lits, length)
        ]
    )

    # This makes sure that all symmetries are accounted for...
    must_be = sum(binom(2 * num_vars, l) for l in range(1, length + 1))
    assert len(clauses) == must_be

    # check entailment property of the added constraints

    # ... except for impossible clauses like 'x and not x', let's delete them
    def possible(clause):
        for i in range(len(clause)):
            for j in range(i + 1, len(clause)):
                if clause[i] == -clause[j]:
                    return False
        return True

    return list(sorted(filter(possible, clauses)))


def sample_models(
    num_models, num_vars, clause_length, num_hard, num_soft, rng
) -> List[MaxSatModel]:
    clauses = _generate_all_clauses_up_to_length(num_vars, clause_length)

    if logger.isEnabledFor(logging.DEBUG):
        # Print the clauses and quit
        from pprint import pprint

        pprint(clauses)

    num_clauses = len(clauses)
    total = num_hard + num_soft
    assert total > 0

    logger.info(f"{num_clauses} clauses total - {num_hard} hard and {num_soft} soft")

    for m in range(num_models):
        logger.info(f"generating model {m + 1} of {num_models}")
        model = []
        wcnf = WCNF()
        indices = get_random_clauses(wcnf, rng, clauses, total)
        if len(indices) < total:
            print(len(clauses), total, len(indices))
        assert len(indices) == total
        hard_indices = list(sorted(rng.permutation(indices)[:num_hard]))
        soft_indices = list(sorted(set(indices) - set(hard_indices)))

        weights = rng.randint(_MIN_WEIGHT, _MAX_WEIGHT, size=num_soft)
        for i in hard_indices:
            model.append((None, set(clauses[i])))
        for i, weight in zip(soft_indices, weights):
            model.append((weight / 100, set(clauses[i])))
        yield model


def contains_entailment(wcnf):
    for i in range(len(wcnf.hard)):
        wcnf_copy = copy.deepcopy(wcnf)
        clause = wcnf.hard[i]
        wcnf_copy.hard.remove(clause)
        if is_entailed(wcnf_copy, clause):
            return True
    return False


def get_random_clauses(wcnf, rng, clauses, num_clauses):
    for trial in range(num_clauses * 10):
        wcnf_copy = copy.deepcopy(wcnf)
        selected_indices = []
        checked_indices = []
        n = num_clauses
        while n > 0:
            indices = [ind for ind in range(len(clauses)) if ind not in checked_indices]
            i = rng.choice(indices)
            checked_indices.append(i)
            wcnf_copy_copy = copy.deepcopy(wcnf_copy)
            wcnf_copy_copy.append(clauses[i])
            # if not is_entailed(wcnf_copy, clauses[i]):
            if not contains_entailment(wcnf_copy_copy):
                wcnf_copy.append(clauses[i])
                selected_indices.append(i)
                n = n - 1
            if len(checked_indices) == len(clauses):
                break
        if n == 0:
            if contains_entailment(wcnf_copy):
                raise Exception("The created model contains an entailment")
            return selected_indices
    return []


def generate_random_clause(rng, num_vars, max_clause_length):
    # Initialise clause
    clause = set()

    # First randomly sample a length:
    #chosen_length = rng.randint(1, max_clause_length+1)
    # even = max_clause_length % 2 == 0
    # if even:
    #     distribution = list(range(1, max_clause_length//2 + 1)) + list(range(max_clause_length//2, 0, -1))
    # else:
    #     distribution = list(range(1, max_clause_length//2 + 2)) + list(range(max_clause_length//2, 0, -1))
    # chosen_length = rng.choice(list(range(1, max_clause_length+1)), 1, p=[distribution[i]/sum(distribution) for i in range(len(distribution))])[0]
    chosen_length = rng.choice(list(range(1, max_clause_length + 1)), 1)[0]

    # Add as many literals as chosen clause length dictates
    variables = set(range(1, num_vars+1))
    for i in range(chosen_length):
        variable = rng.choice(list(variables))
        variables.remove(variable)
        if rng.randint(0, 2) == 0:
            clause.add(int(variable))
        else:
            clause.add(int(-variable))
    return tuple(clause)


def generate_random_clauses(wcnf, rng, num_vars, num_clauses, max_clause_length):
    for trial in range(num_clauses * 10):
        wcnf_copy = copy.deepcopy(wcnf)
        n = num_clauses
        clauses = []
        attempts = 1000
        while n > 0:
            clause = generate_random_clause(rng, num_vars, max_clause_length)
            wcnf_copy_copy = copy.deepcopy(wcnf_copy)
            wcnf_copy_copy.append(clause)
            # if not is_entailed(wcnf_copy, clause):
            if not contains_entailment(wcnf_copy_copy):
                wcnf_copy.append(clause)
                clauses.append(clause)
                n = n - 1
                attempts = 1000
            else:
                attempts = attempts - 1
            if attempts == 0:
                break
        if n == 0:
            if contains_entailment(wcnf_copy):
                raise Exception("The created model contains an entailment")
            return clauses
    raise Exception("Failed to construct a model")


def sample_models_new(num_models, num_vars, max_clause_length, num_hard, num_soft, rng) -> List[MaxSatModel]:
    total = num_hard + num_soft
    assert total > 0
    for m in range(num_models):
        model = []
        wcnf = WCNF()
        clauses = generate_random_clauses(wcnf, rng, num_vars, total, max_clause_length)
    assert len(clauses) == total
    indices = list(range(len(clauses)))
    hard_indices = list(sorted(rng.permutation(indices)[:num_hard]))
    soft_indices = list(sorted(set(indices) - set(hard_indices)))

    weights = rng.randint(_MIN_WEIGHT, _MAX_WEIGHT, size=num_soft)
    for i in hard_indices:
        model.append((None, set(clauses[i])))
    for i, weight in zip(soft_indices, weights):
        model.append((weight / 100, set(clauses[i])))
    yield model


def is_entailed(wcnf, clause):
    wcnf_new = wcnf.copy()
    for literal in clause:
        wcnf_new.append((-literal,))

    fm = FM(wcnf_new, verbose=0)
    #    print(wcnf_new.hard,fm.compute())
    return not fm.compute()


def random_context(n, rng):
    clause = []
    indices = rng.choice(range(n), int(n/2), replace=False)
    for i in range(n):
        if i in indices:
            clause.append(rng.choice([-1, 1]))
        else:
            clause.append(0)
    context = set()
    for j, literal in enumerate(clause):
        if literal != 0:
            context.add((j + 1) * literal)
    return context


def random_data(
    n, model: MaxSatModel, context: Context, num_pos, num_neg, neg_type, rng, conjunctive_contexts=0
):
    data = []
    tmp_data, cst = solve_weighted_max_sat(n, model, context, num_pos * 100, conjunctive_contexts=conjunctive_contexts)
    if len(tmp_data) > num_pos:
        indices = list(rng.choice(range(len(tmp_data)), num_pos, replace=False))
        for i in indices:
            data.append(tmp_data[i])
    else:
        data = tmp_data
    num_pos = len(data)
    labels = [1] * num_pos
    if neg_type == "inf":
        d, l = random_infeasible(n, model, context, num_neg, rng, conjunctive_contexts=conjunctive_contexts)
    elif neg_type == "sub":
        d, l = random_suboptimal(n, model, context, num_neg, rng, conjunctive_contexts=conjunctive_contexts)
    elif neg_type == "both":
        d, l = random_infeasible(n, model, context, int(num_neg / 2), rng, conjunctive_contexts=conjunctive_contexts)
        d1, l1 = random_suboptimal(n, model, context, int(num_neg / 2), rng, conjunctive_contexts=conjunctive_contexts)
        d.extend(d1)
        l.extend(l1)
    data.extend(d)
    labels.extend(l)

    # max_tries = 1000 * num_neg
    # rng = np.random.RandomState(seed)
    # for l in range(max_tries):
    #     instance = rng.rand(n) > 0.5
    #     for i in rng.choice(list(context), 1):
    #         instance[abs(i) - 1] = i > 0
    #     if list(instance) in data:
    #         continue
    #     if not label_instance(model, instance, context):
    #         data.append(list(instance))
    #         if is_infeasible(model, instance, context):
    #             labels.append(-1)
    #         else:
    #             labels.append(0)
    #         if len(data) >= num_neg + num_pos:
    #             break
    return data, labels


def random_infeasible(n, model: MaxSatModel, context: Context, num_neg, rng, conjunctive_contexts=0):
    data = []
    labels = []
    max_tries = 1000 * num_neg
    for l in range(max_tries):
        instance = rng.rand(n) > 0.5
        if not conjunctive_contexts:
            # For disjunctive contexts,
            # only one literal has to occur in instance as it does in the context for the context to be satisfied
            for i in rng.choice(list(context), 1):
                instance[abs(i) - 1] = i > 0
        else:
            # For conjunctive contexts,
            # all literals have to occur in instance as they do in the context for the context to be satisfied
            for i in context:
                instance[abs(i) - 1] = i > 0
        if list(instance) in data:
            continue
        if is_infeasible(model, instance, context, conjunctive_contexts=conjunctive_contexts):
            data.append(list(instance))
            labels.append(-1)
            if len(data) >= num_neg:
                break
    return data, labels


def random_suboptimal(n, model: MaxSatModel, context: Context, num_neg, rng, conjunctive_contexts=0):
    data = []
    labels = []
    max_tries = 1000 * num_neg
    for l in range(max_tries):
        instance = rng.rand(n) > 0.5
        if not conjunctive_contexts:
            # For disjunctive contexts,
            # only one literal has to occur in instance as it does in the context for the context to be satisfied
            for i in rng.choice(list(context), 1):
                instance[abs(i) - 1] = i > 0
        else:
            # For conjunctive contexts,
            # all literals have to occur in instance as they do in the context for the context to be satisfied
            for i in context:
                instance[abs(i) - 1] = i > 0
        if list(instance) in data:
            continue
        if is_suboptimal(model, instance, context, conjunctive_contexts=conjunctive_contexts):
            data.append(list(instance))
            labels.append(0)
            if len(data) >= num_neg:
                break
    return data, labels
