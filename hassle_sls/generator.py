import numpy as np
import logging
import pickle
import os
import itertools as it
import copy

from .pysat_solver import (
    solve_weighted_max_sat,
    label_instance,
    is_infeasible,
    is_suboptimal,
)
from .type_def import MaxSatModel, Context
from pysat.examples.fm import FM
from pysat.formula import WCNF
from scipy.special import binom
from typing import List


logger = logging.getLogger(__name__)
_MIN_WEIGHT, _MAX_WEIGHT = 1, 101


def generate_models(n, max_clause_length, num_hard, num_soft, model_seed):
    param = f"_n_{n}_max_clause_length_{max_clause_length}_num_hard_{num_hard}_num_soft_{num_soft}_model_seed_{model_seed}"
    if os.path.exists("pickles/target_model/" + param + ".pickle"):
        true_model = pickle.load(
            open("pickles/target_model/" + param + ".pickle", "rb")
        )["true_model"]
        return true_model, param
    rng = np.random.RandomState(model_seed)
    true_model = generate_model(n, max_clause_length, num_hard, num_soft, rng)
    pickle_var = {}
    pickle_var["true_model"] = true_model
    if not os.path.exists("pickles/target_model"):
        os.makedirs("pickles/target_model")
    pickle.dump(pickle_var, open("pickles/target_model/" + param + ".pickle", "wb"))
    return true_model, param


def generate_contexts_and_data(
    n, model, num_context, num_pos, num_neg, neg_type, param, context_seed
):
    param += f"_num_context_{num_context}_num_pos_{num_pos}_num_neg_{num_neg}_neg_type_{neg_type}_context_seed_{context_seed}"
    if os.path.exists("pickles/contexts_and_data/" + param + ".pickle"):
        return param
    pickle_var = {}
    rng = np.random.RandomState(context_seed)
    pickle_var["contexts"] = []
    pickle_var["data"] = []
    pickle_var["labels"] = []
    if num_context == 0:
        data, labels = random_data(
            n, model, set(), num_pos, num_neg, neg_type, context_seed
        )
        pickle_var["contexts"].extend([set()] * len(data))
        pickle_var["data"].extend(data)
        pickle_var["labels"].extend(labels)
    else:
        for _ in range(num_context):
            context, data_seed = random_context(n, rng)
            data, labels = random_data(
                n, model, context, num_pos, num_neg, neg_type, data_seed
            )
            pickle_var["contexts"].extend([context] * len(data))
            pickle_var["data"].extend(data)
            pickle_var["labels"].extend(labels)
    if not os.path.exists("pickles/contexts_and_data"):
        os.makedirs("pickles/contexts_and_data")
    pickle.dump(
        pickle_var, open("pickles/contexts_and_data/" + param + ".pickle", "wb")
    )
    return param


def is_entailed(wcnf, clause):
    wcnf_new = wcnf.copy()
    for literal in clause:
        wcnf_new.append((-literal,))

    fm = FM(wcnf_new, verbose=0)
    #    print(wcnf_new.hard,fm.compute())
    return not fm.compute()


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
            if not is_entailed(wcnf_copy, clauses[i]):
                wcnf_copy.append(clauses[i])
                selected_indices.append(i)
                n = n - 1
            if len(checked_indices) == len(clauses):
                break
        if n == 0:
            return selected_indices
    return []


def generate_model(num_vars, clause_length, num_hard, num_soft, rng):
    return list(sample_models(1, num_vars, clause_length, num_hard, num_soft, rng))[0]


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


def random_context(n, rng):
    clause = []
    indices = rng.choice(range(n), 2, replace=False)
    for i in range(n):
        if i in indices:
            clause.append(rng.choice([-1, 1]))
        else:
            clause.append(0)
    context = set()
    for j, literal in enumerate(clause):
        if literal != 0:
            context.add((j + 1) * literal)
    data_seed = rng.randint(1, 1000)
    return context, data_seed


def random_data(
    n, model: MaxSatModel, context: Context, num_pos, num_neg, neg_type, seed
):
    rng = np.random.RandomState(seed)
    data = []
    tmp_data, cst = solve_weighted_max_sat(n, model, context, num_pos * 10)
    if len(tmp_data) > num_pos:
        indices = list(rng.choice(range(len(tmp_data)), num_pos, replace=False))
        for i in indices:
            data.append(tmp_data[i])
    else:
        data = tmp_data
    num_pos = len(data)
    labels = [1] * num_pos
    if neg_type == "inf":
        d, l = random_infeasible(n, model, context, num_neg, seed)
    elif neg_type == "sub":
        d, l = random_suboptimal(n, model, context, num_neg, seed)
    elif neg_type == "both":
        d, l = random_infeasible(n, model, context, int(num_neg / 2), seed)
        d1, l1 = random_suboptimal(n, model, context, int(num_neg / 2), seed)
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


def random_infeasible(n, model: MaxSatModel, context: Context, num_neg, seed):
    data = []
    labels = []
    max_tries = 1000 * num_neg
    rng = np.random.RandomState(seed)
    for l in range(max_tries):
        instance = rng.rand(n) > 0.5
        for i in rng.choice(list(context), 1):
            instance[abs(i) - 1] = i > 0
        if list(instance) in data:
            continue
        if is_infeasible(model, instance, context):
            data.append(list(instance))
            labels.append(-1)
            if len(data) >= num_neg:
                break
    return data, labels


def random_suboptimal(n, model: MaxSatModel, context: Context, num_neg, seed):
    data = []
    labels = []
    max_tries = 1000 * num_neg
    rng = np.random.RandomState(seed)
    for l in range(max_tries):
        instance = rng.rand(n) > 0.5
        for i in rng.choice(list(context), 1):
            instance[abs(i) - 1] = i > 0
        if list(instance) in data:
            continue
        if is_suboptimal(model, instance, context):
            data.append(list(instance))
            labels.append(0)
            if len(data) >= num_neg:
                break
    return data, labels
