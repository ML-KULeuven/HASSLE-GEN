#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 16:46:25 2020

@author: mohit
"""

import numpy as np
import time
from typing import List
from .type_def import Clause
from .maxsat import MaxSAT
import copy
import os
import pickle
from tqdm import tqdm
import auxiliary
import evaluation
import random


# def eval_neighbours(
#     correct_examples, neighbours, data, labels, contexts, num_neighbours, rng, inf=None
# ):
#     if not inf:
#         inf = [None] * len(labels)
#
#     neighbours = copy.copy(neighbours)
#     next_correct_examples = np.zeros([len(neighbours), data.shape[0]])
#
#     scores = [0 for i in range(len(neighbours))]
#
#     for m, nbr in enumerate(neighbours):
#         optimums = {}
#         for i, example in enumerate(data):
#             key = "_".join(map(str, contexts[i]))
#             if key not in optimums:
#                 optimums[key] = nbr.optimal_value(contexts[i])
#             if correct_examples[i] == 1 and nbr.is_correct(
#                 example, labels[i], contexts[i], inf=inf[i], optimum=optimums[key]
#             ):
#                 next_correct_examples[m, i] = 1
#                 scores[m] += 1
#
#     lst_scores = []
#     lst_models = []
#     lst_correct_examples = []
#     for _ in range(num_neighbours):
#         lst_scores.append(max(scores))
#         best_index = rng.choice(
#             [i for i, v in enumerate(scores) if v == lst_scores[-1]]
#         )
#         lst_models.append(neighbours[best_index])
#         del scores[best_index]
#         del neighbours[best_index]
#         optimums = {}
#         for i, example in enumerate(data):
#             if correct_examples[i] == 0:
#                 key = "_".join(map(str, contexts[i]))
#                 if key not in optimums:
#                     optimums[key] = lst_models[-1].optimal_value(contexts[i])
#                 if lst_models[-1].is_correct(
#                     example, labels[i], contexts[i], inf=inf[i], optimum=optimums[key]
#                 ):
#                     next_correct_examples[best_index, i] = 1
#                     lst_scores[-1] += 1
#         lst_correct_examples.append(next_correct_examples[best_index, :])
#     return lst_models, lst_scores, lst_correct_examples


def eval_neighbours(neighbours, data, labels, contexts, num_neighbours, rng, inf=None, conjunctive_contexts=0):
    if not inf:
        inf = [None] * len(labels)

    neighbours = copy.copy(neighbours)
    next_correct_examples = np.zeros([len(neighbours), data.shape[0]])

    scores = [0 for i in range(len(neighbours))]

    for m, nbr in enumerate(neighbours):
        optimums = {}
        for i, example in enumerate(data):
            key = "_".join(map(str, contexts[i]))
            if key not in optimums:
                optimums[key] = nbr.optimal_value(contexts[i], conjunctive_contexts=conjunctive_contexts)
            if nbr.is_correct(
                    example, labels[i], contexts[i], inf=inf[i], optimum=optimums[key],
                    conjunctive_contexts=conjunctive_contexts
            ):
                next_correct_examples[m, i] = 1
                scores[m] += 1

    lst_scores = []
    lst_models = []
    lst_correct_examples = []
    for _ in range(num_neighbours):
        lst_scores.append(max(scores))
        best_index = rng.choice(
            [i for i, v in enumerate(scores) if v == lst_scores[-1]]
        )
        lst_models.append(neighbours[best_index])
        lst_correct_examples.append(next_correct_examples[best_index])
        del scores[best_index]
        del neighbours[best_index]
        next_correct_examples = np.delete(next_correct_examples, best_index, 0)
    return lst_models, lst_scores, lst_correct_examples


def walk_sat(neighbours, data, labels, contexts, rng, inf=None, use_knowledge_compilation=False,
             knowledge_compilation_variant=4, use_diagram_for_instance_evaluation=True,
             conjunctive_contexts=0):
    if use_knowledge_compilation:
        examples = [[contexts[i], data[i], labels[i]] for i in range(len(data))]
        next_models, scores, correct_examples = auxiliary.rank_neigbours_knowledge_compilation(neighbours, examples,
                                                                                               knowledge_compilation_variant=knowledge_compilation_variant,
                                                                                               use_diagram_for_instance_evaluation=use_diagram_for_instance_evaluation,
                                                                                               conjunctive_contexts=conjunctive_contexts,
                                                                                               inf=inf)
        scores = [round(score_as_proportion * len(examples)) for score_as_proportion in scores]
    else:
        next_models, scores, correct_examples = eval_neighbours(
            neighbours, data, labels, contexts, 1, rng, inf, conjunctive_contexts=conjunctive_contexts
        )
    return next_models[0], scores[0], correct_examples[0], len(neighbours)


def novelty(prev_model, neighbours, data, labels, contexts, rng, inf=None, use_knowledge_compilation=False,
            knowledge_compilation_variant=4, use_diagram_for_instance_evaluation=True,
            conjunctive_contexts=0):
    if use_knowledge_compilation:
        examples = [[contexts[i], data[i], labels[i]] for i in range(len(data))]
        lst_models, lst_scores, lst_correct_examples = auxiliary.rank_neigbours_knowledge_compilation(neighbours,
                                                                                                    examples,
                                                                                                    knowledge_compilation_variant=knowledge_compilation_variant,
                                                                                                    use_diagram_for_instance_evaluation=use_diagram_for_instance_evaluation,
                                                                                                    conjunctive_contexts=conjunctive_contexts,
                                                                                                    inf=inf)
        lst_scores = [round(score_as_proportion * len(examples)) for score_as_proportion in lst_scores]
    else:
        lst_models, lst_scores, lst_correct_examples = eval_neighbours(
            neighbours, data, labels, contexts, 2, rng, inf, conjunctive_contexts=conjunctive_contexts
        )
    if not lst_models[0].is_same(prev_model):
        return lst_models[0], lst_scores[0], lst_correct_examples[0], len(neighbours)
    else:
        return lst_models[1], lst_scores[1], lst_correct_examples[1], len(neighbours)


def novelty_large(prev_models, neighbours, data, labels, contexts, rng, inf=None, use_knowledge_compilation=False,
                  knowledge_compilation_variant=4, use_diagram_for_instance_evaluation=True,
                  conjunctive_contexts=0):
    if use_knowledge_compilation:
        examples = [[contexts[i], data[i], labels[i]] for i in range(len(data))]
        lst_models, lst_scores, lst_correct_examples = auxiliary.rank_neigbours_knowledge_compilation(neighbours,
                                                                                                    examples,
                                                                                                    knowledge_compilation_variant=knowledge_compilation_variant,
                                                                                                    use_diagram_for_instance_evaluation=use_diagram_for_instance_evaluation,
                                                                                                    conjunctive_contexts=conjunctive_contexts,
                                                                                                    inf=inf)
        lst_scores = [round(score_as_proportion * len(examples)) for score_as_proportion in lst_scores]
    else:
        lst_models, lst_scores, lst_correct_examples = eval_neighbours(
            neighbours, data, labels, contexts, 2, rng, inf, conjunctive_contexts=conjunctive_contexts
        )
    for i in range(len(lst_models)):
        # Return the best model that is not part of prev_models
        next_best_model = lst_models[i]
        if not any([next_best_model.is_same(a_model) for a_model in prev_models]):
            return next_best_model, lst_scores[i], lst_correct_examples[i], i
    # If all models are part of prev_models, we might as well return the best one
    return lst_models[0], lst_scores[0], lst_correct_examples[0], len(neighbours), len(lst_models)


def novelty_plus(prev_model, neighbours, data, labels, contexts, wp, rng, inf=None, use_knowledge_compilation=False,
                 knowledge_compilation_variant=4, use_diagram_for_instance_evaluation=True,
                 conjunctive_contexts=0):
    if rng.random_sample() < wp:
        next_model = neighbours[rng.randint(0, len(neighbours))]
        score, correct_examples = next_model.score(data, labels, contexts, inf,
                                                   conjunctive_contexts=conjunctive_contexts)
        return next_model, score, correct_examples, 1
    return novelty(prev_model, neighbours, data, labels, contexts, rng, inf, use_knowledge_compilation,
                   knowledge_compilation_variant=knowledge_compilation_variant,
                   use_diagram_for_instance_evaluation=use_diagram_for_instance_evaluation,
                   conjunctive_contexts=conjunctive_contexts)


def adaptive_novelty_plus(
        prev_model,
        neighbours,
        data,
        labels,
        contexts,
        wp,
        theta,
        phi,
        best_scores,
        rng,
        inf=None,
        use_knowledge_compilation=False,
        knowledge_compilation_variant=4,
        use_diagram_for_instance_evaluation=True,
        conjunctive_contexts=0
):
    steps = int(len(labels) * theta)
    if len(best_scores) > steps:
        if best_scores[-steps] == best_scores[-1]:
            wp = wp + (1 - wp) * phi
        else:
            wp = wp - (wp * 2 * phi)
    if rng.random_sample() < wp:
        next_model = neighbours[rng.randint(0, len(neighbours))]
        score, correct_examples = next_model.score(data, labels, contexts, inf,
                                                   conjunctive_contexts=conjunctive_contexts)
        return next_model, score, correct_examples, wp
    next_model, score, correct_examples, evaluations_done = novelty(
        prev_model, neighbours, data, labels, contexts, rng, inf, use_knowledge_compilation,
        knowledge_compilation_variant=knowledge_compilation_variant,
        use_diagram_for_instance_evaluation=use_diagram_for_instance_evaluation,
        conjunctive_contexts=conjunctive_contexts
    )
    return next_model, score, correct_examples, evaluations_done, wp


def ternary(n, length):
    e = n // 3
    q = n % 3
    if length > 1:
        if n == 0:
            return ternary(e, length - 1) + [0]
        elif e == 0:
            return ternary(e, length - 1) + [q]
        else:
            return ternary(e, length - 1) + [q]
    else:
        if n == 0:
            return [0]
        elif e == 0:
            return [q]
        else:
            return ternary(e, length - 1) + [q]


def random_model(num_var, num_constraints, clause_len, rng, variable_absence_bias=1) -> MaxSAT:
    c = [rng.randint(0, 2) for i in range(num_constraints)]
    w = [float("%.3f" % (rng.uniform(0.001, 0.999))) for i in range(num_constraints)]
    l = []
    i = 1
    while i <= num_constraints:
        clause = []
        sample = rng.choice(num_var, clause_len, replace=False)
        for j in range(num_var):
            if j in sample:
                clause.append(int(rng.choice([-1, 0, 1], p=[0.5 / (1 + variable_absence_bias),
                                                            variable_absence_bias / (1 + variable_absence_bias),
                                                            0.5 / (1 + variable_absence_bias)])))
            else:
                clause.append(0)

        if clause not in l:
            l.append(clause)
            i += 1
    return MaxSAT(c, w, l)


def random_incorrect_example_index(model, data, contexts, labels, infeasible, rng, conjunctive_contexts=0):
    x = list(enumerate(np.copy(data)))
    rng.shuffle(x)
    indices, data = zip(*x)
    if not infeasible:
        infeasible = [None] * len(indices)
    index = 0
    for i, example in enumerate(data):
        if not model.is_correct(
                example, labels[indices[i]], contexts[indices[i]], infeasible[indices[i]],
                conjunctive_contexts=conjunctive_contexts
        ):
            index = i
            break
    return indices[index]


def prune_neighbourhood_with_coverage_heuristic(current, neighbours, examples, inf):
    """
    Prunes the neighbourhood so that only those neighbours who do at least as well as the current model according to the
    'coverage heuristic' are kept.
    :param current: The current model
    :param neighbours: A list of neighbours
    :param examples: The list of examples, each of which is a context-instance-label tuple.
    :param inf: Optional. A list with as many entries as there are examples. Each entry is True if the corresponding
    example is infeasible, and False otherwise. An argument should be supplied if a distinction is to be made between
    infeasible and suboptimal negative examples.
    :return: A pruned list of neighbours
    """
    neighbours_pruned = []
    for neighbour in neighbours:
        if has_better_or_equal_coverage_heuristic(neighbour, current, examples, inf):
            neighbours_pruned.append(neighbour)
    return neighbours_pruned


def has_better_or_equal_coverage_heuristic(neighbour, current, examples, inf):
    """
    Computes whether a given neighbouring model does at least as well as the current model according to the 'coverage
    heuristic'. A neighbour that does not differ from the current model in a variable's occurrence in a clause (but
    rather in a constraint's weight or hardness) trivially does equally well on the heuristic. For a neighbour that does
    differ in one specific clause's literals, the coverage of that clause as it occurs in the neighbour is computed, as
    well as for that clause as it occurs in the current model. If the former coverage is at least as large as the
    latter, True is returned. Else, False is returned.
    :param neighbour: The neighbouring model
    :param current: The current model
    :param examples: The list of examples, each of which is a context-instance-label tuple.
    :param inf: Optional. A list with as many entries as there are examples. Each entry is True if the corresponding
    example is infeasible, and False otherwise. An argument should be supplied if a distinction is to be made between
    infeasible and suboptimal negative examples.
    :return: True if the neighbouring model oes at least as well as the current model according to the 'coverage
    heuristic', False otherwise
    """
    neighbour_as_genotype = auxiliary.MaxSAT_to_genotype(neighbour)
    current_as_genotype = auxiliary.MaxSAT_to_genotype(current)

    # Compute the index of the clause in which the current model and the neighbouring model differ in a literal
    index_differing_clause = compute_differing_clause_index(neighbour_as_genotype, current_as_genotype)

    if index_differing_clause is None:
        # If the current model and the neighbouring model do not differ in a variable's occurrence (but rather in a
        # constraint's weight or hardness), the coverage heuristic's value is the same
        return True

    neighbour_constraint = neighbour_as_genotype[index_differing_clause]
    current_constraint = current_as_genotype[index_differing_clause]

    # Some preprocessing to be in accordance with the format expected by the HASSLE-GEN code
    if inf is None:
        use_infeasibility = False
        transformed_examples = examples
    else:
        use_infeasibility = True
        transformed_examples = []
        for i in range(len(examples)):
            example = examples[i]
            if example[2]:
                new_label = 1
            elif inf[i]:
                new_label = -1
            else:
                new_label = 0
            transformed_examples.append((example[0], example[1], new_label))

    # Compute coverages
    neighbour_coverage = sum(auxiliary.compute_clause_coverage_bitvector(neighbour_constraint, transformed_examples,
                                                    clause_bitvector_cache=None, use_infeasibility=use_infeasibility))
    current_coverage = sum(auxiliary.compute_clause_coverage_bitvector(current_constraint, transformed_examples,
                                                    clause_bitvector_cache=None, use_infeasibility=use_infeasibility))

    return neighbour_coverage >= current_coverage


def compute_differing_clause_index(neighbour, current):
    """
    Computes the index of the (first) clause for which the current model and a neighbouring model differ in a variable's
    occurrence. If there is no such clause (and thus the models differ in a constraint's hardness or weight), None is
    returned
    :param neighbour: The neighbouring model
    :param current: The current model
    :return: The index of the (first) clause for which the current model and a neighbouring model differ in a variable's
    occurrence
    """
    for i in range(len(neighbour)):
        for j in range(len(neighbour[i]) - 2):
            if neighbour[i][j] != current[i][j]:
                return i
    return None


def learn_weighted_max_sat(
        num_constraints: int,
        clause_len: int,
        data: np.ndarray,
        labels: np.ndarray,
        contexts: List[Clause],
        method,
        param,
        inf=None,
        p=0.01,
        wp=0.1,
        theta=0.17,
        phi=0.2,
        window_size=100,
        cutoff_time=5,
        seed=1,
        use_knowledge_compilation=False,
        knowledge_compilation_variant=4,
        use_diagram_for_instance_evaluation=True,
        recompute_random_incorrect_examples=True,
        conjunctive_contexts=False,
        perform_random_restarts=True,
        prune_with_coverage_heuristic=False,
        initialization_attempts=1,
        variable_absence_bias=1,
        neighbourhood_limit=None,
        initial_model=None,
        observers=None
):
    """
    Learn a weighted MaxSAT model from examples. Contexts and clauses are set-encoded, i.e., they are represented by
    sets containing positive or negative integers in the range -n-1 to n+1. If a set contains an positive integer i, the i-1th
     Boolean feature is set to True, if it contains a negative integer -i, the i-1th Boolean feature is set to False.
    :param num_constraints:
        The number of clauses in the MaxSAT model to learn
    :param data:
        A Boolean s x n (number examples x number Boolean variables) numpy array in which every row is an example and
        every column a Boolean feature (True = 1 or False = 0)
    :param labels:
        A Boolean numpy array (length s) where the kth entry contains the label (1 or 0) of the kth example
    :param contexts:
        A list of s set-encoded contexts.
    :param use_knowledge_compilation:
        A Boolean flag denoting whether evaluation of a model's neighbours should use knowledge compilation
    :return:
        A list of weights and clauses. Every entry of the list is a tuple containing as first element None for hard
        constraints (clauses) or a floating point number for soft constraints, and as second element a set-encoded clause.
    """
    # if not os.path.exists("pickles/learned_model"):
    #     os.makedirs("pickles/learned_model")

    rng = np.random.RandomState(seed)
    if use_knowledge_compilation or prune_with_coverage_heuristic:
        examples = [[contexts[i], data[i], labels[i]] for i in range(len(data))]

    # Initialising timers
    random_restart_time = 0
    computing_neighbours_time = 0
    evaluation_time = 0
    cumulative_time = 0

    # Initialization
    time_point = time.time()
    bar = tqdm("Score", total=100)
    number_of_evaluations = 0
    if initial_model is None:
        model = None
        score = 0


        for i in range(initialization_attempts):
            new_model = random_model(data.shape[1], num_constraints, clause_len, rng,
                                     variable_absence_bias=variable_absence_bias)
            if use_knowledge_compilation:
                model_as_phenotype = auxiliary.to_phenotype(auxiliary.MaxSAT_to_genotype(new_model))
                new_score_as_proportion, new_correct_examples = evaluation.evaluate_knowledge_compilation_based_dispatch(
                    model_as_phenotype, examples, knowledge_compilation_variant=knowledge_compilation_variant,
                    use_diagram_for_instance_evaluation=use_diagram_for_instance_evaluation,
                    conjunctive_contexts=conjunctive_contexts, inf=inf)
                new_score = int(round(new_score_as_proportion * len(examples)))
            else:
                new_score, new_correct_examples = new_model.score(data, labels, contexts, inf,
                                                                  conjunctive_contexts=conjunctive_contexts)
            if new_score > score:
                model = new_model
                score = new_score
                correct_examples = new_correct_examples

        number_of_evaluations += initialization_attempts
    else:
        model = auxiliary.phenotype_to_MaxSAT(initial_model, data.shape[1])
        model_as_phenotype = initial_model
        score_as_proportion, correct_examples = evaluation.evaluate_knowledge_compilation_based_dispatch(
            model_as_phenotype, examples, knowledge_compilation_variant=knowledge_compilation_variant,
            use_diagram_for_instance_evaluation=use_diagram_for_instance_evaluation,
            conjunctive_contexts=conjunctive_contexts, inf=inf)
        score = int(round(score_as_proportion * len(examples)))
        number_of_evaluations += 1
    evaluation_time += time.time() - time_point
    bar.update(score * 100 / data.shape[0])


    # Update cumulative time
    cumulative_time = random_restart_time + computing_neighbours_time + evaluation_time

    # Some setup
    prev_model = model
    prev_models = [model]
    solutions = [model.deep_copy().maxSatModel()]
    best_scores = [score]
    best_model_correct_examples = [correct_examples]
    time_taken = [cumulative_time]
    random_restart_count = 0
    total_window_hits = 0
    for observer in observers:
        observer.observe_generation(
            gen_count=0,
            best_score=score / data.shape[0],
            gen_duration=time_taken[-1],
            current_model_correct_examples=correct_examples,
            best_model_correct_examples=best_model_correct_examples[-1],
            current_score=score / data.shape[0],
            number_of_neighbours=0,
            number_of_window_hits=total_window_hits,
            number_of_evaluations=number_of_evaluations,
            cumulative_time=cumulative_time,
            random_restart_time=random_restart_time,
            computing_neighbours_time=computing_neighbours_time,
            evaluation_time=evaluation_time,
            random_restart_count=random_restart_count
        )
    iterations = [0]
    itr = 0
    num_neighbours = [0]
    nbr = 0

    num_example = data.shape[0]
    last_update_time = cumulative_time

    while (
            # score < len(labels)
            # and
            cumulative_time < cutoff_time
    ):
        if perform_random_restarts and cumulative_time - last_update_time > cutoff_time / 4:
            # if rng.random_sample() < p:

            # Random restart
            random_restart_count += 1
            time_point = time.time()
            next_model = random_model(data.shape[1], num_constraints, clause_len, rng,
                                      variable_absence_bias=variable_absence_bias)
            random_restart_time += time.time() - time_point

            time_point = time.time()
            if use_knowledge_compilation:
                model_as_phenotype = auxiliary.to_phenotype(auxiliary.MaxSAT_to_genotype(model))
                score_as_proportion, correct_examples = \
                    evaluation.evaluate_knowledge_compilation_based_dispatch(model_as_phenotype, examples,
                                                                             knowledge_compilation_variant=knowledge_compilation_variant,
                                                                             use_diagram_for_instance_evaluation=use_diagram_for_instance_evaluation,
                                                                 conjunctive_contexts=conjunctive_contexts, inf=inf)
                score = int(round(score_as_proportion * len(examples)))
            else:
                score, correct_examples = next_model.score(data, labels, contexts, inf,
                                                           conjunctive_contexts=conjunctive_contexts)
            number_of_evaluations += 1
            evaluation_time += time.time() - time_point
            cumulative_time = random_restart_time + computing_neighbours_time + evaluation_time
            last_update_time = cumulative_time
        else:
            # Compute neighbourhood
            time_point = time.time()
            if "naive" in param:
                neighbours = model.valid_neighbours()
            else:
                if recompute_random_incorrect_examples:
                    index = random_incorrect_example_index(model, data, contexts, labels, inf, rng,
                                                           conjunctive_contexts=conjunctive_contexts)
                else:
                    incorrect_examples = [i for i in range(len(correct_examples)) if correct_examples[i] == 0]
                    if incorrect_examples:
                        index = random.choice(incorrect_examples)
                    else:
                        index = random.choice([i for i in range(len(correct_examples))])

                if inf:
                    infeasible = inf
                else:
                    infeasible = [None] * len(data)
                neighbours = model.get_neighbours(
                    data[index],
                    contexts[index],
                    labels[index],
                    clause_len,
                    rng,
                    infeasible[index],
                    conjunctive_contexts=conjunctive_contexts,
                    neighbourhood_limit=neighbourhood_limit
                )

                if prune_with_coverage_heuristic:
                    neighbours = prune_neighbourhood_with_coverage_heuristic(model, neighbours, examples, inf)

            computing_neighbours_time += time.time() - time_point

            if len(neighbours) == 0 or (method != "walk_sat" and len(neighbours) < 2):
                old_cumulative_time = cumulative_time
                cumulative_time = random_restart_time + computing_neighbours_time + evaluation_time
                itr += 1
                for observer in observers:
                    observer.observe_generation(
                        itr,
                        best_scores[-1] / data.shape[0],
                        gen_duration=cumulative_time - old_cumulative_time,
                        current_model_correct_examples=correct_examples,
                        best_model_correct_examples=best_model_correct_examples[-1],
                        current_score=score / data.shape[0],
                        number_of_neighbours=len(neighbours),
                        number_of_window_hits=total_window_hits,
                        number_of_evaluations=number_of_evaluations,
                        cumulative_time=cumulative_time,
                        random_restart_time=random_restart_time,
                        computing_neighbours_time=computing_neighbours_time,
                        evaluation_time=evaluation_time,
                        random_restart_count=random_restart_count
                    )
                continue
            nbr += len(neighbours)

            # Compute model update
            time_point = time.time()

            if method == "walk_sat":
                next_model, score, correct_examples, evaluations_done = walk_sat(
                    neighbours, data, labels, contexts, rng, inf, use_knowledge_compilation,
                    knowledge_compilation_variant=knowledge_compilation_variant,
                    use_diagram_for_instance_evaluation=use_diagram_for_instance_evaluation,
                    conjunctive_contexts=conjunctive_contexts
                )
            elif method == "novelty":
                next_model, score, correct_examples, evaluations_done = novelty(
                    prev_model, neighbours, data, labels, contexts, rng, inf, use_knowledge_compilation,
                    knowledge_compilation_variant=knowledge_compilation_variant,
                    use_diagram_for_instance_evaluation=use_diagram_for_instance_evaluation,
                    conjunctive_contexts=conjunctive_contexts
                )
            elif method == "novelty_large":
                next_model, score, correct_examples, evaluations_done, window_hits = novelty_large(
                    prev_models, neighbours, data, labels, contexts, rng, inf, use_knowledge_compilation,
                    knowledge_compilation_variant=knowledge_compilation_variant,
                    use_diagram_for_instance_evaluation=use_diagram_for_instance_evaluation,
                    conjunctive_contexts=conjunctive_contexts
                )
                total_window_hits += window_hits
            elif method == "novelty_plus":
                next_model, score, correct_examples, evaluations_done = novelty_plus(
                    prev_model, neighbours, data, labels, contexts, wp, rng, inf, use_knowledge_compilation,
                    knowledge_compilation_variant=knowledge_compilation_variant,
                    use_diagram_for_instance_evaluation=use_diagram_for_instance_evaluation,
                    conjunctive_contexts=conjunctive_contexts
                )
            elif method == "adaptive_novelty_plus":
                next_model, score, correct_examples, evaluations_done, wp = adaptive_novelty_plus(
                    prev_model,
                    neighbours,
                    data,
                    labels,
                    contexts,
                    wp,
                    theta,
                    phi,
                    best_scores,
                    rng,
                    inf,
                    use_knowledge_compilation,
                    knowledge_compilation_variant=knowledge_compilation_variant,
                    use_diagram_for_instance_evaluation=use_diagram_for_instance_evaluation,
                    conjunctive_contexts=conjunctive_contexts
                )
            # Computing a model update almost entirely comes down to evaluation all the model's neighbours, so
            # we include the time this update takes in the evaluation time
            evaluation_time += time.time() - time_point
            number_of_evaluations += evaluations_done

        prev_model = model
        if method == "novelty_large":
            # Only have to keep track of multiple previous models when we are using novelty_large
            if len(prev_models) < window_size:
                prev_models.append(model)
            else:
                prev_models = prev_models[1:] + [prev_model]
        model = next_model
        itr += 1

        # Update cumulative time
        old_cumulative_time = cumulative_time
        cumulative_time = random_restart_time + computing_neighbours_time + evaluation_time

        if score > best_scores[-1]:
            # Found a new best model
            solutions.append(model.deep_copy().maxSatModel())
            bar.update((score - best_scores[-1]) * 100 / num_example)
            iterations.append(itr)
            num_neighbours.append(nbr)
            best_scores.append(score)
            best_model_correct_examples.append(correct_examples)
            last_update_time = cumulative_time
            time_taken.append(cumulative_time)
        for observer in observers:
            observer.observe_generation(
                itr,
                best_scores[-1] / data.shape[0],
                gen_duration=cumulative_time - old_cumulative_time,
                current_model_correct_examples=correct_examples,
                best_model_correct_examples=best_model_correct_examples[-1],
                current_score=score / data.shape[0],
                number_of_neighbours=len(neighbours),
                number_of_window_hits=total_window_hits,
                number_of_evaluations=number_of_evaluations,
                cumulative_time=cumulative_time,
                random_restart_time=random_restart_time,
                computing_neighbours_time=computing_neighbours_time,
                evaluation_time=evaluation_time,
                random_restart_count=random_restart_count
            )

    for i, score in enumerate(best_scores):
        best_scores[i] = score * 100 / num_example
    # pickle_var = {
    #     "learned_model": solutions,
    #     "time_taken": time_taken,
    #     "score": best_scores,
    #     "iterations": iterations,
    #     "num_neighbour": num_neighbours,
    # }
    # if "cnf" in param:
    #     pickle_var = {
    #         "learned_model": [solutions[-1]],
    #         "time_taken": [time_taken[-1]],
    #         "score": [best_scores[-1]],
    #         "iterations": [iterations[-1]],
    #         "num_neighbour": num_neighbours,
    #     }
    #pickle.dump(pickle_var, open("pickles/learned_model/" + param + ".pickle", "wb"))
    print(f"Iterations: {itr}")
    print(f"Timing:\n"
          f"Random restarts: {random_restart_time}\n"
          f"Computing neighbours: {computing_neighbours_time}\n"
          f"Evaluation: {evaluation_time}\n"
          f"Total: {cumulative_time}\n")
    if method == "novelty_large":
        print(f"Number of window hits: {total_window_hits}")
    return solutions[-1]
    # return (solutions, best_scores, time_taken, iterations, num_neighbours)
