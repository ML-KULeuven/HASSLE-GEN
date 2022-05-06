import copy
import random
import numpy as np
from auxiliary import compute_clause_coverage_bitvectors, combine_coverage_bitvectors,\
    combine_coverage_bitvectors_hard_constraints, combine_coverage_bitvectors_soft_constraints


def clause_crossover_1x(ind1, ind2):
    """
    Operates on constraint level.
    Crosses over two individuals by randomly selecting an index and swapping all constraints located prior to that index
    between the two individuals (in-place). Note that this operator has a smaller mixing rate than
    uniform_clause_crossover and scramble_clause_crossover.
    :param ind1: The first individual
    :param ind2: The second individual
    """
    k = len(ind1)
    cx_point = random.randint(1, k - 1)
    temp = ind1[cx_point:]
    ind1[cx_point:] = ind2[cx_point:]
    ind2[cx_point:] = temp


# This uniform crossover operates on the level of literals
def uniform_crossover(ind1, ind2):
    """
    Operates on literal level.
    Crosses over two individuals by, for every variable in every i-th constraint, swapping that variable's occurrence
    (positive, negative, absent) in the constraint at that location between the two individuals (in-place),
    with probability 0.5.
    :param ind1: The first individual
    :param ind2: The second individual
    """
    k = len(ind1)
    for i in range(k):
        for j in range(len(ind1[i])):
            if random.random() < 0.5:
                temp = ind1[i][j]
                ind1[i][j] = ind2[i][j]
                ind2[i][j] = temp


def matched_uniform_crossover(ind1, ind2):
    """
    Operates on literal level.
    Works the same as uniform_crossover, except that constraints are not matched on their (meaningless) index in the
    individual, but based on their syntactic similarity. In a first step, each constraint in each individual is matched
    with the most syntactically similar (based on Hamming distance) unmatched constraint from the other individual.
    Then, for each pair of matched constraints, each variable's occurrence (positive, negative, absent) is swapped
    between the two constraints (in-place) with probability 0.5.
    :param ind1: The first individual
    :param ind2: The second individual
    """
    # We calculate the pairwise match between ind1's and ind2's clauses
    match_matrix = np.zeros((len(ind1), len(ind2)))
    for i in range(len(ind1)):
        clause1 = ind1[i]
        for j in range(len(ind2)):
            clause2 = ind2[j]
            curr_syntactic_match = 0
            for k in range(len(clause1)):
                if k != len(clause1) - 1:
                    if clause1[k] == clause2[k]:
                        curr_syntactic_match += 1
                else:
                    curr_syntactic_match += (1 - abs(clause1[k] - clause2[k]))
            match_matrix[i][j] = curr_syntactic_match

    # Arg-sort the pairwise clause matches from best to worst match
    matches_ordered = np.dstack(np.unravel_index(np.argsort(match_matrix.ravel())[::-1], (len(ind1), len(ind2))))[0]

    # Finally match the clauses, making sure that each clause is only matched once
    # Then perform uniform crossover on matched clauses
    ind1_matched_clauses = set()
    ind2_matched_clauses = set()
    count = 0
    for match in matches_ordered:
        i = match[0]
        j = match[1]
        if match_matrix[i][j] >= len(ind1[0])//2:
            if i not in ind1_matched_clauses and j not in ind2_matched_clauses:
                count += 1
                # Perform the uniform crossover
                for k in range(len(ind1[i])):
                    if random.random() < 0.5:
                        temp = ind1[i][k]
                        ind1[i][k] = ind2[j][k]
                        ind2[j][k] = temp
                ind1_matched_clauses.add(i)
                ind2_matched_clauses.add(j)


# This uniform crossover operates on the level of clauses
def uniform_clause_crossover(ind1, ind2):
    """
    Operates on constraint level.
    Crosses over two individuals by, for every clause index i, swapping the two clauses occurring at that index in the
    two individuals (in-place) with probability 0.5. Note that this operator has a larger mixing rate than
    clause_crossover_1x, but a lower mixing rate than scramble_clause_crossover.
    :param ind1: The first individual
    :param ind2: The second individual
    """
    k = len(ind1)
    for i in range(k):
        if random.random() < 0.5:
            temp = ind1[i]
            ind1[i] = ind2[i]
            ind2[i] = temp


def scramble_clause_crossover(ind1, ind2):
    """
    Operates on constraint level.
    Crosses over two individuals by first collecting all constraints occurring in the individuals in a list of size 2*k,
    randomly scrambling the list, and then assigning the first k constraints in the list to the first individual, and
    the second k constraints to the second individual. Note that this operator has a larger mixing rate than
    clause_crossover_1x and uniform clause_crossover.
    :param ind1: The first individual
    :param ind2: The second individual
    :return:
    """
    all_clauses = ind1 + ind2
    random.shuffle(all_clauses)
    ind1[0:len(ind1)] = all_clauses[0:len(ind1)]
    ind2[0:len(ind2)] = all_clauses[len(ind1):len(ind1) + len(ind2)]


def avoid_duplicate_clauses_scramble_clause_crossover(ind1, ind2):
    """
    Operates on constraint level.
    Works the same as scramble_clause_crossover, except that all occurrences of clauses that occur multiple times in
    either individual, or that occur in both individuals, are first evenly divided over both individuals.
    :param ind1: The first individual
    :param ind2: The second individual
    """
    ind_length = len(ind1)
    ind1_copy = copy.deepcopy(ind1)
    ind2_copy = copy.deepcopy(ind2)

    clauses_both_have = []
    remaining_clauses = []
    for clause in ind1:
        try:
            index = ind2_copy.index(clause)
            clauses_both_have.append(clause)
            del ind2_copy[index]
        except ValueError:
            remaining_clauses.append(clause)

    for clause in ind2:
        try:
            index = ind1_copy.index(clause)
            del ind1_copy[index]
        except ValueError:
            remaining_clauses.append(clause)

    random.shuffle(remaining_clauses)
    ind1[0:len(clauses_both_have)] = clauses_both_have
    ind2[0:len(clauses_both_have)] = clauses_both_have
    ind1[len(clauses_both_have):] = remaining_clauses[:len(remaining_clauses) // 2]
    ind2[len(clauses_both_have):] = remaining_clauses[len(remaining_clauses) // 2:]
    if len(ind1) != ind_length or len(ind2) != ind_length:
        raise Exception("Crossover operator altered the length of an individual")


def smart_clause_crossover_dispatch(ind1, ind2, examples, greedy=True, probability_variant=None, temperature=1, clause_bitvector_cache=None, use_infeasibility=False):
    """
    Dispatches to the appropriate smart clause crossover operator, depending on whether the distinct should be made
    between infeasible and suboptimal negative examples
    :param ind1: The first individual
    :param ind2: The second individual
    :param examples: A list of examples, where each example is a context-instance-label tuple
    :param greedy: A boolean that denotes whether to select clauses according to the coverage heuristic greedily. If
    False, the clauses are selected according to the coverage heuristic probabilistically, depending on the
    argument for probability_variant
    :param probability_variant: What probabilistic selection variant to use. Options are:
    "linear" - The values of the clauses according to the coverage heuristic are linearly converted to probabilities
    "squared" - The values of the clauses according to the coverage heuristic are first squared and then converted to
    probabilities
    "softmax" - The values of the clauses according to the coverage heuristic are cast to probabilities using the
    softmax function, with a temperature corresponding to the argument of the temperature parameter
    Only relevant when greedy is False
    :param temperature: The temperature to use in the softmax conversion of values of the coverage heuristic to
    probabilities. Only relevant when greedy is False and probability_variant is softmax
    :param clause_bitvector_cache: Optional - a dictionary in which to cache clause coverages
    :param use_infeasibility: A Boolean that denotes whether to make use of the distinction between infeasible and
    suboptimal negative examples. If False, a negative example is simply interpreted as being negative, and no
    distinction between infeasibility and suboptimality is made.
    """
    if use_infeasibility:
        smart_clause_crossover_infeasibility(ind1, ind2, examples, greedy=greedy, probability_variant=probability_variant, temperature=temperature, clause_bitvector_cache=clause_bitvector_cache)
    else:
        smart_clause_crossover(ind1, ind2, examples, greedy=greedy, probability_variant=probability_variant, temperature=temperature, clause_bitvector_cache=clause_bitvector_cache)


def smart_clause_crossover(ind1, ind2, examples, greedy=True, probability_variant=None, temperature=1, clause_bitvector_cache=None):
    """
    Operates on constraint level.
    This function does not make use of the distinction between suboptimal and infeasible negative examples.
    Performs a smart crossover on the two given individuals, and produces 1 individual (both individuals will be
    changed into this single resulting individual). Makes use of a heuristic that values constraints and combinations of
    constraints based on their coverage.
    :param ind1: The first individual
    :param ind2: The second individual
    :param examples: A list of examples, where each example is a context-instance-label tuple
    :param greedy: A boolean that denotes whether to select clauses according to the coverage heuristic greedily. If
    False, the clauses are selected according to the coverage heuristic probabilistically, depending on the
    argument for probability_variant
    :param probability_variant: What probabilistic selection variant to use. Options are:
    "linear" - The values of the clauses according to the coverage heuristic are linearly converted to probabilities
    "squared" - The values of the clauses according to the coverage heuristic are first squared and then converted to
    probabilities
    "softmax" - The values of the clauses according to the coverage heuristic are cast to probabilities using the
    softmax function, with a temperature corresponding to the argument of the temperature parameter
    Only relevant when greedy is False
    :param temperature: The temperature to use in the softmax conversion of values of the coverage heuristic to
    probabilities. Only relevant when greedy is False and probability_variant is softmax
    :param clause_bitvector_cache: Optional - a dictionary in which to cache clause coverages
    """
    allow_duplicates = False  # allow_duplicates denotes whether the resulting indivuals may contain duplicate clauses
    number_of_clauses = len(ind1)
    all_clauses = ind1+ind2
    chosen_clauses = []
    chosen_clause_indices = []
    ind1_coverage_bitvectors = compute_clause_coverage_bitvectors(ind1, examples, clause_bitvector_cache=clause_bitvector_cache)
    ind2_coverage_bitvectors = compute_clause_coverage_bitvectors(ind2, examples, clause_bitvector_cache=clause_bitvector_cache)
    all_coverage_bitvectors = ind1_coverage_bitvectors + ind2_coverage_bitvectors

    for i in range(0, number_of_clauses):
        if i == 0:
            combined_coverage_bitvectors = all_coverage_bitvectors
        else:
            combined_coverage_bitvectors = [combine_coverage_bitvectors(chosen_clauses_bitvector, bitvector, examples)
                                            for bitvector in all_coverage_bitvectors]
            if not allow_duplicates:
                for index in chosen_clause_indices:
                    for j in range(len(combined_coverage_bitvectors)):
                        if all_clauses[index] == all_clauses[j]:
                            combined_coverage_bitvectors[j] = [0] * len(examples)
        combined_coverages = [sum(coverage_bitvector) for coverage_bitvector in combined_coverage_bitvectors]
        if greedy:
            best_coverage = max(combined_coverages)
            best_indices = [i for i in range(len(combined_coverages)) if combined_coverages[i] == best_coverage]
            chosen_clause_index = random.choice(best_indices)
        else:
            if probability_variant == "linear":
                sum_coverages = sum(combined_coverages)
                coverages_to_probabilities = [x / sum_coverages for x in combined_coverages]
            elif probability_variant == "squared":
                coverages_squared = [x ** 2 for x in combined_coverages]
                sum_coverages_squared = sum(coverages_squared)
                coverages_to_probabilities = [x ** 2 / sum_coverages_squared for x in combined_coverages]
            elif probability_variant == "softmax":
                # Softmax with normalization to prevent overflow
                coverages_max = max(combined_coverages)
                coverages_for_softmax = [a_coverage - coverages_max for a_coverage in combined_coverages]
                coverages_to_probabilities = np.exp(np.asarray(coverages_for_softmax) / temperature) / sum(
                    np.exp(np.asarray(coverages_for_softmax) / temperature))

            chosen_clause_index = np.random.choice(list(range(0, len(all_coverage_bitvectors))),
                                                   p=coverages_to_probabilities)
        chosen_coverage_bitvector = combined_coverage_bitvectors[chosen_clause_index]
        if chosen_clause_index < number_of_clauses:
            chosen_clause = ind1[chosen_clause_index]
        else:
            chosen_clause = ind2[chosen_clause_index - number_of_clauses]

        chosen_clauses.append(chosen_clause)
        chosen_clause_indices.append(chosen_clause_index)
        chosen_clauses_bitvector = chosen_coverage_bitvector

    for i in range(len(chosen_clauses)):
        clause = chosen_clauses[i]
        # We can safely set ind1 and ind2 to the same computed smart combination, as only one of them will make it
        # to the next generation
        ind1[i] = clause
        ind2[i] = clause


def smart_clause_crossover_infeasibility(ind1, ind2, examples, greedy=True, probability_variant=None, temperature=1, clause_bitvector_cache=None):
    """
    Operates on constraint level.
    This function makes use of the distinction between suboptimal and infeasible negative examples.
    Performs a smart crossover on the two given individuals, and produces 1 individual (both individuals will be
    changed into this single resulting individual). Makes use of a heuristic that values constraints and combinations of
    constraints based on their coverage.
    :param ind1: The first individual
    :param ind2: The second individual
    :param examples: A list of examples, where each example is a context-instance-label tuple
    :param greedy: A boolean that denotes whether to select clauses according to the coverage heuristic greedily. If
    False, the clauses are selected according to the coverage heuristic probabilistically, depending on the
    argument for probability_variant
    :param probability_variant: What probabilistic selection variant to use. Options are:
    "linear" - The values of the clauses according to the coverage heuristic are linearly converted to probabilities
    "squared" - The values of the clauses according to the coverage heuristic are first squared and then converted to
    probabilities
    "softmax" - The values of the clauses according to the coverage heuristic are cast to probabilities using the
    softmax function, with a temperature corresponding to the argument of the temperature parameter
    Only relevant when greedy is False
    :param temperature: The temperature to use in the softmax conversion of values of the coverage heuristic to
    probabilities. Only relevant when greedy is False and probability_variant is softmax
    :param clause_bitvector_cache: Optional - a dictionary in which to cache clause coverages
    """
    allow_duplicates = False  # allow_duplicates denotes whether the resulting indivuals may contain duplicate clauses
    ind1_hard_constraints = [constr for constr in ind1 if constr[-2] == True]
    ind2_hard_constraints = [constr for constr in ind2 if constr[-2] == True]
    all_hard_constraints = ind1_hard_constraints + ind2_hard_constraints
    ind1_soft_constraints = [constr for constr in ind1 if constr[-2] == False]
    ind2_soft_constraints = [constr for constr in ind2 if constr[-2] == False]
    all_soft_constraints = ind1_soft_constraints + ind2_soft_constraints
    ind1_hard_coverage_bitvectors = compute_clause_coverage_bitvectors(ind1_hard_constraints, examples, use_infeasibility=True, clause_bitvector_cache=clause_bitvector_cache)
    ind2_hard_coverage_bitvectors = compute_clause_coverage_bitvectors(ind2_hard_constraints, examples, use_infeasibility=True, clause_bitvector_cache=clause_bitvector_cache)
    ind1_soft_coverage_bitvectors = compute_clause_coverage_bitvectors(ind1_soft_constraints, examples, use_infeasibility=True, clause_bitvector_cache=clause_bitvector_cache)
    ind2_soft_coverage_bitvectors = compute_clause_coverage_bitvectors(ind2_soft_constraints, examples, use_infeasibility=True, clause_bitvector_cache=clause_bitvector_cache)
    all_hard_coverage_bitvectors = ind1_hard_coverage_bitvectors + ind2_hard_coverage_bitvectors
    all_soft_coverage_bitvectors = ind1_soft_coverage_bitvectors + ind2_soft_coverage_bitvectors

    ind1_num_hard = len([constr for constr in ind1 if constr[-2] == True])
    ind2_num_hard = len([constr for constr in ind2 if constr[-2] == True])
    # num_hard = random.choice([ind1_num_hard, ind2_num_hard])
    if ind1_num_hard <= ind2_num_hard:
        num_hard = random.choice(list(range(ind1_num_hard, ind2_num_hard+1)))
    else:
        num_hard = random.choice(list(range(ind2_num_hard, ind1_num_hard + 1)))
    num_soft = len(ind1) - num_hard
    chosen_hard_clauses = []
    chosen_hard_clause_indices = []
    chosen_soft_clauses = []
    chosen_soft_clause_indices = []

    # Choose hard constraints
    for i in range(0, num_hard):
        if i == 0:
            combined_hard_coverage_bitvectors = all_hard_coverage_bitvectors
        else:
            combined_hard_coverage_bitvectors = [combine_coverage_bitvectors_hard_constraints(
                chosen_hard_clauses_bitvector, bitvector, examples) for bitvector in all_hard_coverage_bitvectors]
            if not allow_duplicates:
                for index in chosen_hard_clause_indices:
                    for j in range(len(combined_hard_coverage_bitvectors)):
                        if all_hard_constraints[index][:-2] == all_hard_constraints[j][:-2]:
                            combined_hard_coverage_bitvectors[j] = [0] * len(examples)
        if greedy:
            combined_hard_coverages = [sum(coverage_bitvector) for coverage_bitvector in combined_hard_coverage_bitvectors]
            best_hard_coverage = max(combined_hard_coverages)
            best_hard_indices = [i for i in range(len(combined_hard_coverages)) if combined_hard_coverages[i] == best_hard_coverage]
            chosen_hard_clause_index = random.choice(best_hard_indices)
        else:
            coverages = [sum(x) for x in combined_hard_coverage_bitvectors]
            if probability_variant == "linear":
                sum_coverages = sum(coverages)
                coverages_to_probabilities = [x / sum_coverages for x in coverages]
            elif probability_variant == "squared":
                coverages_squared = [x ** 2 for x in coverages]
                sum_coverages_squared = sum(coverages_squared)
                coverages_to_probabilities = [x ** 2 / sum_coverages_squared for x in coverages]
            elif probability_variant == "softmax":
                # Softmax with normalization to prevent overflow
                coverages_max = max(coverages)
                coverages_for_softmax = [a_coverage - coverages_max for a_coverage in coverages]
                coverages_to_probabilities = np.exp(np.asarray(coverages_for_softmax) / temperature) / sum(
                    np.exp(np.asarray(coverages_for_softmax) / temperature))

            chosen_hard_clause_index = np.random.choice(list(range(0, len(all_hard_coverage_bitvectors))),
                                                   p=coverages_to_probabilities)
        chosen_hard_coverage_bitvector = combined_hard_coverage_bitvectors[chosen_hard_clause_index]
        if chosen_hard_clause_index < len(ind1_hard_constraints):
            chosen_hard_clause = ind1_hard_constraints[chosen_hard_clause_index]
        else:
            chosen_hard_clause = ind2_hard_constraints[chosen_hard_clause_index - len(ind1_hard_constraints)]

        chosen_hard_clauses.append(chosen_hard_clause)
        chosen_hard_clause_indices.append(chosen_hard_clause_index)
        chosen_hard_clauses_bitvector = chosen_hard_coverage_bitvector

    # Choose soft constraints
    for i in range(0, num_soft):
        if i == 0:
            combined_soft_coverage_bitvectors = all_soft_coverage_bitvectors
        else:
            combined_soft_coverage_bitvectors = [combine_coverage_bitvectors_soft_constraints(
                chosen_soft_clauses_bitvector, bitvector, examples) for bitvector in all_soft_coverage_bitvectors]
            if not allow_duplicates:
                for index in chosen_soft_clause_indices:
                    for j in range(len(combined_soft_coverage_bitvectors)):
                        if all_soft_constraints[index][:-2] == all_soft_constraints[j][:-2]:
                            combined_soft_coverage_bitvectors[j] = [0] * len(examples)
        if greedy:
            combined_soft_coverages = [sum(coverage_bitvector) for coverage_bitvector in combined_soft_coverage_bitvectors]
            best_soft_coverage = max(combined_soft_coverages)
            best_soft_indices = [i for i in range(len(combined_soft_coverages)) if combined_soft_coverages[i] == best_soft_coverage]
            chosen_soft_clause_index = random.choice(best_soft_indices)
        else:
            coverages = [sum(x) for x in combined_soft_coverage_bitvectors]
            if probability_variant == "linear":
                sum_coverages = sum(coverages)
                coverages_to_probabilities = [x / sum_coverages for x in coverages]
            elif probability_variant == "squared":
                coverages_squared = [x ** 2 for x in coverages]
                sum_coverages_squared = sum(coverages_squared)
                coverages_to_probabilities = [x ** 2 / sum_coverages_squared for x in coverages]
            elif probability_variant == "softmax":
                # Softmax with normalization to prevent overflow
                coverages_max = max(coverages)
                coverages_for_softmax = [a_coverage - coverages_max for a_coverage in coverages]
                coverages_to_probabilities = np.exp(np.asarray(coverages_for_softmax) / temperature) / sum(
                    np.exp(np.asarray(coverages_for_softmax) / temperature))

            chosen_soft_clause_index = np.random.choice(list(range(0, len(all_soft_coverage_bitvectors))),
                                                        p=coverages_to_probabilities)
        chosen_soft_coverage_bitvector = combined_soft_coverage_bitvectors[chosen_soft_clause_index]
        if chosen_soft_clause_index < len(ind1_soft_constraints):
            chosen_soft_clause = ind1_soft_constraints[chosen_soft_clause_index]
        else:
            chosen_soft_clause = ind2_soft_constraints[chosen_soft_clause_index - len(ind1_soft_constraints)]

        chosen_soft_clauses.append(chosen_soft_clause)
        chosen_soft_clause_indices.append(chosen_soft_clause_index)
        chosen_soft_clauses_bitvector = chosen_soft_coverage_bitvector

    for i in range(len(chosen_hard_clauses)):
        hard_clause = chosen_hard_clauses[i]
        # We can safely set ind1 and ind2 to the same computed smart combination, as only one of them will make it
        # to the next generation
        ind1[i] = hard_clause
        ind2[i] = hard_clause

    for i in range(len(chosen_soft_clauses)):
        soft_clause = chosen_soft_clauses[i]
        ind1[num_hard+i] = soft_clause
        ind2[num_hard+i] = soft_clause
