import random
import copy
import hassle_sls.local_search
import numpy as np
from auxiliary import MaxSAT_to_genotype, genotype_to_MaxSAT, compute_clause_coverage_bitvector,\
    combine_coverage_bitvectors_hard_and_soft_constraints, combine_coverage_bitvectors,\
    combine_coverage_bitvectors_hard_constraints, combine_coverage_bitvectors_soft_constraints


def flip_literal(clause, index):
    """
    Randomly assigns the variable occurrence of the index-th variable in a given clause into positive, negative or
    absent.
    :param clause: The clause
    :param index: The index
    """
    clause[index] = random.choice([i for i in (-1, 0, 1) if i != clause[index]])


def mutate_hardness(individual, trigger_prob, inner_prob):
    """
    Mutates an individual (in-place) by flipping the hardness of each of that individual's constraints (independently),
    with a certain probability.
    :param individual: The individual
    :param trigger_prob: The probability that this operator triggers
    :param inner_prob: The probability that, given that the operator triggered, a constraint's hardness will be flipped.
    Applies independently to each constraint.
    """
    if random.random() < trigger_prob:
        for i in range(len(individual)):
            if random.random() < inner_prob:
                individual[i][-2] = not individual[i][-2]


def mutate_weight(individual, trigger_prob, inner_prob):
    """
    Mutates an individual (in-place) by sampling the weight of each of that individual's constraints (independently)
    uniformly at random from [0, 1], with a certain probability.
    :param individual: The individual
    :param trigger_prob: The probability that this operator triggers
    :param inner_prob: The probability that, given that the operator triggered, a constraint's weight will be resampled.
    Applies independently to each constraint.
    """
    if random.random() < trigger_prob:
        for i in range(len(individual)):
            if random.random() < inner_prob:
                individual[i][-1] = random.uniform(0, 1)


def mutate_clause(individual, trigger_prob, inner_prob):
    """
    Mutates an individual (in-place) by randomly resampling the variable occurrence (positive, negative or absent) of
    each variable in each of that individual's constraints (independently), with a certain probability.
    :param individual: The individual
    :param trigger_prob: The probability that this operator triggers
    :param inner_prob: The probability that, given that the operator triggered, a variable's occurence in a constraint
    will be resampled.
    Applies independently to each constraint.
    """
    if random.random() < trigger_prob:
        for i in range(len(individual)):
            for j in range(len(individual[i]) - 2):
                if random.random() < inner_prob:
                    flip_literal(individual[i], j)


def mutate_to_neighbour(individual, trigger_prob, contexts, data, boolean_labels, inf=None):
    """
    Mutates an individual (in-place) into a random neighbour, where the neighbourhood is defined as it is in HASSLE-SLS.
    :param individual: The individual
    :param trigger_prob: The probability that this operator triggers
    :param contexts: The list of contexts, as they occur in the examples
    :param data: The list of instances, as they occur in the examples
    :param boolean_labels: The list of Boolean labels, as they occur in the examples
    :param inf: Optional - a list with as many entries as there are examples. Each entry is a Boolean value that denotes
    whether said value's label is 'infeasible' or not. Only relevant when the algorithm should make a distinction
    between infeasible and suboptimal negative examples
    """
    if random.random() < trigger_prob:
        seed = int(random.random() * 10 ** 8)
        rng = np.random.RandomState(seed)
        model = genotype_to_MaxSAT(individual)
        index = hassle_sls.local_search.random_incorrect_example_index(model, data, contexts, boolean_labels, infeasible=inf, rng=rng)
        clause_len = len(individual[0]) - 2
        neighbours = model.get_neighbours(
            data[index],
            contexts[index],
            boolean_labels[index],
            clause_len,
            rng,
            infeasible=inf
        )
        if len(neighbours) > 0:
            chosen_neighbour = MaxSAT_to_genotype(random.choice(neighbours))
            for i in range(len(individual)):
                individual[i] = chosen_neighbour[i]


def mutate_clause_smart(individual, trigger_prob, examples, clause_bitvector_cache=None, use_infeasibility=False, temperature=None):
    """
    Mutates an individual by first selecting a random constraint, then computing a neighbourhood around that constraint
    of constraints that differ in a single variable occurrence (positive, negative or absent), and then the randomly
    selected constraint (in-place) into the neighbouring constraint with the largest value according to the coverage
    heuristic (or a constraint sampled according to the value of the coverage heuristic).
    :param individual: The individual
    :param trigger_prob: The probability that this operator triggers
    :param examples: A list of examples, where each example is a context-instance-label tuple
    :param clause_bitvector_cache: Optional - a dictionary in which to cache clause coverages
    :param use_infeasibility: A Boolean that denotes whether to make use of the distinction between infeasible and
    suboptimal negative examples. If False, a negative example is simply interpreted as being negative, and no
    distinction between infeasibility and suboptimality is made.
    :param temperature: Optional - When temperature is not None, the neighbouring constraint to mutate into is selected
    with a probability that results from applying the softmax function to its coverage heuristic value with respect to
    the coverage heursitic values of all other neighbouring constraints. The value of temperature determines the
    temperature to use in the softmax conversion of values of the coverage heuristic to probabilities of being selected.
    When temperature is None, the argmax is taken.
    """
    if random.random() < trigger_prob:
        constraint_index = random.randint(0, len(individual)-1)
        constraint = individual[constraint_index]
        other_constraints = individual[:constraint_index] + individual[constraint_index + 1:]
        possible_lits = {-1, 0, 1}
        constraint_neighbourhood = []
        for lit_idx in range(len(constraint)-2):
            lit = constraint[lit_idx]
            lit_replacements = possible_lits.difference({lit})
            for a_lit in lit_replacements:
                constraint_neighbour = copy.deepcopy(constraint)
                constraint_neighbour[lit_idx] = a_lit
                if not any([other_constraint[:-2] == constraint_neighbour[:-2] for other_constraint in other_constraints]):
                    # Only consider alterations that do not lead to a duplicate constraint
                    constraint_neighbourhood.append(constraint_neighbour)
        constraint_neighbourhood_coverages = [sum(compute_clause_coverage_bitvector(constraint_neighbour, examples, clause_bitvector_cache=clause_bitvector_cache, use_infeasibility=use_infeasibility))
                                              for constraint_neighbour in constraint_neighbourhood]
        if temperature is None:
            best_coverage = max(constraint_neighbourhood_coverages)
            best_indices = [i for i in range(len(constraint_neighbourhood_coverages)) if constraint_neighbourhood_coverages[i] == best_coverage]
            best_index = random.choice(best_indices)
        else:
            # Softmax with normalization to prevent overflow
            constraint_neighbourhood_coverages_max = max(constraint_neighbourhood_coverages)
            coverages_for_softmax = [a_coverage - constraint_neighbourhood_coverages_max for a_coverage in constraint_neighbourhood_coverages]
            coverages_to_probabilities = np.exp(np.asarray(coverages_for_softmax) / temperature) / sum(
                np.exp(np.asarray(coverages_for_softmax) / temperature))

            best_index = np.random.choice(list(range(0, len(constraint_neighbourhood_coverages))),
                                          p=coverages_to_probabilities)
        individual[constraint_index] = constraint_neighbourhood[best_index]


def mutate_clause_smarter(individual, trigger_prob, examples, clause_bitvector_cache=None, use_infeasibility=False, temperature=None):
    """
    Works the same as mutate_clause_smart, except that it is not a neighbouring constraint's value according to the
    coverage heuristic that determines the neighbouring constraint's probability of being mutated into. Rather, this
    probability is determined by the result of combining the neighbouring constraint's coverage bit vector with those of
    all other constraints present in the individual. So, mutate_clause_smarter differs from mutate_clause smart in that
    it does not a value constraint by using an approximation of how well they 'match' with the examples, but rather by
    using an approximation of how well the constraint 'matches' the examples in combination with the other constraints
    present in the individual.
    :param individual: The individual
    :param trigger_prob: The probability that this operator triggers
    :param examples: A list of examples, where each example is a context-instance-label tuple
    :param clause_bitvector_cache: Optional - a dictionary in which to cache clause coverages
    :param use_infeasibility: A Boolean that denotes whether to make use of the distinction between infeasible and
    suboptimal negative examples. If False, a negative example is simply interpreted as being negative, and no
    distinction between infeasibility and suboptimality is made.
    :param temperature: Optional - When temperature is not None, the neighbouring constraint to mutate into is selected
    with a probability that results from applying the softmax function to its combined coverage heuristic value with
    respect to the combined coverage heursitic values of all other neighbouring constraints.
    The value of temperature determines the temperature to use in the softmax conversion of values of the combined
    coverage heuristic to probabilities of being selected. When temperature is None, the argmax is taken.
    """
    if random.random() < trigger_prob:
        constraint_index = random.randint(0, len(individual)-1)
        constraint = individual[constraint_index]
        other_constraints = individual[:constraint_index] + individual[constraint_index + 1:]
        possible_lits = {-1, 0, 1}
        constraint_neighbourhood = []
        for lit_idx in range(len(constraint)-2):
            lit = constraint[lit_idx]
            lit_replacements = possible_lits.difference({lit})
            for a_lit in lit_replacements:
                constraint_neighbour = copy.deepcopy(constraint)
                constraint_neighbour[lit_idx] = a_lit
                if not any([other_constraint[:-2] == constraint_neighbour[:-2] for other_constraint in other_constraints]):
                    # Only consider alterations that do not lead to a duplicate constraint
                    constraint_neighbourhood.append(constraint_neighbour)
        constraint_neighbourhood_bitvectors = [compute_clause_coverage_bitvector(constraint_neighbour, examples, clause_bitvector_cache=clause_bitvector_cache, use_infeasibility=use_infeasibility)
                                              for constraint_neighbour in constraint_neighbourhood]
        constraint_neighbourhood_coverages = [sum(constraint_neighbour_bitvector) for constraint_neighbour_bitvector in constraint_neighbourhood_bitvectors]

        if use_infeasibility:
            other_constraints = individual[:constraint_index] + individual[constraint_index+1:]

            hard_constraints = [constraint for constraint in other_constraints if constraint[-2] == True]
            if len(hard_constraints) > 0:
                hard_constraints_bitvector = compute_clause_coverage_bitvector(hard_constraints[0], examples, clause_bitvector_cache=clause_bitvector_cache, use_infeasibility=use_infeasibility)
                for hard_constraint in hard_constraints[1:]:
                    hard_constraints_bitvector = combine_coverage_bitvectors_hard_constraints(hard_constraints_bitvector,
                                                                                              compute_clause_coverage_bitvector(hard_constraint, examples, clause_bitvector_cache=clause_bitvector_cache, use_infeasibility=use_infeasibility),
                                                                                              examples)
            else:
                hard_constraints_bitvector = [1]*len(examples)

            soft_constraints = [constraint for constraint in other_constraints if constraint[-2] == False]
            if len(soft_constraints) > 0:
                soft_constraints_bitvector = compute_clause_coverage_bitvector(soft_constraints[0], examples, clause_bitvector_cache=clause_bitvector_cache, use_infeasibility=use_infeasibility)
                for soft_constraint in soft_constraints[1:]:
                    soft_constraints_bitvector = combine_coverage_bitvectors_soft_constraints(soft_constraints_bitvector,
                                                                                              compute_clause_coverage_bitvector(soft_constraint, examples, clause_bitvector_cache=clause_bitvector_cache, use_infeasibility=use_infeasibility),
                                                                                              examples)
            else:
                soft_constraints_bitvector = [1]*len(examples)

            combined_coverage_bitvectors = []
            if constraint[-2] == True:
                for neighbour_bitvector in constraint_neighbourhood_bitvectors:
                    new_hard_constraints_bitvector = combine_coverage_bitvectors_hard_constraints(hard_constraints_bitvector, neighbour_bitvector, examples=examples)
                    combined_coverage_bitvectors.append(combine_coverage_bitvectors_hard_and_soft_constraints(new_hard_constraints_bitvector, soft_constraints_bitvector, examples=examples))
            else:
                for neighbour_bitvector in constraint_neighbourhood_bitvectors:
                    new_soft_constraints_bitvector = combine_coverage_bitvectors_soft_constraints(soft_constraints_bitvector, neighbour_bitvector, examples=examples)
                    combined_coverage_bitvectors.append(combine_coverage_bitvectors_hard_and_soft_constraints(hard_constraints_bitvector, new_soft_constraints_bitvector, examples=examples))

        else:
            other_constraints = individual[:constraint_index] + individual[constraint_index+1:]

            combined_coverage_bitvector = compute_clause_coverage_bitvector(other_constraints[0], examples, clause_bitvector_cache=clause_bitvector_cache, use_infeasibility=use_infeasibility)
            for other_constraint in other_constraints[1:]:
                combined_coverage_bitvector = combine_coverage_bitvectors(combined_coverage_bitvector,
                                                                          compute_clause_coverage_bitvector(other_constraint, examples, clause_bitvector_cache=clause_bitvector_cache, use_infeasibility=use_infeasibility),
                                                                          examples)

            combined_coverage_bitvectors = [combine_coverage_bitvectors(combined_coverage_bitvector, neighbour_bitvector, examples)
                                            for neighbour_bitvector in constraint_neighbourhood_bitvectors]
        #print([sum(combined_coverage_bitvector) for combined_coverage_bitvector in combined_coverage_bitvectors])
        # best_coverage_bitvector = max(combined_coverage_bitvectors, key=lambda x: sum(x))
        combined_coverages = [sum(x) for x in combined_coverage_bitvectors]
        if temperature is None:
            best_coverage = max(combined_coverages)
            best_indices = [i for i in range(len(combined_coverages)) if combined_coverages[i] == best_coverage]
            best_index = random.choice(best_indices)
        else:
            # Softmax with normalization to prevent overflow
            combined_coverages_max = max(combined_coverages)
            coverages_for_softmax = [a_coverage - combined_coverages_max for a_coverage in combined_coverages]
            coverages_to_probabilities = np.exp(np.asarray(coverages_for_softmax) / temperature) / sum(
                np.exp(np.asarray(coverages_for_softmax) / temperature))

            best_index = np.random.choice(list(range(0, len(combined_coverages))),
                                                   p=coverages_to_probabilities)
        individual[constraint_index] = constraint_neighbourhood[best_index]
