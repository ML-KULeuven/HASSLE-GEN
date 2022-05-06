import copy
import random
import hassle_sls.local_search
import numpy as np
from evaluation import evaluate_knowledge_compilation_based_dispatch
from hassle_sls import maxsat


def make_individual(n, k, variable_absence_bias=1):
    """
    Generates a new individual
    :param n: The number of variables
    :param k: The numnber of constraints
    :param variable_absence_bias: Determines how much more likely it is for a variable to be absent from a generated
    clause rather than present (positively or negatively). A value of 1 makes it equally likely for a variable to be
    present than to be absent.
    """
    clauses = []
    for i in range(k):
        literals = [random.choices([-1, 0, 1], weights=[0.5, variable_absence_bias, 0.5])[0] for _ in range(n)]

        hard = random.random() < 0.5
        w = random.random()

        clauses.append(literals + [hard, w])

    return clauses


def to_phenotype(individual):
    """
    Returns the phenotype representation of a given individual's genotype
    :param individual: The individual's genotype representation
    :return: The individual's phenotype representation
    """
    model = []
    for weighted_clause in individual:
        w = None if weighted_clause[-2] else weighted_clause[-1]
        clause = set()
        for j in range(len(weighted_clause) - 2):
            if weighted_clause[j] < 0:
                clause.add(-j - 1)
            elif weighted_clause[j] > 0:
                clause.add(j + 1)
        model.append((w, clause))
    return model


def local_search_knowledge_compilation_based(examples, iterations, individual, conjunctive_contexts=0, inf=None):
    """
    Performs a given amount of local search iterations of HASSLE-SLS on a given individual, using
    knowledge-compilation based caching in evaluation, using variant 4 and also using the compiled diagrams for
    instance evaluation, rather than merely for computing the optimal values
    :param examples: A list of examples, where each example is a context-instance-label tuple
    :param iterations: The number of iterations
    :param individual: The individual
    :param conjunctive_contexts: A Boolean that denotes whether the contexts occurring in the examples should be
    interpreted as conjunctions. If False, the contexts are interpreted as disjunctions.
    :param inf: Optional - a list with as many entries as there are examples. Each entry is a Boolean value that denotes
    whether said value's label is 'infeasible' or not. Only relevant when the algorithm should make a distinction
    between infeasible and suboptimal negative examples
    :return: The individual resulting from the given number of local search iterations
    """
    if iterations == 0:
        return individual
    model = genotype_to_MaxSAT(individual)
    scores = []
    best_scores = []
    seed = int(random.random() * 10 ** 8)
    rng = np.random.RandomState(seed)

    contexts = [e[0] for e in examples]
    data = np.array([e[1] for e in examples])
    labels = [e[2] for e in examples]
    boolean_labels = np.array([True if l == 1 else False for l in labels])
    examples_in_HASSLE_SLS_representation = [(context, instance, boolean_label) for (context, instance, boolean_label) in list(zip(contexts, data, boolean_labels))]
    #(score, _) = evaluate_knowledge_compilation_based_dispatch(to_phenotype(individual), examples, conjunctive_contexts=conjunctive_contexts, inf=inf)
    #scores.append(score)
    score = -1
    solution = model.deep_copy()
    best_score = score
    best_corr_pred_bitvect = None
    for i in range(iterations):
        index = hassle_sls.local_search.random_incorrect_example_index(model, data, contexts, boolean_labels, infeasible=inf, rng=rng, conjunctive_contexts=conjunctive_contexts)
        clause_len = len(individual[0]) - 2
        neighbours = model.get_neighbours(
            data[index],
            contexts[index],
            boolean_labels[index],
            clause_len,
            rng,
            infeasible=inf,
            conjunctive_contexts=conjunctive_contexts
        )
        if len(neighbours) == 0:
            continue

        model, score, corr_pred_bitvect = compute_best_neighbour_knowledge_compilation(neighbours, examples_in_HASSLE_SLS_representation, conjunctive_contexts=conjunctive_contexts, inf=inf)
        scores.append(score)

        if score > best_score:
            solution = model.deep_copy()
            best_score = score
            best_corr_pred_bitvect = corr_pred_bitvect
        best_scores.append(best_score)

    return MaxSAT_to_genotype(solution), best_score, best_corr_pred_bitvect


def MaxSAT_to_genotype(model):
    """
    Takes a MAX-SAT model as an object of HASSLE's MaxSAT class and returns an equivalent model
    in the genotype representation used in the evolutionary algorithm
    :param model The MAX-SAT model as an object of HASSLE's MaxSAT class
    :return The genotype representation of the given MAX-SAT model
    """
    model_as_individual = []
    for j in range(model.k):
        model_as_individual.append(model.l[j] + [True if model.c[j] == 1 else False] + [model.w[j]])
    return model_as_individual


def genotype_to_MaxSAT(model):
    """
    Takes a MAX-SAT model in the genotype representation used in the evolutionary algorithm
    and returns an equivalent model as an object of HASSLE's MaxSAT class
    :param model The MAX-SAT model in the genotype representation used in the evolutionary algorithm
    :return The given MAX-SAT model as an object of HASSLE's MaxSAT class
    """
    hardness = [1 if clause[-2] == True else 0 for clause in model]
    weights = [clause[-1] for clause in model]
    clauses = [clause[:-2] for clause in model]
    return maxsat.MaxSAT(hardness, weights, clauses)


def phenotype_to_MaxSAT(model, n):
    """
    Takes a MAX-SAT model in the phenotype representation used in the evolutionary algorithm
    and returns an equivalent model as an object of HASSLE's MaxSAT class
    :param model The MAX-SAT model in the phenotype representation used in the evolutionary algorithm
    :param n The number of variables
    :return The given MAX-SAT model as an object of HASSLE's MaxSAT class
    """
    hardness = [1 if clause[0] is None else 0 for clause in model]
    weights = [clause[0] if clause[0] is not None else 0.5 for clause in model]
    clauses = []
    for phenotype_clause in model:
        literals_in_phenotype_clause = phenotype_clause[1]
        clause = [0] * n
        for literal in literals_in_phenotype_clause:
            if literal > 0:
                clause[abs(literal) - 1] = 1
            else:
                clause[abs(literal) - 1] = -1
        clauses.append(clause)
    return maxsat.MaxSAT(hardness, weights, clauses)

def rank_neigbours_knowledge_compilation(neighbours, examples, knowledge_compilation_variant=4, use_diagram_for_instance_evaluation=True, conjunctive_contexts=0, inf=None):
    """
    Ranks a given collection of neighbours according to their training set accuracy
    :param neighbours: The collection of neighbours
    :param examples: A list of examples, where each example is a context-instance-label tuple
    :param knowledge_compilation_variant: What variant of knowledge-compilation-based caching to use. Options are:
    1, 2, 3 and 4.
    :param use_diagram_for_instance_evaluation: A Boolean that denotes whether to use the knowledge-compiled
    representation not just for computing the optimal value in a context, but also to evaluate the instances themselves.
    :param conjunctive_contexts: A Boolean that denotes whether the contexts occurring in the examples should be
    interpreted as conjunctions. If False, the contexts are interpreted as disjunctions.
    :param inf: Optional - a list with as many entries as there are examples. Each entry is a Boolean value that denotes
    whether said value's label is 'infeasible' or not. Only relevant when the algorithm should make a distinction
    between infeasible and suboptimal negative examples
    :return: A tuple consisting of (i) the list of neighbours, ranked descending training set accuracy, (ii) a list
    with the respective training set accuracies and (iii) a list of accuracy bitvectors, where each bitvector consists
    of as many entries as there are examples, and each entry denotes whether the individual correctly labeled the
    corresponding example.
    """
    # Shuffle neighbours for randomness in how neighbours with equal score are ranked
    neighbours_copy = copy.deepcopy(neighbours)
    random.shuffle(neighbours_copy)

    # First cast the objects of the MaxSAT class to their equivalent genotype representations
    neighbours_as_genotype = [MaxSAT_to_genotype(neighbour) for neighbour in neighbours_copy]

    scores = []
    corr_pred_bitvectors = []

    for neighbour in neighbours_as_genotype:
        (score, correctly_predicted_bitvector) = evaluate_knowledge_compilation_based_dispatch(to_phenotype(neighbour), examples,
                                                                                               knowledge_compilation_variant=knowledge_compilation_variant,
                                                                                               use_diagram_for_instance_evaluation=use_diagram_for_instance_evaluation,
                                                                                               conjunctive_contexts=conjunctive_contexts, inf=inf)
        scores.append(score)
        corr_pred_bitvectors.append(correctly_predicted_bitvector)

    sort_permutation = np.argsort(scores)[::-1]

    lst_models = [neighbours_copy[i] for i in sort_permutation]
    lst_scores = [scores[i] for i in sort_permutation]
    lst_correct_examples = [corr_pred_bitvectors[i] for i in sort_permutation]
    return lst_models, lst_scores, lst_correct_examples


def compute_best_neighbour_knowledge_compilation(neighbours, examples, conjunctive_contexts=0, inf=None):
    """
    Computes the best neighbour out of a set of neighbours, using knowledge-compilation based caching in evaluation,
    using variant 4 and also using the compiled diagrams for instance evaluation, rather than merely for computing the
    optimal values
    :param neighbours: The collection of neighbours
    :param examples: A list of examples, where each example is a context-instance-label tuple
    :param conjunctive_contexts: A Boolean that denotes whether the contexts occurring in the examples should be
    interpreted as conjunctions. If False, the contexts are interpreted as disjunctions.
    :param inf: Optional - a list with as many entries as there are examples. Each entry is a Boolean value that denotes
    whether said value's label is 'infeasible' or not. Only relevant when the algorithm should make a distinction
    between infeasible and suboptimal negative examples
    :return: A tuple containing (i) the best neighbour, (ii) its training set accuracy and (iii) an accuracy bitvector
    with as many entries as there are examples, where each entry denotes whether the individual correctly labeled the
    corresponding example.
    """
    # First cast the objects of the MaxSAT class to their equivalent genotype representations
    neighbours_as_genotype = [MaxSAT_to_genotype(neighbour) for neighbour in neighbours]

    # Compute the best neighbour using knowledge compilation based evaluation
    max_score = -1
    best = None
    corresponding_corr_pred_bitvect = None
    for neighbour in neighbours_as_genotype:
        (score, correctly_predicted_bitvector) = evaluate_knowledge_compilation_based_dispatch(to_phenotype(neighbour), examples, conjunctive_contexts=conjunctive_contexts, inf=inf)
        if score > max_score:
            max_score = score
            best = neighbour
            corresponding_corr_pred_bitvect = correctly_predicted_bitvector

    # Cast the best neighbour back to the equivalent MaxSAT representation
    best_as_MaxSAT = genotype_to_MaxSAT(best)

    return best_as_MaxSAT, max_score, corresponding_corr_pred_bitvect


def compute_clause_coverage_bitvectors(model, examples, clause_bitvector_cache=None, use_infeasibility=False):
    """
    For a given model and set of examples, computes the coverage bitvector of each of the model's clauses with respect
    to the examples. Note: the coverage bitvector is not the same as the accuracy bitvector. A coverage bitvector
    is associated with a clause, and an approximation of how well that clause 'matches' each of the labeled examples. An
    accuracy bitvector is associated with an individual and is an exact representation of that individual's training set
    accuracy.
    :param model: The MAX-SAT model to evaluate, in genotype representation
    :param examples: A list of examples, where each example is a context-instance-label tuple
    :param clause_bitvector_cache: Optional - a dictionary in which to cache clause coverages
    :param use_infeasibility: A Boolean that denotes whether to make use of the distinction between infeasible and
    suboptimal negative examples. If False, a negative example is simply interpreted as being negative, and no
    distinction between infeasibility and suboptimality is made.
    :return: A list of respective coverage bitvectors, one for each clause in the individual
    """
    coverage_bitvector_list = []
    for constraint in model:
        coverage_bitvector = compute_clause_coverage_bitvector(constraint, examples,
                                                               clause_bitvector_cache=clause_bitvector_cache,
                                                               use_infeasibility=use_infeasibility)
        coverage_bitvector_list.append(coverage_bitvector)
    return coverage_bitvector_list


def compute_clause_coverage_bitvector(constraint, examples, clause_bitvector_cache=None, use_infeasibility=False):
    """
    Computes the coverage bitvector of a given constraint with respect to the examples. Note: the coverage bitvector is
    not the same as the accuracy bitvector. A coverage bitvector is associated with a clause, and an approximation of
    how well that clause 'matches' each of the labeled examples. An accuracy bitvector is associated with an individual
    and is an exact representation of that individual's training set accuracy.
    :param constraint: The constraint
    :param examples: A list of examples, where each example is a context-instance-label tuple
    :param clause_bitvector_cache: Optional - a dictionary in which to cache clause coverages
    :param use_infeasibility: A Boolean that denotes whether to make use of the distinction between infeasible and
    suboptimal negative examples. If False, a negative example is simply interpreted as being negative, and no
    distinction between infeasibility and suboptimality is made.
    :return: The coverage bitvector
    """
    if clause_bitvector_cache is not None and tuple(constraint[:-1]) in clause_bitvector_cache:
        return clause_bitvector_cache[tuple(constraint[:-1])]
    else:
        clause = constraint[:-2]
        coverage_bitvector = [0] * len(examples)
        for k in range(len(examples)):
            example = examples[k]
            instance = example[1]
            label = example[2]

            # We assume a positvely labeled example is always satisfied by its context, regardless of its label
            if label == 1:
                covered = False
                for i in range(len(clause)):
                    literal = clause[i]
                    if (literal == 1 and instance[i] == True) or (literal == -1 and instance[i] == False):
                        covered = True
            else:
                if use_infeasibility:
                    if label == 0:
                        # suboptimal
                        if constraint[-2] == True:
                            # hard
                            covered = False
                            for i in range(len(clause)):
                                literal = clause[i]
                                if (literal == 1 and instance[i] == True) or (literal == -1 and instance[i] == False):
                                    covered = True
                        else:
                            # soft
                            covered = True
                            for i in range(len(clause)):
                                literal = clause[i]
                                if (literal == 1 and instance[i] == True) or (literal == -1 and instance[i] == False):
                                    covered = False
                    if label == -1:
                        # infeasible
                        if constraint[-2] == True:
                            # hard
                            covered = True
                            for i in range(len(clause)):
                                literal = clause[i]
                                if (literal == 1 and instance[i] == True) or (literal == -1 and instance[i] == False):
                                    covered = False
                        else:
                            # soft
                            covered = True
                else:
                    covered = True
                    for i in range(len(clause)):
                        literal = clause[i]
                        if (literal == 1 and instance[i] == True) or (literal == -1 and instance[i] == False):
                            covered = False
            if covered:
                coverage_bitvector[k] = 1

        if clause_bitvector_cache is not None:
            clause_bitvector_cache[tuple(constraint[:-1])] = coverage_bitvector
        return coverage_bitvector


def combine_coverage_bitvectors(bitvector1, bitvector2, examples):
    """
    Combines two bitvectors denoting clause-examples coverage, using information about the examples.
    This function does not make use of the distinction between suboptimal and infeasible negative examples.
    Concretely, the combination of two clauses covers a positively labeled examples if both clauses cover it.
    The combination of two clauses covers a negative example if at least one of them covers it. Note that the meaning
    of 'cover' can spark some confusion. Here, to 'cover' means to be accordance with, or to explain, the example. So,
    a clause covers a negatively labeled example, if the example is **not** satisfied by the clause.
    :param bitvector1: The first coverage bitvector
    :param bitvector2: The second coverage bitvector
    :param examples: A list of examples, where each example is a context-instance-label tuple
    :return The combined coverage bitvector
    """
    labels = [example[2] for example in examples]
    combined_bitvector = [0] * len(bitvector1)

    for i in range(len(bitvector1)):
        if labels[i] == 1:
            combined_bitvector[i] = bitvector1[i] & bitvector2[i]
        else:
            combined_bitvector[i] = bitvector1[i] | bitvector2[i]

    return combined_bitvector


def combine_coverage_bitvectors_hard_constraints(bitvector1, bitvector2, examples):
    """
    Combines two bitvectors denoting clause-example coverage for two hard constraints, using information about the
    examples. This function makes use of the distinction between suboptimal and infeasible negative examples.
    :param bitvector1: The first coverage bitvector (of a hard constraint)
    :param bitvector2: The second coverage bitvector (of a hard constraint)
    :param examples: A list of examples, where each example is a context-instance-label tuple
    :return The combined coverage bitvector
    """
    labels = [example[2] for example in examples]
    combined_bitvector = [0] * len(bitvector1)

    for i in range(len(bitvector1)):
        if labels[i] == 1:
            combined_bitvector[i] = bitvector1[i] & bitvector2[i]
        elif labels[i] == 0:
            # suboptimal
            combined_bitvector[i] = bitvector1[i] & bitvector2[i]
        elif labels[i] == -1:
            # infeasible
            combined_bitvector[i] = bitvector1[i] | bitvector2[i]
    return combined_bitvector


def combine_coverage_bitvectors_soft_constraints(bitvector1, bitvector2, examples):
    """
    Combines two bitvectors denoting clause-example coverage for two soft constraints, using information about the
    examples. This function makes use of the distinction between suboptimal and infeasible negative examples.
    :param bitvector1: The first coverage bitvector (of a soft constraint)
    :param bitvector2: The second coverage bitvector (of a soft constraint)
    :param examples: A list of examples, where each example is a context-instance-label tuple
    :return The combined coverage bitvector
    """
    labels = [example[2] for example in examples]
    combined_bitvector = [0] * len(bitvector1)

    for i in range(len(bitvector1)):
        if labels[i] == 1:
            combined_bitvector[i] = bitvector1[i] & bitvector2[i]
        elif labels[i] == 0:
            # suboptimal
            combined_bitvector[i] = bitvector1[i] | bitvector2[i]
        elif labels[i] == -1:
            # infeasible
            combined_bitvector[i] = bitvector1[i] & bitvector2[i]
            if combined_bitvector[i] == False:
                print("Error!")
                #raise Exception("Error!")
    return combined_bitvector


def combine_coverage_bitvectors_hard_and_soft_constraints(bitvector1, bitvector2, examples):
    """
    Combines a coverage bitvector that represents the combination of several hard constraints with the coverage
    bitvector that represents the combination of several hard constraints, using information about the examples.
    This function makes use of the distinction between suboptimal and infeasible negative examples.
    :param bitvector1: A coverage bitvector that represents all hard constraints combined
    :param bitvector2: A coverage bitvector that represents all soft constraints combined
    :param examples: A list of examples, where each example is a context-instance-label tuple
    :return The combined coverage bitvector
    """
    combined_bitvector = [0] * len(bitvector1)

    for i in range(len(bitvector1)):
        combined_bitvector[i] = bitvector1[i] & bitvector2[i]

    return combined_bitvector
