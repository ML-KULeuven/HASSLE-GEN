from functools import partial
import numpy as np
from auxiliary import make_individual, to_phenotype, local_search_knowledge_compilation_based
from crossover import clause_crossover_1x, uniform_crossover, matched_uniform_crossover, uniform_clause_crossover, \
    scramble_clause_crossover, avoid_duplicate_clauses_scramble_clause_crossover, smart_clause_crossover_dispatch
from evaluation import evaluate, evaluate_use_infeasibility
from hassle_gen import HassleGen
from mutation import mutate_hardness, mutate_weight, mutate_clause, mutate_to_neighbour,\
    mutate_clause_smart, mutate_clause_smarter


def learn_max_sat_model(n,
                        k,
                        examples,
                        population_size,
                        tournament_size=None,
                        use_crowding=False,
                        crowding_variant="semantic_relative",
                        generations=10,
                        prob_crossover=0.9,
                        crossover_operators=None,
                        mutation_operators=None,
                        use_local_search=False,
                        cutoff_time=None,
                        cutoff_time_flags=None,
                        use_knowledge_compilation_caching=True,
                        knowledge_compilation_variant=4,
                        use_diagram_for_instance_evaluation=True,
                        conjunctive_contexts=0,
                        variable_absence_bias=1,
                        use_infeasibility=False,
                        use_clause_bitvector_cache=True,
                        observers=None):
    """
    Configures HASSLE-GEN as desired and subsequently runs the algorithm.
    :param n: The number of variables
    :param k: The number of constraints
    :param examples: A list of examples, where each example is a context-instance-label tuple
    :param population_size: The population size
    :param tournament_size: The tournament size - only relevant when not using crowding
    :param use_crowding: A Boolean that denotes whether to use crowding
    :param crowding_variant: What crowding variant to use: "semantic_relative", "semantic_absolute", "syntactic". Only
    relevant when use_crowding is True.
    :param generations: The maximum number of generations
    :param prob_crossover: The probability of applying a crossover operator on a pair of parents
    :param crossover_operators: A list of crossover operators to use. Each operator is itself a list of which the first
    element denotes the name of the crossover operator, and optional additional elements are used to configure the
    operator
    :param mutation_operators: A list of mutation operators to use. Each operator is itself a list of which the first
    element denotes the name of the mutation operator, and optional additional elements are used to configure the
    operator.
    :param use_local_search: Whether to apply a HASSLE-SLS local search step to each individual in each iteration
    :param cutoff_time: The maximum number of seconds to run the algorithm for
    :param cutoff_time_flags: A list of strings that denotes what operations should be counted in the runtime of the
    algorithm. Options are: "selection", "crossover", "mutation", "local search" and "evaluation". By default all
    are included.
    :param use_knowledge_compilation_caching: A Boolean that denotes whether to use knowledge-compilation-based caching
    :param knowledge_compilation_variant: What variant of knowledge-compilation-based caching to use. Options are:
    1, 2, 3 and 4. Only relevant when use_knowledge_compilation_caching is True.
    :param use_diagram_for_instance_evaluation: A Boolean that denotes whether to use the knowledge-compiled
    representation not just for computing the optimal value in a context, but also to evaluate the instances themselves.
    Only relevant when use_knowledge_compilation_caching is True.
    :param conjunctive_contexts: A Boolean that denotes whether the contexts occurring in the examples should be
    interpreted as conjunctions. If False, the contexts are interpreted as disjunctions.
    :param variable_absence_bias: A float that can be used to determine the length of constraints in the initial
    population. In initializing a model, in each constraint, each variable can occur as a negative
    literal, as a positive literal, or be absent from the constraint. With variable_absence_bias = 1, any variable
    has the same probability of occurring (either positively or negatively) as being absent in any constraint. With
    variable_absence_bias = 2, any variable is twice as likely to be absent as to occur, in any constraint.
    :param use_infeasibility: A Boolean that denotes whether to make use of the distinction between infeasible and
    suboptimal negative examples. If False, a negative example is simply interpreted as being negative, and no
    distinction between infeasibility and suboptimality is made.
    :param use_clause_bitvector_cache: A Boolean that denotes whether to cache coverage bitvectors computed during
    execution. Relevant when using smart mutation, smarter mutation or smart crossover.
    :param observers: A list of Observer objects to report aggregate data to in every generation.
    :return: A tuple consisting of (i) The best training set accuracy achieved, (ii) the phenotype that achieved said
    training set accuracy and (iii) the runtime of the algorithm (including only the runtime of the operations
    specified in cutoff_time_flags).
    """

    if not use_crowding and tournament_size is None:
        raise ValueError("Specify a tournament size when not using crowding")

    ga = HassleGen(
        partial(make_individual, n, k, variable_absence_bias=variable_absence_bias),
        partial(evaluate, examples=examples, use_knowledge_compilation_caching=use_knowledge_compilation_caching,
                knowledge_compilation_variant=knowledge_compilation_variant,
                use_diagram_for_instance_evaluation=use_diagram_for_instance_evaluation,
                conjunctive_contexts=conjunctive_contexts) if use_infeasibility == False else
        partial(evaluate_use_infeasibility, examples=examples,
                use_knowledge_compilation_caching=use_knowledge_compilation_caching,
                knowledge_compilation_variant=knowledge_compilation_variant,
                use_diagram_for_instance_evaluation=use_diagram_for_instance_evaluation,
                conjunctive_contexts=conjunctive_contexts),
        population_size=population_size,
        prob_crossover=prob_crossover,
        use_crowding=use_crowding,
        crowding_variant=crowding_variant,
        tournament_size=tournament_size
    )

    # Make smart clause mutation and smart clause crossover more efficient by caching
    if use_clause_bitvector_cache:
        clause_bitvector_cache = dict()
    else:
        clause_bitvector_cache = None

    # Add crossover operators
    if crossover_operators is None:
        # Use defaults
        ga.add_crossover(scramble_clause_crossover)
        ga.add_crossover(uniform_crossover)
    else:
        # Use the requested operators
        for crossover_operator in crossover_operators:
            if crossover_operator[0] == "clause_crossover_1x":
                ga.add_crossover(clause_crossover_1x)
            elif crossover_operator[0] == "uniform_crossover":
                ga.add_crossover(uniform_crossover)
            elif crossover_operator[0] == "matched_uniform_crossover":
                ga.add_crossover(matched_uniform_crossover)
            elif crossover_operator[0] == "uniform_clause_crossover":
                ga.add_crossover(uniform_clause_crossover)
            elif crossover_operator[0] == "scramble_clause_crossover":
                ga.add_crossover(scramble_clause_crossover)
            elif crossover_operator[0] == "avoid_duplicate_clauses_scramble_clause_crossover":
                ga.add_crossover(avoid_duplicate_clauses_scramble_clause_crossover)
            elif crossover_operator[0] == "smart_clause_crossover":
                greedy = crossover_operator[1]
                probability_variant = crossover_operator[2]
                temperature = crossover_operator[3]
                ga.add_crossover(partial(smart_clause_crossover_dispatch, examples=examples, greedy=greedy,
                                         probability_variant=probability_variant, temperature=temperature,
                                         clause_bitvector_cache=clause_bitvector_cache,
                                         use_infeasibility=use_infeasibility))
                # With smart clause crossover, 2 parents produce only a single offspring
                ga.single_offspring_per_parent_couple = True
            else:
                raise Exception("A valid crossover operator should be chosen")

    # Add mutation operators
    if mutation_operators is None:
        # Use defaults
        ga.add_mutation(mutate_hardness, trigger_prob=1, inner_prob=0.05)
        ga.add_mutation(mutate_clause, trigger_prob=1, inner_prob=0.05)
        ga.add_mutation(mutate_weight, trigger_prob=1, inner_prob=0.05)
    else:
        # Use the requested operators
        for mutation_operator in mutation_operators:
            if mutation_operator[0] == "mutate_hardness":
                trigger_prob = mutation_operator[1]
                inner_prob = mutation_operator[2]
                ga.add_mutation(partial(mutate_hardness, trigger_prob=trigger_prob, inner_prob=inner_prob))
            elif mutation_operator[0] == "mutate_clause":
                trigger_prob = mutation_operator[1]
                inner_prob = mutation_operator[2]
                ga.add_mutation(partial(mutate_clause, trigger_prob=trigger_prob, inner_prob=inner_prob))
            elif mutation_operator[0] == "mutate_weight":
                trigger_prob = mutation_operator[1]
                inner_prob = mutation_operator[2]
                ga.add_mutation(partial(mutate_weight, trigger_prob=trigger_prob, inner_prob=inner_prob))
            elif mutation_operator[0] == "mutate_to_neighbour":
                trigger_prob = mutation_operator[1]
                contexts = [e[0] for e in examples]
                data = np.array([e[1] for e in examples])
                labels = [e[2] for e in examples]
                boolean_labels = np.array([True if l == 1 else False for l in labels])
                if use_infeasibility:
                    inf = [True if l == -1 else False for l in labels]
                else:
                    inf = None
                ga.add_mutation(partial(mutate_to_neighbour, trigger_prob=trigger_prob, contexts=contexts, data=data,
                                        boolean_labels=boolean_labels, inf=inf))
            elif mutation_operator[0] == "mutate_clause_smart":
                trigger_prob = mutation_operator[1]
                temperature = None
                if len(mutation_operator) > 2:
                    temperature = mutation_operator[2]
                ga.add_mutation(partial(mutate_clause_smart, trigger_prob=trigger_prob, examples=examples,
                                        clause_bitvector_cache=clause_bitvector_cache,
                                        use_infeasibility=use_infeasibility,
                                        temperature=temperature))
            elif mutation_operator[0] == "mutate_clause_smarter":
                trigger_prob = mutation_operator[1]
                temperature = None
                if len(mutation_operator) > 2:
                    temperature = mutation_operator[2]
                ga.add_mutation(partial(mutate_clause_smarter, trigger_prob=trigger_prob, examples=examples,
                                        clause_bitvector_cache=clause_bitvector_cache,
                                        use_infeasibility=use_infeasibility,
                                        temperature=temperature))
            else:
                raise Exception("A valid mutation operator should be chosen")

    # Add method that converts genotype to phenotype representation
    ga.to_phenotype = to_phenotype

    if use_local_search:
        # Perform a local search step to every individual in every generation
        if use_infeasibility:
            contexts = [e[0] for e in examples]
            data = np.array([e[1] for e in examples])
            labels = [e[2] for e in examples]
            inf = [True if l == -1 else False for l in labels]
        else:
            inf = None
        ga.local_search_function = partial(local_search_knowledge_compilation_based, examples,
                                           conjunctive_contexts=conjunctive_contexts, inf=inf)

    if observers is not None:
        # Add observers
        for observer in observers:
            ga.observers.append(observer)

    # Run HASSLE-GEN
    return ga.run(generations, 1, cutoff_time, cutoff_time_flags)
