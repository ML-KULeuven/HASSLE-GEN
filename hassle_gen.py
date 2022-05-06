import random
import time
from copy import deepcopy
from genetic import get_absolute_semantic_distance, get_relative_semantic_distance, get_syntactic_distance
from genetic import GeneticAlgorithm


class HassleGen(GeneticAlgorithm):
    """
    The HASSLE-GEN algorithm. Has various parameters that can be set to configure the algorithm.
    """
    def __init__(self,
                 individual_factory=None,
                 score_function=None,
                 population_size=10,
                 tournament_size=None,
                 prob_crossover=0.9,
                 single_offspring_per_parent_couple=False,
                 use_crowding=False,
                 crowding_variant="semantic_relative"):
        """
        :param individual_factory: The function to use for initialising MAX-SAT models
        :param score_function: The function to use to evaluating MAX-SAT models
        :param population_size: The population size
        :param tournament_size: The tournament size - only relevant when not using crowding
        :param prob_crossover: The probability of applying a crossover operator on a pair of parents
        :param single_offspring_per_parent_couple: A Boolean that denotes whether crossover only produces a single
        unique individual or two distinct individuals. Should be set to True when using a crossover operator that
        produces only a single new individual.
        :param use_crowding: A Boolean that denotes whether to use crowding
        :param crowding_variant: What crowding variant to use: "semantic_relative", "semantic_absolute", "syntactic".
        Only relevant when use_crowding is True.
        """
        super().__init__(population_size, individual_factory, score_function)

        if not use_crowding and tournament_size is None:
            raise ValueError("Specify a tournament size when not using crowding")

        self.prob_crossover = prob_crossover
        self.local_search_function = None
        self.single_offspring_per_parent_couple = single_offspring_per_parent_couple
        self.tournament_size = tournament_size
        self.use_crowding = use_crowding
        self.crowding_variant = crowding_variant
        self.p_replace_anyway = 0

    def run(self, generations, max_score=None, cutoff_time=None, cutoff_time_flags=None):
        """
        Runs the genetic algorithm
        :param generations: The maximum number of generations
        :param max_score: The maximum training set accuracy
        :param cutoff_time: The maximum number of seconds to run the algorithm for
        :param cutoff_time_flags: A list of strings that denotes what operations should be counted in the runtime of the
        algorithm. Options are: "selection", "crossover", "mutation", "local search" and "evaluation". By default all
        are included.
        :return: A tuple consisting of (i) The best training set accuracy achieved, (ii) the phenotype that achieved
        said training set accuracy and (iii) the runtime of the algorithm (including only the runtime of the operations
        specified in cutoff_time_flags).
        """

        cumulative_time = 0

        # Initialize population
        time_point = time.time()
        self.init_population()
        initialisation_time = time.time() - time_point

        # Compute metrics for observer(s)
        best_score = max(self.scores)
        best_index = self.scores.index(best_score)
        best_phenotype = self.to_phenotype(self.population[best_index])
        worst_score = min(self.scores)
        avg_score = sum(self.scores) / len(self.scores)
        num_distinct_individuals = self.get_number_of_distinct_individuals()
        num_distinct_clauses = self.get_number_of_distinct_clauses()
        num_distinct_clauses_per_individual = self.get_number_of_distinct_clauses_per_individual()
        syntactic_population_diversity = self.get_syntactic_population_diversity()
        semantic_population_diversity = self.get_absolute_semantic_population_diversity()
        semantic_population_diversity_alternative = self.get_relative_semantic_population_diversity()
        example_coverage_vector = self.get_example_coverage_vector()
        cumulative_time += initialisation_time

        # Store computed metrics in observer(s)
        for observer in self.observers:
            observer.observe_generation(
                0,
                best_score,
                worst_score=worst_score,
                avg_score=avg_score,
                num_distinct_individuals=num_distinct_individuals,
                num_distinct_clauses=num_distinct_clauses,
                num_distinct_clauses_per_individual=num_distinct_clauses_per_individual,
                syntactic_population_diversity=syntactic_population_diversity,
                example_coverage_vector=example_coverage_vector,
                semantic_population_diversity=semantic_population_diversity,
                semantic_population_diversity_alternative=semantic_population_diversity_alternative,
                gen_duration=initialisation_time,
                selection_duration=0,
                crossover_duration=0,
                mutation_duration=0,
                local_search_duration=0,
                evaluation_duration=initialisation_time
            )

        print(f"gen {0} ({initialisation_time}) - {best_score}")

        # Genetic algorithm loop
        for gen in range(generations):
            # Initialising timers
            selection_time = 0
            crossover_time = 0
            mutation_time = 0
            local_search_time = 0
            evaluation_time = 0

            # Generating new generation
            population = []
            scores = []
            correctly_predicted_bitvectors = []

            def add_to_new_gen(new_ind, new_score, new_corr_pred_bitvector):
                # This function helps to keep the code below somewhat readable
                population.append(new_ind)
                scores.append(new_score)
                correctly_predicted_bitvectors.append(new_corr_pred_bitvector)

            if self.use_crowding:
                indices_to_pair = set(range(self.population_size))
                # No explicit elitism needed here - elitism is implicit in crowding
            else:
                # Elitism
                best_score = max(self.scores)
                best_index = self.scores.index(best_score)
                elite_ind, elite_score, elite_corr_pred_bitvector = self.select(best_index, copy=True)
                add_to_new_gen(elite_ind, elite_score, elite_corr_pred_bitvector)

            while len(population) < self.population_size:
                if self.use_crowding:
                    # Pairing for deterministic crowding
                    time_point = time.time()
                    ind1_index, ind2_index = random.sample(indices_to_pair, 2)
                    indices_to_pair.remove(ind1_index)
                    indices_to_pair.remove(ind2_index)
                    parent1, parent1_score, parent1_corr_pred_bitvector = self.select(ind1_index, copy=True)
                    parent2, parent2_score, parent2_corr_pred_bitvector = self.select(ind2_index, copy=True)
                    ind1 = deepcopy(parent1)
                    ind2 = deepcopy(parent2)
                    selection_time += time.time() - time_point
                else:
                    # Tournament selection
                    time_point = time.time()
                    ind1_index = self.tournament_select(self.tournament_size, return_index=True)
                    ind2_index = self.tournament_select(self.tournament_size, return_index=True,
                                                        index_to_exclude=ind1_index)
                    ind1, _, _ = self.select(ind1_index, copy=True)
                    ind2, _, _ = self.select(ind2_index, copy=True)
                    selection_time += time.time() - time_point

                # Crossover
                time_point = time.time()
                if random.random() < self.prob_crossover:
                    if ind1 != ind2:
                        self.crossover(ind1, ind2)
                crossover_time += time.time() - time_point
                if not self.use_crowding or self.single_offspring_per_parent_couple:
                    ind = ind1 if random.random() < 0.5 else ind2

                # Mutation
                time_point = time.time()
                if not self.use_crowding or self.single_offspring_per_parent_couple:
                    self.mutate(ind)
                else:
                    self.mutate(ind1)
                    self.mutate(ind2)
                mutation_time += time.time() - time_point

                # Local Search
                if self.local_search_function is not None and False:
                    time_point = time.time()
                    iterations = 1
                    if not self.use_crowding or self.single_offspring_per_parent_couple:
                        ind, _, _ = self.local_search_function(iterations, ind)
                    else:
                        ind1, _, _ = self.local_search_function(iterations, ind1)
                        ind2, _, _ = self.local_search_function(iterations, ind2)
                    local_search_time += time.time() - time_point

                if self.use_crowding:
                    # Survivor selection using crowding
                    time_point = time.time()
                    if self.single_offspring_per_parent_couple:
                        # Single offspring per parent couple
                        self.crowding_replacement_single_offspring(add_to_new_gen, ind, parent1, parent1_score,
                                                                   parent1_corr_pred_bitvector, parent2, parent2_score,
                                                                   parent2_corr_pred_bitvector, best_score)

                    else:
                        # Two offspring per parent couple
                        self.crowding_replacement_two_offspring(add_to_new_gen, ind1, ind2, parent1, parent1_score,
                                                                parent1_corr_pred_bitvector, parent2, parent2_score,
                                                                parent2_corr_pred_bitvector, best_score)
                    evaluation_time += time.time() - time_point
                else:
                    # No crowding
                    time_point = time.time()
                    ind_score, ind_corr_pred_bitvector = self.compute_scores([ind])
                    ind_score, ind_corr_pred_bitvector = ind_score[0], ind_corr_pred_bitvector[0]
                    add_to_new_gen(ind, ind_score, ind_corr_pred_bitvector)
                    evaluation_time += time.time() - time_point

            # Setting the new population
            self.replace_population(population, scores, correctly_predicted_bitvectors)

            # Compute metrics for observer(s)
            prev_best_score = best_score
            prev_best_phenotype = best_phenotype
            best_score = max(self.scores)
            best_index = self.scores.index(best_score)
            best_phenotype = self.to_phenotype(self.population[best_index])
            worst_score = min(self.scores)
            avg_score = sum(self.scores) / len(self.scores)
            num_distinct_individuals = self.get_number_of_distinct_individuals()
            num_distinct_clauses = self.get_number_of_distinct_clauses()
            num_distinct_clauses_per_individual = self.get_number_of_distinct_clauses_per_individual()
            syntactic_population_diversity = self.get_syntactic_population_diversity()
            semantic_population_diversity = self.get_absolute_semantic_population_diversity()
            semantic_population_diversity_alternative = self.get_relative_semantic_population_diversity()
            example_coverage_vector = self.get_example_coverage_vector()

            # Local search
            if best_score == prev_best_score and self.local_search_function is not None:
                print("Trying local optimization of best individual")
                continue_optimizing = True
                time_point = time.time()
                while continue_optimizing:
                    iterations = 1
                    best_individual = self.population[best_index]
                    individual_opt, individual_opt_score, individual_opt_corr_pred_bitvect = self.local_search_function(
                        iterations, best_individual)
                    print(f"Score of best neighbour: {individual_opt_score}")
                    if individual_opt_score > best_score:
                        print("Best individual replaced")
                        self.population[best_index] = individual_opt
                        self.scores[best_index] = individual_opt_score
                        self.correctly_predicted_bitvectors[best_index] = individual_opt_corr_pred_bitvect
                        best_score = individual_opt_score
                    else:
                        print("No improvement")
                        continue_optimizing = False
                local_search_time += time.time() - time_point

            # Update cumulative time, keeping timing flags in mind
            lap_time = selection_time + crossover_time + mutation_time + local_search_time + evaluation_time
            if cutoff_time_flags is None:
                cumulative_time_update = lap_time
            else:
                cumulative_time_update = 0
                if "selection" in cutoff_time_flags:
                    cumulative_time_update += selection_time
                if "crossover" in cutoff_time_flags:
                    cumulative_time_update += crossover_time
                if "mutation" in cutoff_time_flags:
                    cumulative_time_update += mutation_time
                if "local search" in cutoff_time_flags:
                    cumulative_time_update += local_search_time
                if "evaluation" in cutoff_time_flags:
                    cumulative_time_update += evaluation_time
            cumulative_time += cumulative_time_update

            # Store computed metrics in observer(s)
            for observer in self.observers:
                observer.observe_generation(
                    gen + 1,
                    best_score,
                    worst_score=worst_score,
                    avg_score=avg_score,
                    num_distinct_individuals=num_distinct_individuals,
                    num_distinct_clauses=num_distinct_clauses,
                    num_distinct_clauses_per_individual=num_distinct_clauses_per_individual,
                    syntactic_population_diversity=syntactic_population_diversity,
                    example_coverage_vector=example_coverage_vector,
                    semantic_population_diversity=semantic_population_diversity,
                    semantic_population_diversity_alternative=semantic_population_diversity_alternative,
                    gen_duration=lap_time,
                    selection_duration=selection_time,
                    crossover_duration=crossover_time,
                    mutation_duration=mutation_time,
                    local_search_duration=local_search_time,
                    evaluation_duration=evaluation_time
                )

            print(f"gen {gen + 1} ({cumulative_time}) - {best_score}")

            # Additional stop condition
            if cutoff_time is not None and cumulative_time >= cutoff_time:
                # In case the current best score was attained after the cutoff_time expired,
                # use the last best score and best phenotype attained within the cutoff_time for reporting
                # and subtract the last time update from the cumulative time, as individuals generated in this update
                # are not considered in the best score and best phenotype reporting
                best_score, best_phenotype = prev_best_score, prev_best_phenotype
                cumulative_time -= cumulative_time_update
                break

            # Additional stop condition
            # if max_score is not None and best_score >= max_score:
            #     break

        return best_score, best_phenotype, cumulative_time

    def crowding_replacement_single_offspring(self, add_to_new_gen, ind,
                                              parent1, parent1_score, parent1_corr_pred_bitvector,
                                              parent2, parent2_score, parent2_corr_pred_bitvector,
                                              best_score):
        ind_score, ind_corr_pred_bitvector = self.compute_scores([ind])
        ind_score, ind_corr_pred_bitvector = ind_score[0], ind_corr_pred_bitvector[0]
        # Compute the distances depending on the distance metric chosen
        if self.crowding_variant == "semantic_absolute":
            pairing1_dist = get_absolute_semantic_distance(ind_corr_pred_bitvector, parent1_corr_pred_bitvector)
            pairing2_dist = get_absolute_semantic_distance(ind_corr_pred_bitvector, parent2_corr_pred_bitvector)
        elif self.crowding_variant == "semantic_relative":
            pairing1_dist = get_relative_semantic_distance(ind_corr_pred_bitvector, parent1_corr_pred_bitvector)
            pairing2_dist = get_relative_semantic_distance(ind_corr_pred_bitvector, parent2_corr_pred_bitvector)
        elif self.crowding_variant == "syntactic":
            pairing1_dist = get_syntactic_distance(ind, parent1)
            pairing2_dist = get_syntactic_distance(ind, parent2)
        else:
            raise Exception("Choose a valid crowding variant")
        # Select survivors depending on the computed distances
        if pairing1_dist < pairing2_dist:
            if parent1_score <= ind_score or (random.random() < self.p_replace_anyway and parent1_score != best_score):
                add_to_new_gen(ind, ind_score, ind_corr_pred_bitvector)
            else:
                add_to_new_gen(parent1, parent1_score, parent1_corr_pred_bitvector)
            add_to_new_gen(parent2, parent2_score, parent2_corr_pred_bitvector)
        else:
            if parent2_score <= ind_score or (random.random() < self.p_replace_anyway and parent2_score != best_score):
                add_to_new_gen(ind, ind_score, ind_corr_pred_bitvector)
            else:
                add_to_new_gen(parent2, parent2_score, parent2_corr_pred_bitvector)
            add_to_new_gen(parent1, parent1_score, parent1_corr_pred_bitvector)

    def crowding_replacement_two_offspring(self, add_to_new_gen, ind1, ind2,
                                           parent1, parent1_score, parent1_corr_pred_bitvector,
                                           parent2, parent2_score, parent2_corr_pred_bitvector,
                                           best_score):
        ind1_score, ind1_corr_pred_bitvector = self.compute_scores([ind1])
        ind1_score, ind1_corr_pred_bitvector = ind1_score[0], ind1_corr_pred_bitvector[0]
        ind2_score, ind2_corr_pred_bitvector = self.compute_scores([ind2])
        ind2_score, ind2_corr_pred_bitvector = ind2_score[0], ind2_corr_pred_bitvector[0]
        # Compute the distances depending on the distance metric chosen
        if self.crowding_variant == "semantic_absolute":
            pairing1_dist = get_absolute_semantic_distance(ind1_corr_pred_bitvector, parent1_corr_pred_bitvector) + \
                            get_absolute_semantic_distance(ind2_corr_pred_bitvector, parent2_corr_pred_bitvector)
            pairing2_dist = get_absolute_semantic_distance(ind1_corr_pred_bitvector, parent2_corr_pred_bitvector) + \
                            get_absolute_semantic_distance(ind2_corr_pred_bitvector, parent1_corr_pred_bitvector)
        elif self.crowding_variant == "semantic_relative":
            pairing1_dist = get_relative_semantic_distance(ind1_corr_pred_bitvector, parent1_corr_pred_bitvector) + \
                            get_relative_semantic_distance(ind2_corr_pred_bitvector, parent2_corr_pred_bitvector)
            pairing2_dist = get_relative_semantic_distance(ind1_corr_pred_bitvector, parent2_corr_pred_bitvector) + \
                            get_relative_semantic_distance(ind2_corr_pred_bitvector, parent1_corr_pred_bitvector)
        elif self.crowding_variant == "syntactic":
            pairing1_dist = get_syntactic_distance(ind1, parent1) + get_syntactic_distance(ind2, parent2)
            pairing2_dist = get_syntactic_distance(ind1, parent2) + get_syntactic_distance(ind2, parent1)
        else:
            raise Exception("Choose a valid crowding variant")
        # Select survivors depending on the computed distances
        if pairing1_dist < pairing2_dist:
            if parent1_score <= ind1_score or (random.random() < self.p_replace_anyway and parent1_score != best_score):
                add_to_new_gen(ind1, ind1_score, ind1_corr_pred_bitvector)
            else:
                add_to_new_gen(parent1, parent1_score, parent1_corr_pred_bitvector)
            if parent2_score <= ind2_score or (
                    random.random() < self.p_replace_anyway and parent2_score != best_score):
                add_to_new_gen(ind2, ind2_score, ind2_corr_pred_bitvector)
            else:
                add_to_new_gen(parent2, parent2_score, parent2_corr_pred_bitvector)
        else:
            if parent1_score <= ind2_score or (random.random() < self.p_replace_anyway and parent1_score != best_score):
                add_to_new_gen(ind2, ind2_score, ind2_corr_pred_bitvector)
            else:
                add_to_new_gen(parent1, parent1_score, parent1_corr_pred_bitvector)
            if parent2_score <= ind1_score or \
                    (random.random() < self.p_replace_anyway and parent2_score != best_score):
                add_to_new_gen(ind1, ind1_score, ind1_corr_pred_bitvector)
            else:
                add_to_new_gen(parent2, parent2_score, parent2_corr_pred_bitvector)
