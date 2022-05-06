import random
from copy import deepcopy
from functools import partial
import numpy as np


def get_syntactic_distance(ind1, ind2):
    """
    Calculates the syntactic distance between two individuals. Calculating this distance consists of two steps:
    1) Matching each clause of each individual to the most syntactically similar still unmatched clause of the other
    individual
    2) Summing up the Hamming distances between each pair of matched clauses
    :param ind1: The first individual
    :param ind2: The second individual
    :return The syntactic distance between the two individuals
    """
    # We calculate the pairwise distance between ind1's and ind2's clauses
    distance_matrix = np.zeros((len(ind1), len(ind2)))
    for i in range(len(ind1)):
        clause1 = ind1[i]
        for j in range(len(ind2)):
            clause2 = ind2[j]
            curr_syntactic_distance = 0
            for k in range(len(clause1)):
                if clause1[k] != clause2[k]:
                    curr_syntactic_distance += 1
            distance_matrix[i][j] = curr_syntactic_distance

    # Arg-sort the pairwise clause distances from least to most different
    distances_ordered = np.dstack(np.unravel_index(np.argsort(distance_matrix.ravel()), (len(ind1), len(ind2))))[0]

    # Finally match the clauses, making sure that each clause is only matched once
    # Calculate the total distance between matched clauses
    ind1_matched_clauses = set()
    ind2_matched_clauses = set()
    count = 0
    total_syntactic_distance = 0
    for match in distances_ordered:
        i = match[0]
        j = match[1]
        if i not in ind1_matched_clauses and j not in ind2_matched_clauses:
            count += 1
            total_syntactic_distance += distance_matrix[i][j]
            ind1_matched_clauses.add(i)
            ind2_matched_clauses.add(j)

    if count != len(ind1):
        # Something went wrong in case there were less or more matches than the number of clauses in an individual
        raise Exception("Something went wrong in the matched uniform crossover")

    return total_syntactic_distance


def get_absolute_semantic_distance(ind1_correctly_predicted_bitvector, ind2_correctly_predicted_bitvector):
    """
    Computes the absolute semantic distance between two individuals. This distance metric does not express the
    difference between the two individuals on a syntactic level, but rather on a semantic level, in terms of how
    different between the sets of examples that the individuals correctly label. For each of the two individuals, this
    function takes an accuracy bitvector with as many entries as there are examples.
    Each entry in the bitvector denotes whether the associated individual correctly labeled the corresponding example.
    :param ind1_correctly_predicted_bitvector: The first individual's associated bitvector
    :param ind2_correctly_predicted_bitvector: The second individual's associated bitvector
    :return: The absolute semantic distance between the two individuals
    """
    semantic_distance = 0
    for k in range(len(ind1_correctly_predicted_bitvector)):
        if ind1_correctly_predicted_bitvector[k] != ind2_correctly_predicted_bitvector[k]:
            semantic_distance += 1
    return semantic_distance


def get_relative_semantic_distance(ind1_correctly_predicted_bitvector, ind2_correctly_predicted_bitvector):
    """
    Does the same as function get_absolute_semantic_distance, but returns a relative distance, rather than an
    absolute one. The relative metric lies between 0 and 1, where 0 corresponds to the minimum possible distance and
    1 to the maximum possible distance between the two individuals. The minimum and maximum distances depend on the
    accuracy bitvectors of the two individuals.
    :param ind1_correctly_predicted_bitvector: The first individual's associated bitvector
    :param ind2_correctly_predicted_bitvector: The second individual's associated bitvector
    :return: The relative semantic distance between the two individuals
    """
    ind1_score_absolute = sum(ind1_correctly_predicted_bitvector)
    ind2_score_absolute = sum(ind2_correctly_predicted_bitvector)

    semantic_distance = 0
    for k in range(len(ind1_correctly_predicted_bitvector)):
        if ind1_correctly_predicted_bitvector[k] != ind2_correctly_predicted_bitvector[k]:
            semantic_distance += 1

    # Given the individuals' scores, there is a maximum semantic distance between the individuals
    max_semantic_distance = len(ind1_correctly_predicted_bitvector) - abs(ind1_score_absolute + ind2_score_absolute - len(ind1_correctly_predicted_bitvector))
    min_semantic_distance = abs(ind2_score_absolute - ind2_score_absolute)
    if max_semantic_distance != min_semantic_distance:
        return (semantic_distance-min_semantic_distance)/(max_semantic_distance-min_semantic_distance)
    else:
        return 1


class GeneticAlgorithm:
    """
    A generic genetic algorithm class
    """
    def __init__(self,
                 population_size,
                 individual_factory=None,
                 score_function=None):
        """
        :param population_size: The population size
        :param individual_factory: The function to use for initialising MAX-SAT models
        :param score_function: The function to use to evaluating MAX-SAT models
        """
        self.individual_factory = individual_factory
        self.score_function = score_function
        self.crossovers = []
        self.mutations = []
        self.to_phenotype = lambda ind: ind
        self.population_size = population_size
        self.population = None
        self.scores = None
        self.correctly_predicted_bitvectors = None
        self.observers = []

    def select(self, index, copy=True):
        """
        Returns the individual at a certain index in the population, as well as its score and associated
        correctly_predicted_bitvector
        :param index: The requested index
        :param copy: A Boolean denoting whether to return copies or not
        :return: The individual, its score and its associated correctly_predicted_bitvector
        """
        if copy:
            return deepcopy(self.population[index]), deepcopy(self.scores[index]),\
                   deepcopy(self.correctly_predicted_bitvectors[index])
        else:
            return self.population[index], self.scores[index], self.correctly_predicted_bitvectors[index]

    def replace_population(self, gen_pop, gen_pop_scores, gen_pop_corr_class_bitvector):
        """
        Replaces the current population
        :param gen_pop: The new population as a list of individuals
        :param gen_pop_scores: A list of associated scores
        :param gen_pop_corr_class_bitvector: A list of associated accuracy bitvectors
        """
        self.population = gen_pop
        self.scores = gen_pop_scores
        self.correctly_predicted_bitvectors = gen_pop_corr_class_bitvector

    def crossover(self, ind1, ind2):
        """
        Crosses over two individuals in-place, using one of the genetic algorithm's crossover operators, selected
        at random
        :param ind1: The first individual
        :param ind2: The second individual
        """
        distribution = [1 / len(self.crossovers) for _ in self.crossovers]
        i = random.choices(range(len(distribution)), weights=distribution)[0]
        self.crossovers[i](ind1, ind2)

    def add_crossover(self, func, *args, **kwargs):
        """
        Adds a crossover operator to the genetic algorithm
        :param func: The crossover operator as a function
        :param args: Associated arguments
        :param kwargs: Associated keyword arguments
        """
        self.crossovers.append(partial(func, *args, **kwargs))

    def mutate(self, individual):
        """
        Mutates an individual in-place, using each of the genetic algorithm's mutation operators in turn
        :param individual: The individual
        """
        for mutation in self.mutations:
            mutation(individual)

    def add_mutation(self, func, *args, **kwargs):
        """
        Adds a mutation operator to the genetic algorithm
        :param func: The mutation operator as a function
        :param args: Associated arguments
        :param kwargs: Associated keyword arguments
        """
        self.mutations.append(partial(func, *args, **kwargs))

    def init_population(self):
        """
        Initializes the population using the genetic algorithm's individual factory
        """
        population = [self.individual_factory() for _ in range(self.population_size)]
        self.population = population
        self.scores, self.correctly_predicted_bitvectors = self.compute_scores(population)

    def tournament_select(self, t_size=None, clone=False, return_index=False, index_to_exclude=None):
        """
        Performs tournament selection
        :param t_size: The tournament size
        :param clone: A Boolean that denotes whether to return a copy of the selected individual, or the individual
        itself. Only relevant when index is False.
        :param return_index: A Boolean that denotes whether to return the index of the selected individual in the
        population, or (a copy of) the individual itself.
        :param index_to_exclude: Optional - An index to exclude. The individual positioned at this index cannot be
        selected
        :return: The selected individual itself, or a copy of the selected individual, or the index of the selected
        individual, depending on the arguments given
        """
        if t_size is None or t_size >= self.population_size:
            if index_to_exclude is None:
                selected = range(self.population_size)
            else:
                selected = list(range(index_to_exclude)) + list(range(index_to_exclude + 1, self.population_size))
        else:
            if index_to_exclude is None:
                selected = random.sample(range(self.population_size), t_size)
            else:
                selected = random.sample(
                    list(range(index_to_exclude)) + list(range(index_to_exclude + 1, self.population_size)), t_size)

        selected_index = max(selected, key=lambda i: self.scores[i])

        if return_index:
            return selected_index

        individual = self.population[selected_index]

        if clone:
            individual = deepcopy(individual)

        return individual

    def compute_scores(self, population):
        """
        Evaluates all individuals in the given population
        :param population: A population of individuals
        :return: A tuple consisting of (i) a list of training set accuracies and (ii) a list of accuracy bitvectors,
        where each bitvector consists of as many entries as there are examples, and each entry denotes whether the
        individual correctly labeled the corresponding example.
        """
        score_list = []
        correctly_predicted_bitvector_list = []
        for phenotype in (self.to_phenotype(ind) for ind in population):
            score, correctly_predicted_bitvector = self.score_function(phenotype)
            score_list.append(score)
            correctly_predicted_bitvector_list.append(correctly_predicted_bitvector)
        return score_list, correctly_predicted_bitvector_list

    def get_number_of_distinct_individuals(self):
        """
        Returns the number of distinct individuals currently present in the population
        """
        set_of_individuals = set()
        for individual in self.population:
            individual_as_set = set()
            for clause in individual:
                individual_as_set.add(tuple(clause))
            set_of_individuals.add(frozenset(individual_as_set))
        return len(set_of_individuals)

    def get_number_of_distinct_clauses(self):
        """
        Returns the number of distinct clauses currently present in the population
        """
        clause_set = set()
        for individual in self.population:
            for clause in individual:
                clause_set.add(tuple(clause))
        return len(clause_set)

    def get_number_of_distinct_clauses_per_individual(self):
        """
        Returns the average number of distinct clauses per individual in the current population
        """
        distinct_clauses_per_individual_list = []
        for individual in self.population:
            individual_clause_set = set()
            for clause in individual:
                individual_clause_set.add(tuple(clause))
            distinct_clauses_per_individual_list.append(len(individual_clause_set))
        return sum(distinct_clauses_per_individual_list) / len(distinct_clauses_per_individual_list)

    def get_syntactic_population_diversity(self):
        """
        The average pair-wise syntactic distance between individual genotypes in the population.
        """
        total_syntactic_pairwise_distance = 0
        count = 0
        for i in range(self.population_size):
            for j in range(i + 1, self.population_size):
                total_syntactic_pairwise_distance += get_syntactic_distance(self.population[i], self.population[j])
                count += 1
        return total_syntactic_pairwise_distance / count

    def get_absolute_semantic_population_diversity(self):
        """
        Returns the absolute semantic population diversity, which is the average absolute semantic distance between
        any two individuals in the population, as computed by function get_absolute_semantic_distance.
        :return: The absolute semantic population diversity
        """
        total_semantic_pairwise_distance = 0
        count = 0
        for i in range(self.population_size):
            for j in range(i + 1, self.population_size):
                bitvector_1 = self.correctly_predicted_bitvectors[i]
                bitvector_2 = self.correctly_predicted_bitvectors[j]
                semantic_distance = 0
                for k in range(len(bitvector_1)):
                    if bitvector_1[k] != bitvector_2[k]:
                        semantic_distance += 1
                total_semantic_pairwise_distance += semantic_distance
                count += 1
        return total_semantic_pairwise_distance / count

    def get_relative_semantic_population_diversity(self):
        """
        Returns the relative semantic population diversity, which is the average relative semantic distance between
        any two individuals in the population, as computed by function get_relative_semantic_distance.
        :return: The relative semantic population diversity
        """
        total_semantic_pairwise_distance = 0
        count = 0
        for i in range(self.population_size):
            for j in range(i + 1, self.population_size):
                bitvector_1 = self.correctly_predicted_bitvectors[i]
                bitvector_2 = self.correctly_predicted_bitvectors[j]
                score_1 = self.scores[i] * len(bitvector_1)
                score_2 = self.scores[j] * len(bitvector_1)
                semantic_distance = 0
                for k in range(len(bitvector_1)):
                    if bitvector_1[k] != bitvector_2[k]:
                        semantic_distance += 1

                # Given the individuals' scores, there is a maximum semantic distance between the individuals
                max_semantic_distance = len(bitvector_1) - abs(score_1 + score_2 - len(bitvector_1))
                min_semantic_distance = abs(score_1 - score_2)
                if max_semantic_distance != min_semantic_distance:
                    total_semantic_pairwise_distance += (semantic_distance - min_semantic_distance)/(max_semantic_distance-min_semantic_distance)
                else:
                    total_semantic_pairwise_distance += 1
                count += 1
        return total_semantic_pairwise_distance / count

    def get_example_coverage_vector(self):
        """
        Returns the 'example coverage vector', which is a vector with as many entries as there are examples. Each entry
        is a value between 0 and 1, which denotes the proportion of the current population that correctly labels
        the corresponding example.
        :return: The example coverage vector
        """
        number_of_examples = len(self.correctly_predicted_bitvectors[0])
        example_coverage_bitvector = []
        for i in range(number_of_examples):
            covered_by_amount = \
                len([j for j in range(self.population_size) if self.correctly_predicted_bitvectors[j][i] == 1])
            covered_by_proportion = covered_by_amount / self.population_size
            example_coverage_bitvector.append(covered_by_proportion)
        return example_coverage_bitvector

