import platform
import os
import numpy as np
import copy
import matplotlib as mpl
import matplotlib.markers as mark
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from plotting import ScatterData

if platform.system() == "Darwin":
    mpl.use("TkAgg")

"""
This module contains code related to Observers. These Observers are useful to track runs of genetic algorithms. They
may contain information about the various generations attained throughout the run, as well as information about the
genetic algorithm configuration which was used.
"""


class Observer:
    def observe_generation(self, gen_count, best_score, **kwargs):
        raise NotImplementedError()


class TrackObserver(Observer):
    """An observer that can be used to observe HASSLE-GEN runs"""
    def __init__(self,
                 max_generations=None,
                 population_size=None,
                 selection_operator="tournament",
                 tournament_size=None,
                 crossover_operators=None,
                 mutation_operators=None,
                 crossover_rate=None,
                 replacement_strategy=None,
                 use_local_search=None,
                 cutoff_time=None,
                 use_knowledge_compilation_caching=True,
                 knowledge_compilation_variant=4,
                 use_diagram_for_instance_evaluation=True,
                 variable_absence_bias=1,
                 use_infeasibility=False,
                 use_clause_bitvector_cache=True,
                 crowding_variant=None,
                 legend_entry=None):
        self.log = []
        self.max_generations = max_generations
        self.population_size = population_size
        self.selection_operator = selection_operator
        self.tournament_size = tournament_size
        self.use_infeasibility = use_infeasibility

        if crossover_operators is None:
            # Use default
            self.crossover_operators = ["scramble_clause_crossover", "uniform_crossover"]
        else:
            self.crossover_operators = crossover_operators

        if mutation_operators is None:
            # Use default
            self.mutation_operators = ["mutate_hardness, P_trigger = 1, P_inner = 0.05",
                                        "mutate_clause, P_trigger = 1, P_inner = 0.05",
                                        "mutate_weight, P_trigger = 1, P_inner = 0.05"]
        else:
            self.mutation_operators = mutation_operators

        self.crossover_rate = crossover_rate
        if replacement_strategy is None:
            self.replacement_strategy = "Age-based replacement"
        else:
            self.replacement_strategy = replacement_strategy
        self.use_local_search = use_local_search
        self.cutoff_time = cutoff_time
        self.use_knowledge_compilation_caching = use_knowledge_compilation_caching
        self.knowledge_compilation_variant = knowledge_compilation_variant
        self.use_diagram_for_instance_evaluation = use_diagram_for_instance_evaluation
        self.variable_absence_bias = variable_absence_bias
        self.use_clause_bitvector_cache = use_clause_bitvector_cache
        self.crowding_variant = crowding_variant
        self.legend_entry = legend_entry

    def get_last_log_entry(self):
        return copy.deepcopy(self.log[-1])

    def get_entries(self, key):
        return [e[key] for e in self.log]

    def get_array(self, key):
        return np.array(self.get_entries(key))

    def plot(self, *args, use_time=False):
        scatter = ScatterData("Evolution", [])
        scatter.y_lim((0, 1))
        if use_time:
            x = []
            for duration in self.get_entries("gen_duration"):
                if len(x) == 0:
                    x.append(duration)
                else:
                    x.append(x[-1] + duration)
            x = np.array(x)
        else:
            x = self.get_array("gen_count")

        for key in args:
            scatter.add_data(key, x, self.get_array(key))
        scatter.plot(log_x=False, log_y=False, filename="output_evolutionary")

    def observe_generation(self, gen_count, best_score, **kwargs):
        kwargs.update({"gen_count": gen_count, "best_score": best_score})
        self.log.append(kwargs)

    def remove_entries_after_cutoff(self):
        """
        Returns a deepcopy of this observer in which all elements in the log that took place after the observer's cutoff
        time are removed. If the cutoff time is None, a deepcopy of this observer is returned.
        """
        observer_copy = copy.deepcopy(self)
        if observer_copy.cutoff_time is None:
            return observer_copy
        cumulative_duration_array = transform_to_cumulative(observer_copy.get_entries("gen_duration"))
        index = next((i for i in range(len(cumulative_duration_array)) if cumulative_duration_array[i] >
                      observer_copy.cutoff_time), None)
        if index is not None:
            observer_copy.log = observer_copy.log[:index]
        return observer_copy


class SLSObserver(Observer):
    """An observer that can be used to observe HASSLE-SLS runs"""
    def __init__(self,
                 cutoff_time=None,
                 use_knowledge_compilation_caching=True,
                 legend_entry=None,
                 perform_random_restarts=True,
                 method="walk_sat",
                 initialization_attempts=1,
                 variable_absence_bias=1,
                 use_infeasibility=False,
                 neighbourhood_limit=None,
                 prune_with_coverage_heuristic=False,
                 recompute_random_incorrect_examples=True):
        self.log = []
        self.cutoff_time = cutoff_time
        self.legend_entry = legend_entry
        self.method = method
        self.initialization_attempts=initialization_attempts
        self.variable_absence_bias = variable_absence_bias
        self.perform_random_restarts = perform_random_restarts
        self.use_knowledge_compilation_caching = use_knowledge_compilation_caching
        self.use_infeasibility = use_infeasibility
        self.neighbourhood_limit = neighbourhood_limit
        self.prune_with_coverage_heuristic = prune_with_coverage_heuristic
        self.recompute_random_incorrect_examples = recompute_random_incorrect_examples

    def get_last_log_entry(self):
        return copy.deepcopy(self.log[-1])

    def get_entries(self, key):
        return [e[key] for e in self.log]

    def get_array(self, key):
        return np.array(self.get_entries(key))

    def observe_generation(self, gen_count, best_score, **kwargs):
        kwargs.update({"gen_count": gen_count, "best_score": best_score})
        self.log.append(kwargs)

    def remove_entries_after_cutoff(self):
        """
        Returns a deepcopy of this observer in which all elements in the log that took place after the observer's cutoff
        time are removed. If the cutoff time is None, a deepcopy of this observer is returned.
        """
        observer_copy = copy.deepcopy(self)
        if observer_copy.cutoff_time is None:
            return observer_copy
        cumulative_duration_array = transform_to_cumulative(observer_copy.get_entries("gen_duration"))
        index = next((i for i in range(len(cumulative_duration_array)) if cumulative_duration_array[i] >
                      observer_copy.cutoff_time), None)
        if index is not None:
            observer_copy.log = observer_copy.log[:index]
        return observer_copy


def transform_to_cumulative(duration_array):
    """
    Transforms an array of durations to a cumulative array of durations.
    Consider the following example:
    duration_array = [1, 2, 3]
    output_array = [1, 3, 6]
    """
    output_array = []
    cumulative_value = 0
    for element in duration_array:
        cumulative_value += element
        output_array.append(cumulative_value)
    return output_array
