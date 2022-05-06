import csv
import copy
from plotting import transform_to_cumulative
from math import sqrt
import numpy as np

"""
This module contains methods useful for reporting (aggregates of) data contained within observers to CSV files.
"""


def get_mean_and_se(a_list):
    a_list_np = np.array(a_list)
    a_list_np_mean = a_list_np.mean()
    a_list_np_se = a_list_np.std() / len(a_list)
    return a_list_np_mean, a_list_np_se


def initialise_report_evolutionary(pathname, *args):
    """
    Initialises a CSV file for reporting evolutionary observer data at the provided pathname. A default set of
    column names is added to the file. The function provides the possibility of adding more column names via additional
    arguments. In other words, to let more column names be initialised, additional strings have to be passed as
    arguments to this function.
    """
    csvfile = open(f"{pathname}/evaluation.csv", "w")
    filewriter = csv.writer(csvfile, delimiter=",")

    row_to_write = ["Number of runs",
                    "Max generations",
                    "Cutoff time",
                    "Population size",
                    "Variable absence bias",
                    "Selection operator",
                    "Tournament size",
                    "Crowding variant",
                    "Crossover operators",
                    "Crossover rate",
                    "Mutation operators",
                    "Use clause bitvector cache",
                    "Replacement strategy",
                    "Use local search",
                    "Use infeasibility",
                    "Using knowledge compilation",
                    "Knowledge compilation variant",
                    "Using diagram for instance evaluation",
                    "Average Time",
                    "Average actual generations",
                    "Average best Score",
                    "Average average score",
                    "Average worst score",
                    "Average average best score"]
    for arg in args:
        row_to_write.append(arg)

    filewriter.writerow(row_to_write)
    csvfile.close()


def report_averages_over_observers_evolutionary(observer_list, pathname, *args):
    """
    Takes a list of evolutionary observers and reports about averages of certain data contained in these
    observers. The data reported on are the genetic algorithm configurations corresponding to the observers and the
    average time the runs took and the average best and worst scores at the end of the run.
    The computed averages are added to a report in CSV format, that should have been initialised prior to running this
    function at the location specified by the pathname argument. All provided observers are assumed to correspond to
    genetic algorithm runs with an identical setup. The first observer in the list is used to report on the specifics
    of this set-up. For example, if in the first observer list the information is contained that a population size of 35
    was used, then this population size will be reported in the entry in the CSV file, even if the second observer in
    the list erroneously corresponds to a run with a different population size. In addition to the default (aggregate)
    data from the provided observers, extra data can also be provided to be part of the added row in the CSV file.
    In order to do this, the additional data should be provided as extra arguments. Each separate argument will be
    placed in a separate column.
    """
    csvfile = open(f"{pathname}/evaluation.csv", "a")
    filewriter = csv.writer(csvfile, delimiter=",")
    processed_observer_list = [observer.remove_entries_after_cutoff() for observer in observer_list]

    times = []
    actual_generations = []
    best_scores = []
    average_scores = []
    worst_scores = []
    average_best_scores = []

    for processed_observer in processed_observer_list:
        times.append(transform_to_cumulative(processed_observer.get_entries("gen_duration"))[-1])
        actual_generations.append(
            len(processed_observer.log) - 1)  # - 1 because first log entry is from initialization, not an actual generation
        best_scores.append(processed_observer.get_last_log_entry()["best_score"])
        average_scores.append(processed_observer.get_last_log_entry()["avg_score"])
        worst_scores.append(processed_observer.get_last_log_entry()["worst_score"])
        average_best_scores.append(np.array(processed_observer.get_entries("best_score")).mean())

    average_time, se_time = get_mean_and_se(times)
    average_actual_generations, se_actual_generations = get_mean_and_se(actual_generations)
    average_best_score, se_best_score = get_mean_and_se(best_scores)
    average_average_score, se_average_score = get_mean_and_se(average_scores)
    average_worst_score, se_worst_score = get_mean_and_se(worst_scores)
    # To clarify: with the average best score of a run, we mean the best score achieved in the
    # population, averaged over all generations. Then, average_average_best_score is the average of this metric
    # over multiple runs
    average_average_best_score, se_average_best_score = get_mean_and_se(average_best_scores)

    a_processed_observer = processed_observer_list[0]

    row_to_write = [len(processed_observer_list),
                    a_processed_observer.max_generations,
                    a_processed_observer.cutoff_time,
                    a_processed_observer.population_size,
                    a_processed_observer.variable_absence_bias,
                    a_processed_observer.selection_operator,
                    a_processed_observer.tournament_size,
                    a_processed_observer.crowding_variant,
                    a_processed_observer.crossover_operators,
                    a_processed_observer.crossover_rate,
                    a_processed_observer.mutation_operators,
                    a_processed_observer.use_clause_bitvector_cache,
                    a_processed_observer.replacement_strategy,
                    a_processed_observer.use_local_search,
                    a_processed_observer.use_infeasibility,
                    a_processed_observer.use_knowledge_compilation_caching,
                    a_processed_observer.knowledge_compilation_variant,
                    a_processed_observer.use_diagram_for_instance_evaluation,
                    str(round(average_time, 2)) + " +- " + str(round(se_time, 3)),
                    str(round(average_actual_generations, 2)) + " +- " + str(round(se_actual_generations, 3)),
                    str(round(average_best_score, 2)) + " +- " + str(round(se_best_score, 3)),
                    str(round(average_average_score, 2)) + " +- " + str(round(se_average_score, 3)),
                    str(round(average_worst_score, 2)) + " +- " + str(round(se_worst_score, 3)),
                    str(round(average_average_best_score, 2)) + " +- " + str(round(se_average_best_score, 3))]
    for arg in args:
        row_to_write.append(arg)

    filewriter.writerow(row_to_write)
    csvfile.close()


def initialise_report_sls(pathname, *args):
    """
    Initialises a csv file for reporting sls observer data at the provided pathname. A default set of
    column names is added to the file. The function provides the possibility of adding more column names via additional
    arguments. In other words, to let more column names be initialised, additional strings have to be passed as
    arguments to this function.
    """
    csvfile = open(f"{pathname}/evaluation_sls.csv", "w")
    filewriter = csv.writer(csvfile, delimiter=",")

    row_to_write = ["Number of runs", "Cutoff time", "Perform random restarts",
                    "Method", "Initialization attempts", "Variable absence bias", "Neighbourhood limit",
                    "Prune with coverage heuristic", "Recompute random incorrect examples",
                    "Using knowledge compilation", "Using infeasibility", "Average best score",
                    "Average iterations", "Average total time", "Average random reset time", "Average computing neighbours time",
                    "Average evaluation time", "Average number of random restarts", "Average number of window hits"]
    for arg in args:
        row_to_write.append(arg)

    filewriter.writerow(row_to_write)
    csvfile.close()


def report_averages_over_observers_sls(observer_list, pathname, *args):
    """
    This function takes a list of sls observers and reports about averages of certain data contained in these observers.
    The computed averages are added to a report in CSV format, that should have been initialised prior to running this
    method at the location specified by the pathname argument.
    """
    csvfile = open(f"{pathname}/evaluation_sls.csv", "a")
    filewriter = csv.writer(csvfile, delimiter=",")
    processed_observer_list = [observer.remove_entries_after_cutoff() for observer in observer_list]

    best_scores = []
    actual_iterations = []
    total_times = []
    random_restart_times = []
    computing_neighbours_times = []
    evaluation_times = []
    number_of_random_restarts = []
    number_of_window_hits = []

    for processed_observer in processed_observer_list:
        best_scores.append(processed_observer.get_last_log_entry()["best_score"])
        actual_iterations.append(
            len(processed_observer.log) - 1)  # - 1 because first log entry is from initialization, not an actual generation
        total_times.append(processed_observer.get_last_log_entry()["cumulative_time"])
        random_restart_times.append(processed_observer.get_last_log_entry()["random_restart_time"])
        computing_neighbours_times.append(processed_observer.get_last_log_entry()["computing_neighbours_time"])
        evaluation_times.append(processed_observer.get_last_log_entry()["evaluation_time"])
        number_of_random_restarts.append(processed_observer.get_last_log_entry()["random_restart_count"])
        number_of_window_hits.append(processed_observer.get_last_log_entry()["number_of_window_hits"])
    average_best_score, se_best_score = get_mean_and_se(best_scores)
    average_actual_iterations, se_actual_iterations = get_mean_and_se(actual_iterations)
    average_total_time, se_total_time = get_mean_and_se(total_times)
    average_random_restart_time, se_random_restart_time = get_mean_and_se(random_restart_times)
    average_computing_neighbours_time, se_computing_neighbours_time = get_mean_and_se(computing_neighbours_times)
    average_evaluation_time, se_evaluation_time = get_mean_and_se(evaluation_times)
    average_number_of_random_restarts, se_number_of_random_restarts = get_mean_and_se(number_of_random_restarts)
    average_number_of_window_hits, se_number_of_window_hits = get_mean_and_se(number_of_window_hits)
    a_processed_observer = processed_observer_list[0]

    row_to_write = [len(processed_observer_list), a_processed_observer.cutoff_time,
                    a_processed_observer.perform_random_restarts,
                    a_processed_observer.method, a_processed_observer.initialization_attempts,
                    a_processed_observer.variable_absence_bias,
                    a_processed_observer.neighbourhood_limit,
                    a_processed_observer.prune_with_coverage_heuristic,
                    a_processed_observer.recompute_random_incorrect_examples,
                    a_processed_observer.use_knowledge_compilation_caching, a_processed_observer.use_infeasibility,
                    str(round(average_best_score, 2)) + " +- " + str(round(se_best_score, 3)),
                    str(round(average_actual_iterations, 2)) + " +- " + str(round(se_actual_iterations, 3)),
                    str(round(average_total_time, 2)) + " +- " + str(round(se_total_time, 3)),
                    str(round(average_random_restart_time, 2)) + " +- " + str(round(se_random_restart_time, 3)),
                    str(round(average_computing_neighbours_time, 2)) + " +- " + str(
                        round(se_computing_neighbours_time, 3)),
                    str(round(average_evaluation_time, 2)) + " +- " + str(round(se_evaluation_time, 3)),
                    str(round(average_number_of_random_restarts, 2)) + " +- " + str(round(se_number_of_random_restarts, 3)),
                    str(round(average_number_of_window_hits, 2)) + " +- " + str(round(se_number_of_window_hits, 3))]
    for arg in args:
        row_to_write.append(arg)

    filewriter.writerow(row_to_write)
    csvfile.close()
