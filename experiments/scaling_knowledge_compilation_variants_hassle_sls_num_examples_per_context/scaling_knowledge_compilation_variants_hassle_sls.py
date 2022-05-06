import pickle
from datetime import datetime
from os import listdir, path, makedirs
import numpy as np
import random
import multiprocessing as mp
import re
from observe import TrackObserver, SLSObserver
from learn_max_sat_model import learn_max_sat_model
from plotting import plot_time_relations_over_multiple_patterns
from reporting import initialise_report_evolutionary, report_averages_over_observers_evolutionary, \
    initialise_report_sls, report_averages_over_observers_sls
from hassle_sls.local_search import learn_weighted_max_sat


def run_experiment(problem_structure_pattern):
    # Find all other context-and-data files with this pattern, i.e., that differ merely in their model or context seeds
    files_with_this_pattern = []
    for f in listdir(pathname_contexts_and_data):
        filename_without_extension = re.search(r"(.*)(\.pickle)", f).group(1)
        filename_without_seed_info = re.sub(r"(_model_seed_)(\d*)", '', filename_without_extension)
        filename_without_seed_info = re.sub(r"(_context_seed_)(\d*)", '', filename_without_seed_info)
        if filename_without_seed_info == problem_structure_pattern:
            files_with_this_pattern.append(f)

    # Construct a folder for results and initialize a results CSV file

    observer_lists_this_pattern = [[] for _ in range(len(knowledge_compilation_configurations))]
    for f in files_with_this_pattern:
        # Open pickle file
        with open(pathname_contexts_and_data + f, "rb") as fp:
            pickle_var = pickle.load(fp)


        # Get information from file
        # n = number of variables, k = number of constraints
        n = int(re.search(r"(_n_)(\d*)", f).group(2))
        k = int(re.search(r"(_num_hard_)(\d*)", f).group(2)) + int(re.search(r"(_num_soft_)(\d*)", f).group(2))
        match = re.search(r"(_conjunctive_contexts_)(\d*)", f)
        if match is not None:
            conjunctive_contexts = int(match.group(2))
        else:
            conjunctive_contexts = 0
        contexts, instances, labels = (
            [{int(str(lit)) for lit in context} for context in pickle_var["contexts"]],
            pickle_var["data"],
            pickle_var["labels"],
        )
        print(f"Starting experiment on file {f}\nwith number of variables: {n} and number of constraints: {k}")


        # Run actual experiment
        for i in range(len(knowledge_compilation_configurations)):
            knowledge_compilation_configuration = knowledge_compilation_configurations[i]
            use_knowledge_compilation_caching = knowledge_compilation_configuration[0]
            knowledge_compilation_variant = knowledge_compilation_configuration[1]
            use_diagram_for_instance_evaluation = knowledge_compilation_configuration[2]

            observer_list = observer_lists_this_pattern[i]
            for j in range(1, runs_per_configuration + 1):
                print(f"Starting run: {j}")
                observer = SLSObserver(cutoff_time=cutoff_time, use_knowledge_compilation_caching=use_knowledge_compilation_caching,
                                       method="walk_sat",
                                       perform_random_restarts=perform_random_restarts,
                                       use_infeasibility=use_infeasibility,
                                       initialization_attempts=initialization_attempts,
                                       recompute_random_incorrect_examples=recompute_random_incorrect_examples,
                                       variable_absence_bias=variable_absence_bias,
                                       legend_entry=f"KC approach: {knowledge_compilation_variant}",
                                       #legend_entry=""
                                       )
                instances_np = np.array(instances)
                labels_np = np.array([True if l == 1 else False for l in labels])
                inf = [True if l == -1 else False for l in labels]
                seed = random.randint(0, 100000)
                best_solution = learn_weighted_max_sat(k, n, instances_np, labels_np, contexts, "walk_sat", param="",
                                                       inf=inf if use_infeasibility else None,
                                                       cutoff_time=cutoff_time, seed=seed,
                                                       variable_absence_bias=variable_absence_bias,
                                                       use_knowledge_compilation=use_knowledge_compilation_caching,
                                                       knowledge_compilation_variant=knowledge_compilation_variant,
                                                       use_diagram_for_instance_evaluation=use_diagram_for_instance_evaluation,
                                                       conjunctive_contexts=conjunctive_contexts,
                                                       perform_random_restarts=perform_random_restarts,
                                                       window_size=window_size,
                                                       initialization_attempts=initialization_attempts,
                                                       recompute_random_incorrect_examples=recompute_random_incorrect_examples,
                                                       observers=[observer])
                observer_list.append(observer)
    return observer_lists_this_pattern


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)

    experiment_time_stamp = datetime.now().strftime("%d-%m-%y (%H:%M:%S.%f)")

    # Setup
    runs_per_configuration = 3
    generations_per_run = 10000
    cutoff_time = 60
    knowledge_compilation_configurations = [(False, None, False),
                                            (True, 4, True)]
    perform_random_restarts = False
    recompute_random_incorrect_examples = False
    window_size = 100000
    initialization_attempts = 1
    variable_absence_bias = 2
    use_infeasibility = False

    # What parameter is being varied? Used in file name
    varied_parameter = "num_pos"

    pathname = path.dirname(path.realpath(__file__))
    pathname_contexts_and_data = pathname + "/pickles/contexts_and_data/"

    # By 'problem structure pattern', we mean the specification of the problem - as it appears in the file name - but without
    # information about the model seed and context seed used to construct that problem
    problem_structure_patterns = set()
    for f in listdir(pathname_contexts_and_data):
        filename_without_extension = re.search(r"(.*)(\.pickle)", f).group(1)
        filename_without_seed_info = re.sub(r"(_model_seed_)(\d*)", '', filename_without_extension)
        filename_without_seed_info = re.sub(r"(_context_seed_)(\d*)", '', filename_without_seed_info)
        problem_structure_patterns.add(filename_without_seed_info)

    problem_structure_patterns = list(problem_structure_patterns)
    all_observer_lists = []

    # Sort the problem structure patterns
    values_of_varied_parameter = [re.search(r"(_"+varied_parameter+r"_)(\d*)", problem_structure_pattern).group(2) for problem_structure_pattern in problem_structure_patterns]
    values_of_varied_parameter = [int(x) for x in values_of_varied_parameter]
    problem_structure_patterns = [x for _, x in sorted(zip(values_of_varied_parameter, problem_structure_patterns))]

    pathname_results_curr_file = f"{pathname}/results/{experiment_time_stamp}/varying_{varied_parameter}_num_of_files_per_pattern_{len(problem_structure_patterns)//len(listdir(pathname_contexts_and_data))}"
    makedirs(pathname_results_curr_file)


    # Running the experiment
    num_threads = mp.cpu_count()
    pool = mp.Pool(num_threads - 1)
    all_observer_lists = pool.map(run_experiment, problem_structure_patterns)


    # Further processing of results
    values_of_varied_parameter.sort()
    plot_time_relations_over_multiple_patterns(all_observer_lists,
                                               pathname=pathname_results_curr_file + "/plot",
                                               cutoff_time=cutoff_time,
                                               varied_parameter="Number of examples\nper context",
                                               #values_of_varied_parameter=values_of_varied_parameter,
                                               values_of_varied_parameter=[2*value for value in values_of_varied_parameter],
                                               size=(2, 1.5), show_legend=False)

    # Proportion of time spent computing neighbours
    for observer_lists_a_pattern in all_observer_lists:
            for observer_list in observer_lists_a_pattern:
                for observer in observer_list:
                    cumulative_time = observer.get_last_log_entry()["cumulative_time"]
                    evaluation_time = observer.get_last_log_entry()["evaluation_time"]
                    print(f"Proportion of time spent evaluating: {evaluation_time/cumulative_time}")

    with open(pathname_results_curr_file+"/all_observer_lists.pickle", 'wb') as handle:
        pickle.dump(all_observer_lists, handle, protocol=pickle.HIGHEST_PROTOCOL)
