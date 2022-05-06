import pickle
import time
from datetime import datetime
from os import listdir, path, makedirs
import numpy as np
import random
import multiprocessing as mp
import re
from observe import TrackObserver, SLSObserver
from learn_max_sat_model import learn_max_sat_model
from plotting import plot_average_final_entry_over_multiple_patterns, plot_average_evaluation_statistics_over_multiple_patterns
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

    observer_lists_this_pattern = [[] for _ in range(2)]
    learned_model_lists_this_pattern = [[] for _ in range(2)]
    target_model_list_this_pattern = [[] for _ in range(2)]
    number_of_variables_list_this_pattern = [[] for _ in range(2)]
    for f in files_with_this_pattern:
        # Open pickle file
        with open(pathname_contexts_and_data + f, "rb") as fp:
            pickle_var = pickle.load(fp)

        # Retrieve target model associated with this problem
        split_result = f.split("_num_context")
        f_model = pathname_target_models + split_result[0] + ".pickle"
        with open(f_model, "rb") as fp:
            target_model_phenotype = pickle.load(fp)['true_model']

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

        examples = list(zip(contexts, instances, labels))

        print(f"Starting experiment on file {f}\nwith number of variables: {n} and number of constraints: {k}")
        # Run actual experiment
        # Run HASSLE-SLS
        observer_list = observer_lists_this_pattern[0]
        learned_model_list = learned_model_lists_this_pattern[0]
        target_model_list = target_model_list_this_pattern[0]
        number_of_variables_list = number_of_variables_list_this_pattern[0]
        for j in range(1, runs_per_configuration + 1):
            print(f"Starting run: {j}")
            observer = SLSObserver(cutoff_time=cutoff_time, use_infeasibility=use_infeasibility,
                                   use_knowledge_compilation_caching=use_knowledge_compilation_caching,
                                   recompute_random_incorrect_examples=recompute_random_incorrect_examples,
                                   legend_entry=f"HASSLE-SLS: walk_sat, use knowledge compilation: {use_knowledge_compilation_caching}")
            instances_np = np.array(instances)
            labels_np = np.array([True if l == 1 else False for l in labels])
            if use_infeasibility:
                inf = [True if l == -1 else False for l in labels]
            else:
                inf = None
            seed = random.randint(0, 100000)
            best_solution = learn_weighted_max_sat(k, n, instances_np, labels_np, contexts, "walk_sat", param="",
                                                   inf=inf, cutoff_time=cutoff_time, seed=seed,
                                                   use_knowledge_compilation=use_knowledge_compilation_caching,
                                                   conjunctive_contexts=conjunctive_contexts,
                                                   initialization_attempts=initialization_attempts,
                                                   variable_absence_bias=variable_absence_bias,
                                                   recompute_random_incorrect_examples=recompute_random_incorrect_examples,
                                                   perform_random_restarts=perform_random_restarts,
                                                   observers=[observer])
            observer_list.append(observer)
            learned_model_list.append(best_solution)
            target_model_list.append(target_model_phenotype)
            number_of_variables_list.append(n)

        # Run HASSLE-GEN
        tournament_size = round(relative_tournament_size * population_size)
        observer_list = observer_lists_this_pattern[1]
        learned_model_list = learned_model_lists_this_pattern[1]
        target_model_list = target_model_list_this_pattern[1]
        number_of_variables_list = number_of_variables_list_this_pattern[1]
        for j in range(1, runs_per_configuration + 1):
            print(f"Starting run: {j}")
            observer = TrackObserver(max_generations=generations_per_run,
                                     population_size=population_size,
                                     tournament_size=tournament_size,
                                     crossover_operators=crossover_operator_configuration,
                                     mutation_operators=mutation_operator_configuration,
                                     crossover_rate=crossover_rate,
                                     replacement_strategy=None,
                                     use_local_search=use_local_search,
                                     cutoff_time=cutoff_time,
                                     legend_entry=f"Crossover rate: {crossover_rate}",
                                     use_knowledge_compilation_caching=use_knowledge_compilation_caching,
                                     use_infeasibility=use_infeasibility,
                                     crowding_variant=crowding_variant,
                                     variable_absence_bias=variable_absence_bias)
            start_time = time.time()
            (best_score, best_phenotype, cumulative_duration) = \
                learn_max_sat_model(n,
                                    k,
                                    examples,
                                    population_size,
                                    generations=generations_per_run,
                                    tournament_size=tournament_size,
                                    prob_crossover=crossover_rate,
                                    crossover_operators=crossover_operator_configuration,
                                    mutation_operators=mutation_operator_configuration,
                                    use_local_search=use_local_search,
                                    cutoff_time=cutoff_time,
                                    conjunctive_contexts=conjunctive_contexts,
                                    use_knowledge_compilation_caching=use_knowledge_compilation_caching,
                                    use_infeasibility=use_infeasibility,
                                    variable_absence_bias=variable_absence_bias,
                                    use_crowding=crowding,
                                    crowding_variant=crowding_variant,
                                    observers=[observer])
            observer_list.append(observer)
            learned_model_list.append(best_phenotype)
            target_model_list.append(target_model_phenotype)
            number_of_variables_list.append(n)

    # Storing observer lists related to this problem pattern for later processing over multiple patterns
    #all_observer_lists.append(observer_lists_this_pattern)
    #all_learned_model_lists.append(learned_model_lists_this_pattern)
    #all_target_model_lists.append(target_model_list_this_pattern)
    #all_number_of_variables_lists.append(number_of_variables_list_this_pattern)
    return observer_lists_this_pattern, learned_model_lists_this_pattern, target_model_list_this_pattern, number_of_variables_list_this_pattern


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)

    experiment_time_stamp = datetime.now().strftime("%d-%m-%y (%H:%M:%S.%f)")

    # Setup
    runs_per_configuration = 1
    generations_per_run = 123456
    cutoff_time = 150

    # General
    use_knowledge_compilation_caching = True
    variable_absence_bias = 2
    use_infeasibility = False

    # HASSLE-SLS
    perform_random_restarts = True
    recompute_random_incorrect_examples = False
    initialization_attempts = 1

    # Genetic algorithm configuration
    population_size = 20
    relative_tournament_size = 4/20
    crossover_rate = 0.5
    crossover_operator_configuration = [("smart_clause_crossover", True, "softmax", 1)]
    mutation_operator_configuration = [("mutate_hardness", 1, 0.05), ("mutate_clause_smarter", 1, None), ("mutate_weight", 1, 0.05)]
    use_local_search = False
    crowding = True
    crowding_variant = "semantic_relative"

    # What parameter is being varied? Used in file name
    varied_parameter = "num_context"

    pathname = path.dirname(path.realpath(__file__))
    pathname_contexts_and_data = pathname + "/pickles/contexts_and_data/"
    pathname_target_models = pathname + "/pickles/target_models/"

    # By 'problem structure pattern', we mean the specification of the problem - as it appears in the file name - but without
    # information about the model seed and context seed used to construct that problem
    problem_structure_patterns = set()
    for f in listdir(pathname_contexts_and_data):
        filename_without_extension = re.search(r"(.*)(\.pickle)", f).group(1)
        filename_without_seed_info = re.sub(r"(_model_seed_)(\d*)", '', filename_without_extension)
        filename_without_seed_info = re.sub(r"(_context_seed_)(\d*)", '', filename_without_seed_info)
        problem_structure_patterns.add(filename_without_seed_info)

    problem_structure_patterns = list(problem_structure_patterns)

    # Sort the problem structure patterns
    values_of_varied_parameter = [re.search(r"(_"+varied_parameter+r"_)(\d*)", problem_structure_pattern).group(2) for problem_structure_pattern in problem_structure_patterns]
    values_of_varied_parameter = [int(x) for x in values_of_varied_parameter]
    problem_structure_patterns = [x for _, x in sorted(zip(values_of_varied_parameter, problem_structure_patterns))]

    pathname_results_curr_file = f"{pathname}/results/{experiment_time_stamp}/varying_{varied_parameter}_num_of_files_per_pattern_{len(problem_structure_patterns)//len(listdir(pathname_contexts_and_data))}"
    makedirs(pathname_results_curr_file)
    
    # Running the experiment
    num_threads = mp.cpu_count()
    pool = mp.Pool(num_threads - 1)
    results = pool.map(run_experiment, problem_structure_patterns)

    all_observer_lists = [result[0] for result in results]
    all_learned_model_lists = [result[1] for result in results]
    all_target_model_lists = [result[2] for result in results]
    all_number_of_variables_lists = [result[3] for result in results]


    # Storing results
    with open(pathname_results_curr_file+"/all_observer_lists.pickle", 'wb') as handle:
        pickle.dump(all_observer_lists, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(pathname_results_curr_file+"/all_learned_model_lists.pickle", 'wb') as handle:
        pickle.dump(all_learned_model_lists, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(pathname_results_curr_file+"/all_target_model_lists.pickle", 'wb') as handle:
        pickle.dump(all_target_model_lists, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Further processing
    values_of_varied_parameter.sort()
    plot_average_final_entry_over_multiple_patterns(all_observer_lists,
                                                    pathname=pathname_results_curr_file + "/plot",
                                                    key="best_score",
                                                    varied_parameter="Number of contexts",
                                                    values_of_varied_parameter=values_of_varied_parameter,
                                                    # values_of_varied_parameter=[2*value for value in values_of_varied_parameter],
                                                    # values_of_varied_parameter=["Disjunctive\ncontexts", "Conjunctive\ncontexts"],
                                                    y_label="Score best model",
                                                    size=(2, 1.5), show_legend=False)
    plot_average_evaluation_statistics_over_multiple_patterns(all_learned_model_lists, all_target_model_lists, all_number_of_variables_lists,
                                                              pathname=pathname_results_curr_file + "/",
                                                              varied_parameter="Number of contexts",
                                                              values_of_varied_parameter=values_of_varied_parameter,
                                                              # values_of_varied_parameter=[2*value for value in values_of_varied_parameter],
                                                              # values_of_varied_parameter=["Disjunctive\ncontexts", "Conjunctive\ncontexts"],
                                                              size=(2, 1.5))


