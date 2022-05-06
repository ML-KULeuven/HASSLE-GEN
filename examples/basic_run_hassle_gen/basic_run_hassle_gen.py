import pickle
import re
from datetime import datetime
from os import listdir, path
from observe import TrackObserver
from learn_max_sat_model import learn_max_sat_model

if __name__ == "__main__":
    experiment_time_stamp = datetime.now().strftime("%d-%m-%y (%H:%M:%S.%f)")

    # General setup
    generations_per_run = 1000
    cutoff_time = 1800
    pathname = path.dirname(path.realpath(__file__))
    pathname_contexts_and_data = pathname + "/pickles/contexts_and_data/"

    # HASSLE-GEN configuration
    population_size = 20
    relative_tournament_size = 4/20 # Only relevant when use_crowding == False
    tournament_size = round(relative_tournament_size * population_size)
    crossover_rate = 0.5
    crossover_operator_configuration = [("smart_clause_crossover", True, "softmax", 1)]
    mutation_operator_configuration = [("mutate_hardness", 1, 0.05),
                                       ("mutate_clause_smarter", 1, None),
                                       ("mutate_weight", 1, 0.05)]
    use_local_search = False
    use_knowledge_compilation_caching = True
    use_infeasibility = False
    variable_absence_bias = 2
    use_crowding = True
    crowding_variant = "semantic_relative"

    f = listdir(pathname_contexts_and_data)[0]
    filename_without_extension = re.search(r"(.*)(\.pickle)", f).group(1)
    filename_without_seed_info = re.sub(r"(_model_seed_)(\d*)", '', filename_without_extension)
    filename_without_seed_info = re.sub(r"(_context_seed_)(\d*)", '', filename_without_seed_info)

    # Open pickle file
    with open(pathname_contexts_and_data + f, "rb") as fp:
        pickle_var = pickle.load(fp)

    # n = number of variables, k = number of constraints
    n = int(re.search(r"(_n_)(\d*)", f).group(2))
    k = int(re.search(r"(_num_hard_)(\d*)", f).group(2)) + int(re.search(r"(_num_soft_)(\d*)", f).group(2))
    match = re.search(r"(_conjunctive_contexts_)(\d*)", f)
    if match is not None:
        conjunctive_contexts = int(match.group(2))
    else:
        conjunctive_contexts = 0

    print(f"Starting experiment on file {f}\nwith number of variables: {n} and number of constraints: {k}")
    contexts, instances, labels = (
        [{int(str(lit)) for lit in context} for context in pickle_var["contexts"]],
        pickle_var["data"],
        pickle_var["labels"],
    )

    examples = list(zip(contexts, instances, labels))

    observer = TrackObserver(max_generations=generations_per_run,
                             population_size=population_size,
                             tournament_size=tournament_size,
                             crossover_operators=crossover_operator_configuration,
                             mutation_operators=mutation_operator_configuration,
                             crossover_rate=crossover_rate,
                             replacement_strategy=None,
                             use_local_search=use_local_search,
                             cutoff_time=cutoff_time,
                             legend_entry=f"Crossover prob: {crossover_rate},"
                                          f"mutation:{mutation_operator_configuration}",
                             use_knowledge_compilation_caching=use_knowledge_compilation_caching,
                             use_infeasibility=use_infeasibility,
                             variable_absence_bias=variable_absence_bias,
                             crowding_variant=crowding_variant if use_crowding else None)

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
                            use_crowding=use_crowding,
                            crowding_variant=crowding_variant,
                            observers=[observer])

    print(f"In {cumulative_duration} seconds, the best score achieved was {best_score}")
