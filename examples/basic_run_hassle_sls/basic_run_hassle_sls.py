import pickle
from datetime import datetime
from typing import List
from os import listdir, path
import numpy as np
import random

import re
from observe import SLSObserver
from hassle_sls.local_search import learn_weighted_max_sat

Individual = List[List]

if __name__ == "__main__":
    experiment_time_stamp = datetime.now().strftime("%d-%m-%y (%H:%M:%S.%f)")

    # General setup
    cutoff_time = 60
    use_infeasibility = False

    # HASSLE-SLS configuration
    use_knowledge_compilation_caching = True
    variable_absence_bias = 2
    perform_random_restarts = True
    recompute_random_incorrect_examples = False
    initialization_attempts = 20

    pathname = path.dirname(path.realpath(__file__))
    pathname_contexts_and_data = pathname + "/pickles/contexts_and_data/"
    f = listdir(pathname_contexts_and_data)[0]

    with open(pathname_contexts_and_data+f, "rb") as fp:
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

    instances_np = np.array(instances)
    labels_np = np.array([True if l == 1 else False for l in labels])

    if use_infeasibility:
        inf = [True if l == -1 else False for l in labels]
    else:
        inf = None

    seed = random.randint(0, 100000)

    observer = SLSObserver(cutoff_time=cutoff_time, use_infeasibility=use_infeasibility, use_knowledge_compilation_caching=use_knowledge_compilation_caching,
                           recompute_random_incorrect_examples=recompute_random_incorrect_examples,
                           legend_entry=f"HASSLE-SLS: walk_sat, use knowledge compilation: {use_knowledge_compilation_caching}")

    best_solution = learn_weighted_max_sat(k, n, instances_np, labels_np, contexts, "walk_sat", param="",
                                           inf=inf, cutoff_time=cutoff_time, seed=seed,
                                           use_knowledge_compilation=use_knowledge_compilation_caching,
                                           conjunctive_contexts=conjunctive_contexts,
                                           initialization_attempts=initialization_attempts,
                                           variable_absence_bias=variable_absence_bias,
                                           recompute_random_incorrect_examples=recompute_random_incorrect_examples,
                                           perform_random_restarts=perform_random_restarts,
                                           observers=[observer])
    print(best_solution)





