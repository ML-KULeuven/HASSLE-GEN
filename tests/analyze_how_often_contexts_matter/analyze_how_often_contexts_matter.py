import pickle
import numpy as np
from os import listdir, path
from hassle.pysat_solver import get_value, solve_weighted_max_sat
from hassle.type_def import MaxSatModel, Clause, Instance, Context
import auxiliary
import genetic
import re
"""
The provided model should be in phenotype representation
"""

def get_example_indices_where_context_affects_best_value_possible(model, examples, conjunctive_contexts=0):
    relevant_indices = []
    for i in range(len(examples)):
        context, instance, label = examples[i]
        best_instance_with_context, cst1 = solve_weighted_max_sat(len(instance), model, context, 1, conjunctive_contexts=conjunctive_contexts)
        best_instance_without_context, cst2 = solve_weighted_max_sat(len(instance), model, set(), 1, conjunctive_contexts=conjunctive_contexts)
        if cst1 < 0:
            best_value_with_context = None
        else:
            best_value_with_context = get_value(model, best_instance_with_context, context, conjunctive_contexts=conjunctive_contexts)
        if cst2 < 0:
            best_value_without_context = None
        else:
            best_value_without_context = get_value(model, best_instance_without_context, set(), conjunctive_contexts=conjunctive_contexts)
        if best_value_with_context != best_value_without_context:
            relevant_indices.append(i)
    return relevant_indices

if __name__ == "__main__":
    pathname = path.dirname(path.realpath(__file__))
    pathname_contexts_and_data = pathname + "/pickles/contexts_and_data/"

    problem_structure_patterns = set()
    for f in listdir(pathname_contexts_and_data):
        filename_without_extension = re.search(r"(.*)(\.pickle)", f).group(1)
        filename_without_seed_info = re.sub(r"(_model_seed_)(\d*)", '', filename_without_extension)
        filename_without_seed_info = re.sub(r"(_context_seed_)(\d*)", '', filename_without_seed_info)
        problem_structure_patterns.add(filename_without_seed_info)

    for problem_structure_pattern in problem_structure_patterns:
        # Find all other context-and-data files with this pattern, i.e., that differ merely in their model or context seeds
        files_with_this_pattern = []
        for f in listdir(pathname_contexts_and_data):
            filename_without_extension = re.search(r"(.*)(\.pickle)", f).group(1)
            filename_without_seed_info = re.sub(r"(_model_seed_)(\d*)", '', filename_without_extension)
            filename_without_seed_info = re.sub(r"(_context_seed_)(\d*)", '', filename_without_seed_info)
            if filename_without_seed_info == problem_structure_pattern:
                files_with_this_pattern.append(f)

        proportions = []
        for f in files_with_this_pattern:
            #print(f"\n\n\n\n\n NEW FILE: {f}")
            n = int(re.search(r"(_n_)(\d*)", f).group(2))
            match = re.search(r"(_conjunctive_contexts_)(\d*)", f)
            if match is not None:
                conjunctive_contexts = int(match.group(2))
            else:
                conjunctive_contexts = 0

            with open(pathname_contexts_and_data + f, "rb") as fp:
                pickle_var = pickle.load(fp)

            split_result = f.split("_num_context")
            f_model = pathname+"/pickles/target_models/" + split_result[0] + ".pickle"
            with open(f_model, "rb") as fp:
                target_model_phenotype = pickle.load(fp)['true_model']

            contexts, instances, labels = (
                [{int(str(lit)) for lit in context} for context in pickle_var["contexts"]],
                pickle_var["data"],
                pickle_var["labels"],
            )
            examples = list(zip(contexts, instances, labels))


            examples_indices_context_affects_best_value_target_model = get_example_indices_where_context_affects_best_value_possible(target_model_phenotype, examples, conjunctive_contexts=conjunctive_contexts)
            proportions.append(len(examples_indices_context_affects_best_value_target_model) * 100 / len(examples))
            #print(f"The examples for which the context changes the best value possible w.r.t. the target model are: {examples_indices_context_affects_best_value_target_model}")
            #print(f"This is {len(examples_indices_context_affects_best_value_target_model)*100/len(examples)}% of examples")
        print(f"For pattern: {problem_structure_pattern}")
        print(f"{sum(proportions)/len(proportions)}% of contexts affected the best value possible w.r.t. the target model")
        print(f"Standard deviation: {np.std(proportions)}\n\n")