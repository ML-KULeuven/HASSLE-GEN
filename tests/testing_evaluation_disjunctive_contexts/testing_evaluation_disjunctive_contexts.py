import pickle
from os import listdir, path, makedirs
import random
import numpy as np
import re
from auxiliary import to_phenotype, MaxSAT_to_genotype
from evaluation import evaluate, evaluate_knowledge_compilation_based_dispatch
from hassle.local_search import random_model


def make_individual(n, k):
    clauses = []
    for i in range(k):
        literals = [random.choices([-1, 0, 1], weights=[1, 2, 1])[0] for _ in range(n)]
        hard = random.random() < 0.5
        w = random.random()
        clauses.append(literals + [hard, w])
    return clauses


if __name__ == "__main__":

    pathname = path.dirname(path.realpath(__file__))
    pathname_contexts_and_data = pathname + "/pickles/contexts_and_data/"
    f = listdir(pathname_contexts_and_data)[0]
    with open(pathname_contexts_and_data+f, "rb") as fp:
        pickle_var = pickle.load(fp)

    # n = number of variables, k = number of constraints
    n = int(re.search(r"(_n_)(\d*)", f).group(2))
    k = int(re.search(r"(_num_hard_)(\d*)", f).group(2)) + int(re.search(r"(_num_soft_)(\d*)", f).group(2))
    conjunctive_contexts = 0

    contexts, instances, labels = (
        [{int(str(lit)) for lit in context} for context in pickle_var["contexts"]],
        pickle_var["data"],
        pickle_var["labels"],
    )

    instances_np = np.array(instances)
    labels_np = np.array([True if l == 1 else False for l in labels])
    inf = None
    examples_evolutionary = list(zip(contexts, instances, labels))
    examples_hassle = list(zip(contexts, instances, labels_np))

    number_of_individuals_to_test = 10
    model_list = []
    seed = random.randint(0, 100000)
    rng = np.random.RandomState(seed)
    for _ in range(number_of_individuals_to_test):
        model = random_model(instances_np.shape[1], k, n, rng)
        model_list.append(model)

    for model in model_list:
        score_list = []
        score_list.append(int(round(evaluate(to_phenotype(MaxSAT_to_genotype(model)), examples_evolutionary, False, 1, False, conjunctive_contexts)[0] * len(examples_evolutionary))))
        score_list.append(int(round(evaluate(to_phenotype(MaxSAT_to_genotype(model)), examples_evolutionary, True, 1, False, conjunctive_contexts)[0] * len(examples_evolutionary))))
        score_list.append(int(round(evaluate(to_phenotype(MaxSAT_to_genotype(model)), examples_evolutionary, True, 2, False, conjunctive_contexts)[0] * len(examples_evolutionary))))
        score_list.append(int(round(evaluate(to_phenotype(MaxSAT_to_genotype(model)), examples_evolutionary, True, 3, False, conjunctive_contexts)[0] * len(examples_evolutionary))))
        score_list.append(int(round(evaluate(to_phenotype(MaxSAT_to_genotype(model)), examples_evolutionary, True, 4, False, conjunctive_contexts)[0] * len(examples_evolutionary))))
        score_list.append(int(round(evaluate(to_phenotype(MaxSAT_to_genotype(model)), examples_evolutionary, True, 1, True, conjunctive_contexts)[0] * len(examples_evolutionary))))
        score_list.append(int(round(evaluate(to_phenotype(MaxSAT_to_genotype(model)), examples_evolutionary, True, 2, True, conjunctive_contexts)[0] * len(examples_evolutionary))))
        score_list.append(int(round(evaluate(to_phenotype(MaxSAT_to_genotype(model)), examples_evolutionary, True, 3, True, conjunctive_contexts)[0] * len(examples_evolutionary))))
        score_list.append(int(round(evaluate(to_phenotype(MaxSAT_to_genotype(model)), examples_evolutionary, True, 4, True, conjunctive_contexts)[0] * len(examples_evolutionary))))
        score_list.append(model.score(instances_np, labels_np, contexts, inf=inf, conjunctive_contexts=conjunctive_contexts)[0])
        score_list.append(int(round(evaluate_knowledge_compilation_based_dispatch(to_phenotype(MaxSAT_to_genotype(model)), examples_hassle, knowledge_compilation_variant=1, use_diagram_for_instance_evaluation=False, conjunctive_contexts=conjunctive_contexts, inf=inf)[0] * len(examples_hassle))))
        score_list.append(int(round(evaluate_knowledge_compilation_based_dispatch(to_phenotype(MaxSAT_to_genotype(model)), examples_hassle, knowledge_compilation_variant=2, use_diagram_for_instance_evaluation=False, conjunctive_contexts=conjunctive_contexts, inf=inf)[0] * len(examples_hassle))))
        score_list.append(int(round(evaluate_knowledge_compilation_based_dispatch(to_phenotype(MaxSAT_to_genotype(model)), examples_hassle, knowledge_compilation_variant=3, use_diagram_for_instance_evaluation=False, conjunctive_contexts=conjunctive_contexts, inf=inf)[0] * len(examples_hassle))))
        score_list.append(int(round(evaluate_knowledge_compilation_based_dispatch(to_phenotype(MaxSAT_to_genotype(model)), examples_hassle, knowledge_compilation_variant=4, use_diagram_for_instance_evaluation=False, conjunctive_contexts=conjunctive_contexts, inf=inf)[0] * len(examples_hassle))))
        score_list.append(int(round(evaluate_knowledge_compilation_based_dispatch(to_phenotype(MaxSAT_to_genotype(model)), examples_hassle, knowledge_compilation_variant=1, use_diagram_for_instance_evaluation=True, conjunctive_contexts=conjunctive_contexts, inf=inf)[0] * len(examples_hassle))))
        score_list.append(int(round(evaluate_knowledge_compilation_based_dispatch(to_phenotype(MaxSAT_to_genotype(model)), examples_hassle, knowledge_compilation_variant=2, use_diagram_for_instance_evaluation=True, conjunctive_contexts=conjunctive_contexts, inf=inf)[0] * len(examples_hassle))))
        score_list.append(int(round(evaluate_knowledge_compilation_based_dispatch(to_phenotype(MaxSAT_to_genotype(model)), examples_hassle, knowledge_compilation_variant=3, use_diagram_for_instance_evaluation=True, conjunctive_contexts=conjunctive_contexts, inf=inf)[0] * len(examples_hassle))))
        score_list.append(int(round(evaluate_knowledge_compilation_based_dispatch(to_phenotype(MaxSAT_to_genotype(model)), examples_hassle, knowledge_compilation_variant=4, use_diagram_for_instance_evaluation=True, conjunctive_contexts=conjunctive_contexts, inf=inf)[0] * len(examples_hassle))))
        print(score_list)
        if len(np.unique(score_list)) > 1:
            print(model)
            raise ValueError("Some evaluations of the same individual gave different results")

    pathname_target_model = pathname + "/pickles/target_models/"
    f = listdir(pathname_target_model)[0]
    with open(pathname_target_model + f, "rb") as fp:
        pickle_var = pickle.load(fp)
    target_model = pickle_var['true_model']

    score_list = []
    score_list.append(int(round(evaluate(target_model, examples_evolutionary, False, 1, False, conjunctive_contexts)[0] * len(examples_evolutionary))))
    score_list.append(int(round(evaluate(target_model, examples_evolutionary, True, 1, False, conjunctive_contexts)[0] * len(examples_evolutionary))))
    score_list.append(int(round(evaluate(target_model, examples_evolutionary, True, 2, False, conjunctive_contexts)[0] * len(examples_evolutionary))))
    score_list.append(int(round(evaluate(target_model, examples_evolutionary, True, 3, False, conjunctive_contexts)[0] * len(examples_evolutionary))))
    score_list.append(int(round(evaluate(target_model, examples_evolutionary, True, 4, False, conjunctive_contexts)[0] * len(examples_evolutionary))))
    score_list.append(int(round(evaluate(target_model, examples_evolutionary, True, 1, True, conjunctive_contexts)[0] * len(examples_evolutionary))))
    score_list.append(int(round(evaluate(target_model, examples_evolutionary, True, 2, True, conjunctive_contexts)[0] * len(examples_evolutionary))))
    score_list.append(int(round(evaluate(target_model, examples_evolutionary, True, 3, True, conjunctive_contexts)[0] * len(examples_evolutionary))))
    score_list.append(int(round(evaluate(target_model, examples_evolutionary, True, 4, True, conjunctive_contexts)[0] * len(examples_evolutionary))))
    score_list.append(int(round(evaluate_knowledge_compilation_based_dispatch(target_model, examples_hassle, knowledge_compilation_variant=1, use_diagram_for_instance_evaluation=False, conjunctive_contexts=conjunctive_contexts, inf=inf)[0] * len(examples_hassle))))
    score_list.append(int(round(evaluate_knowledge_compilation_based_dispatch(target_model, examples_hassle, knowledge_compilation_variant=2, use_diagram_for_instance_evaluation=False, conjunctive_contexts=conjunctive_contexts, inf=inf)[0] * len(examples_hassle))))
    score_list.append(int(round(evaluate_knowledge_compilation_based_dispatch(target_model, examples_hassle, knowledge_compilation_variant=3, use_diagram_for_instance_evaluation=False, conjunctive_contexts=conjunctive_contexts, inf=inf)[0] * len(examples_hassle))))
    score_list.append(int(round(evaluate_knowledge_compilation_based_dispatch(target_model, examples_hassle, knowledge_compilation_variant=4, use_diagram_for_instance_evaluation=False, conjunctive_contexts=conjunctive_contexts, inf=inf)[0] * len(examples_hassle))))
    score_list.append(int(round(evaluate_knowledge_compilation_based_dispatch(target_model, examples_hassle, knowledge_compilation_variant=1, use_diagram_for_instance_evaluation=True, conjunctive_contexts=conjunctive_contexts, inf=inf)[0] * len(examples_hassle))))
    score_list.append(int(round(evaluate_knowledge_compilation_based_dispatch(target_model, examples_hassle, knowledge_compilation_variant=2, use_diagram_for_instance_evaluation=True, conjunctive_contexts=conjunctive_contexts, inf=inf)[0] * len(examples_hassle))))
    score_list.append(int(round(evaluate_knowledge_compilation_based_dispatch(target_model, examples_hassle, knowledge_compilation_variant=3, use_diagram_for_instance_evaluation=True, conjunctive_contexts=conjunctive_contexts, inf=inf)[0] * len(examples_hassle))))
    score_list.append(int(round(evaluate_knowledge_compilation_based_dispatch(target_model, examples_hassle, knowledge_compilation_variant=4, use_diagram_for_instance_evaluation=True, conjunctive_contexts=conjunctive_contexts, inf=inf)[0] * len(examples_hassle))))
    print(score_list)
    if len(np.unique(score_list)) > 1 or any(score != len(examples_evolutionary) for score in score_list):
        print(target_model)
        raise ValueError("Some evaluations of the same individual gave different results")
