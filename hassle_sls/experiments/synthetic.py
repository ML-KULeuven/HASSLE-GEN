import logging
import pickle
import csv
import argparse
import itertools as it
import numpy as np
from datetime import datetime
import os
import json
import time
from tqdm import tqdm
import random

from hassle_sls.type_def import MaxSatModel, Context
from hassle_sls.generator import generate_models, generate_contexts_and_data
from hassle_sls.pysat_solver import solve_weighted_max_sat, get_value, label_instance
from hassle_sls.local_search import learn_weighted_max_sat
from hassle_sls.milp_learner import learn_weighted_max_sat_MILP
from hassle_sls.verify import get_recall_precision_wmc, get_infeasibility_wmc


def generate(args):
    iterations = (
        len(args.num_vars)
        * len(args.num_hard)
        * len(args.num_soft)
        * len(args.model_seeds)
        * len(args.num_context)
        * len(args.context_seeds)
    )
    bar = tqdm(total=iterations)
    for n, h, s, seed in it.product(
        args.num_vars, args.num_hard, args.num_soft, args.model_seeds
    ):
        model, param = generate_models(n, int(n / 2), h, s, seed)

        for c, context_seed in it.product(args.num_context, args.context_seeds):
            tag = generate_contexts_and_data(
                n,
                model,
                c,
                args.num_pos,
                args.num_neg,
                args.neg_type,
                param,
                context_seed,
            )
            tqdm.write(tag)
            bar.update(1)


def learn(args):
    iterations = (
        len(args.num_vars)
        * len(args.num_hard)
        * len(args.num_soft)
        * len(args.model_seeds)
        * len(args.num_context)
        * len(args.context_seeds)
        * len(args.method)
        * len(args.cutoff)
        * len(args.noise)
    )
    bar = tqdm(total=iterations)
    for n, h, s, seed, c, context_seed, m, t, p in it.product(
        args.num_vars,
        args.num_hard,
        args.num_soft,
        args.model_seeds,
        args.num_context,
        args.context_seeds,
        args.method,
        args.cutoff,
        args.noise,
    ):
        param = f"_n_{n}_max_clause_length_{int(n / 2)}_num_hard_{h}_num_soft_{s}_model_seed_{seed}_num_context_{c}_num_pos_{args.num_pos}_num_neg_{args.num_neg}_context_seed_{context_seed}"
        if args.neg_type:
            param = f"_n_{n}_max_clause_length_{int(n / 2)}_num_hard_{h}_num_soft_{s}_model_seed_{seed}_num_context_{c}_num_pos_{args.num_pos}_num_neg_{args.num_neg}_neg_type_{args.neg_type}_context_seed_{context_seed}"
        if m == "MILP":
            try:
                learn_model_MILP(h + s, m, t, param, p)
            except FileNotFoundError:
                continue
        else:
            try:
                learn_model_sls(h + s, m, t, param, p, args.naive, args.clause_len)
            except FileNotFoundError:
                continue
        bar.update(1)


def evaluate(args, bl):
    iterations = (
        len(args.num_vars)
        * len(args.num_hard)
        * len(args.num_soft)
        * len(args.model_seeds)
        * len(args.num_context)
        * len(args.context_seeds)
        * len(args.method)
        * len(args.cutoff)
        * len(args.noise)
    )
    bar = tqdm(total=iterations)
    folder_name = datetime.now().strftime("%d-%m-%y (%H:%M:%S.%f)")
    os.makedirs(f"results/{folder_name}")
    with open(f"results/{folder_name}/arguments.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)
    csvfile = open(f"results/{folder_name}/evaluation.csv", "w")
    filewriter = csv.writer(csvfile, delimiter=",")
    filewriter.writerow(
        [
            "num_vars",
            "num_hard",
            "num_soft",
            "model_seed",
            "num_context",
            "context_seed",
            "num_pos",
            "num_neg",
            "pos_per_context",
            "neg_per_context",
            "method",
            "score",
            "recall",
            "precision",
            "accuracy",
            "f1_score",
            "regret",
            "infeasiblity",
            "time_taken",
            "cutoff",
            "noise_probability",
            "iterations",
            "neighbours",
        ]
    )
    for n, h, s, seed in it.product(
        args.num_vars, args.num_hard, args.num_soft, args.model_seeds
    ):
        param = f"_n_{n}_max_clause_length_{int(n/2)}_num_hard_{h}_num_soft_{s}_model_seed_{seed}"
        target_model = pickle.load(
            open("pickles/target_model/" + param + ".pickle", "rb")
        )["true_model"]
        max_t = max(args.cutoff)
        for c, context_seed in it.product(args.num_context, args.context_seeds):
            tag_cnd = (
                param
                + f"_num_context_{c}_num_pos_{args.num_pos}_num_neg_{args.num_neg}_context_seed_{context_seed}"
            )
            if args.neg_type:
                tag_cnd = (
                    param
                    + f"_num_context_{c}_num_pos_{args.num_pos}_num_neg_{args.num_neg}_neg_type_{args.neg_type}_context_seed_{context_seed}"
                )
            pickle_cnd = pickle.load(
                open("pickles/contexts_and_data/" + tag_cnd + ".pickle", "rb")
            )
            for m, p in it.product(args.method, args.noise):
                tag = tag_cnd + f"_method_{m}_cutoff_{max_t}_noise_{p}"
                if args.naive == 1:
                    tag += "_naive"
                if bl == 1:
                    tag += "_bl"
                pickle_var = pickle.load(
                    open("pickles/learned_model/" + tag + ".pickle", "rb")
                )
                if c == 0:
                    c = 1
                labels = [True if l == 1 else False for l in pickle_cnd["labels"]]
                pos_per_context = labels.count(True) / c
                neg_per_context = labels.count(False) / c
                last_index = -2
                recall, precision, accuracy = (-1, -1, -1)
                regret, infeasiblity, f1_score = (-1, -1, -1)
                for t in args.cutoff:
                    index = get_learned_model(pickle_var["time_taken"], max_t, t)
                    learned_model = None
                    time_taken = t
                    iteration = 0
                    num_nbr = 0
                    score = -1
                    if index is not None:
                        learned_model = pickle_var["learned_model"][index]
                        time_taken = pickle_var["time_taken"][index]
                        if m != "MILP":
                            iteration = pickle_var["iterations"][index]
                            num_nbr = pickle_var["num_neighbour"][index]
                        if learned_model:
                            score = pickle_var["score"][index]

                    if index != last_index:
                        last_index = index
                        contexts = pickle_cnd["contexts"]
                        global_context = set()
                        for context in contexts:
                            global_context.update(context)
                        if learned_model:
                            recall, precision, accuracy, regret, infeasiblity = evaluate_statistics(
                                n, target_model, learned_model, global_context
                            )
                        f1_score = 2 * recall * precision / (recall + precision)
                    filewriter.writerow(
                        [
                            n,
                            h,
                            s,
                            seed,
                            c,
                            context_seed,
                            args.num_pos,
                            args.num_neg,
                            pos_per_context,
                            neg_per_context,
                            m,
                            score,
                            recall,
                            precision,
                            accuracy,
                            f1_score,
                            regret,
                            infeasiblity,
                            time_taken,
                            t,
                            p,
                            iteration,
                            num_nbr,
                        ]
                    )
                    bar.update(1)

    csvfile.close()


def learn_model_sls(num_constraints, method, cutoff, param, p, naive=0, clause_len=0):
    pickle_var = pickle.load(
        open("pickles/contexts_and_data/" + param + ".pickle", "rb")
    )

    param += f"_method_{method}_cutoff_{cutoff}_noise_{p}"
    if naive == 1:
        param += "_naive"
    if os.path.exists("pickles/learned_model/" + param + ".pickle"):
        pickle_var = pickle.load(
            open("pickles/learned_model/" + param + ".pickle", "rb")
        )
        tqdm.write("Exists: " + param + "\n")
        return pickle_var["learned_model"], pickle_var["time_taken"]

    data = np.array(pickle_var["data"])
    labels = np.array(pickle_var["labels"])
    contexts = pickle_var["contexts"]

    inf = [True if l == -1 else False for l in labels]
    labels = np.array([True if l == 1 else False for l in labels])

    if p != 0:
        rng = np.random.RandomState(111)
        for i, label in enumerate(labels):
            if rng.random_sample() < p:
                labels[i] = not label

    if clause_len == 0:
        clause_len = data.shape[1]

    return learn_weighted_max_sat(
        num_constraints,
        clause_len,
        data,
        labels,
        contexts,
        method,
        param,
        inf,
        cutoff_time=cutoff,
    )


def learn_model_MILP(num_constraints, method, cutoff, param, p):
    pickle_var = pickle.load(
        open("pickles/contexts_and_data/" + param + ".pickle", "rb")
    )

    param += f"_method_{method}_cutoff_{cutoff}_noise_{p}"
    if os.path.exists("pickles/learned_model/" + param + ".pickle"):
        pickle_var = pickle.load(
            open("pickles/learned_model/" + param + ".pickle", "rb")
        )
        tqdm.write("Exists: " + param + ": " + str(pickle_var["score"]) + "\n")
        return pickle_var["learned_model"], pickle_var["time_taken"]

    data = np.array(pickle_var["data"])
    labels = np.array(pickle_var["labels"])
    contexts = pickle_var["contexts"]

    # inf = [True if l == -1 else False for l in labels]
    labels = np.array([True if l == 1 else False for l in labels])

    if p != 0:
        rng = np.random.RandomState(111)
        for i, label in enumerate(labels):
            if rng.random_sample() < p:
                labels[i] = not label

    start = time.time()
    learned_model = learn_weighted_max_sat_MILP(
        num_constraints, data, labels, contexts, cutoff
    )
    end = time.time()

    score = 0
    if learned_model:
        for k in range(data.shape[0]):
            instance = data[k, :]
            label = labels[k]
            learned_label = label_instance(learned_model, instance, contexts[k])
            if label == learned_label:
                score += 1
    pickle_var = {}
    pickle_var["learned_model"] = [learned_model]
    pickle_var["time_taken"] = [end - start]
    pickle_var["score"] = [score * 100 / data.shape[0]]
    if not os.path.exists("pickles/learned_model"):
        os.makedirs("pickles/learned_model")
    pickle.dump(pickle_var, open("pickles/learned_model/" + param + ".pickle", "wb"))
    # tqdm.write(param + ": " + str(pickle_var["score"]) + "\n")
    return learned_model, end - start


def evaluate_statistics(
    n, target_model: MaxSatModel, learned_model: MaxSatModel, context: Context
):
    recall, precision, accuracy = get_recall_precision_wmc(
        n, target_model, learned_model, context
    )
    reg = regret(n, target_model, learned_model, context)
    infeasiblity = get_infeasibility_wmc(n, target_model, learned_model, context)

    return recall, precision, accuracy, reg, infeasiblity


def regret(n, target_model, learned_model, context):
    learned_feasible_model = learned_model.copy()
    for w, clause in target_model:
        if w is None:
            learned_feasible_model.append((w, clause))
    sol, cost = solve_weighted_max_sat(n, target_model, context, 1)
    opt_val = get_value(target_model, sol, context)
    avg_regret = 0
    learned_sols, cost = solve_weighted_max_sat(
        n, learned_feasible_model, context, 1000
    )
    if len(learned_sols) == 0:
        return -1
    for learned_sol in learned_sols:
        if len(learned_sol) < n:
            for i in range(len(learned_sol), n):
                learned_sol.append(random.random() < 0.5)
        learned_opt_val = get_value(target_model, learned_sol, context)
        if learned_opt_val is None or learned_opt_val > opt_val:
            raise Exception("error: calculating regret")
        regret = (opt_val - learned_opt_val) / opt_val
        avg_regret += regret
    return avg_regret * 100 / len(learned_sols)


def get_learned_model(time_taken, max_cutoff, cutoff):
    if cutoff == max_cutoff:
        return -1
    elif cutoff < max_cutoff:
        ind = 0
        for index, t in enumerate(time_taken):
            if t <= cutoff:
                ind = 1
            elif cutoff < t and ind == 1:
                return index - 1
            elif cutoff < t and ind == 0:
                return None
    return index


logger = logging.getLogger(__name__)
if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--function", type=str, default="evaluate")
    CLI.add_argument("--num_vars", nargs="*", type=int, default=[10])
    CLI.add_argument("--num_hard", nargs="*", type=int, default=[10])
    CLI.add_argument("--num_soft", nargs="*", type=int, default=[10])
    CLI.add_argument(
        "--model_seeds", nargs="*", type=int, default=[111, 222, 333, 444, 555]
    )
    CLI.add_argument("--num_context", nargs="*", type=int, default=[100])
    CLI.add_argument(
        "--context_seeds", nargs="*", type=int, default=[111, 222, 333, 444, 555]
    )
    CLI.add_argument("--num_pos", type=int, default=2)
    CLI.add_argument("--num_neg", type=int, default=2)
    CLI.add_argument("--neg_type", type=str, default="both")
    CLI.add_argument(
        "--method",
        nargs="*",
        type=str,
        default=[
            "walk_sat",
            "novelty",
            "novelty_plus",
            "adaptive_novelty_plus",
            "MILP",
        ],
    )
    CLI.add_argument(
        "--cutoff", nargs="*", type=int, default=[60, 300, 600, 900, 1200, 1500, 1800]
    )
    CLI.add_argument("--noise", nargs="*", type=float, default=[0.05, 0.1, 0.2])
    CLI.add_argument("--weighted", type=int, default=1)
    CLI.add_argument("--naive", type=int, default=0)
    CLI.add_argument("--clause_len", type=int, default=0)

    args = CLI.parse_args()

    if args.function == "generate":
        generate(args)

    elif args.function == "learn":
        learn(args)

    elif args.function == "evaluate":
        evaluate(args, 0)
