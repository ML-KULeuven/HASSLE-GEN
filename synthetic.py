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
from generator import generate_models, generate_contexts_and_data



def generate(args):
    iterations = (
        len(args.num_vars)
        * len(args.num_hard)
        * len(args.num_soft)
        * len(args.model_seeds)
        * len(args.num_context)
        * len(args.context_seeds)
        * len(args.conjunctive_contexts)
        * len(args.use_new_generation_method)
    )
    bar = tqdm(total=iterations)
    for n, h, s, seed, use_new_gen_method in it.product(
        args.num_vars, args.num_hard, args.num_soft, args.model_seeds, args.use_new_generation_method
    ):
        rng = np.random.RandomState(seed)
        success = False
        while not success:
            # model, param = generate_models(n, int(n / 2), h, s, seed, use_new_gen_method)
            model, param = generate_models(n, 5, h, s, seed, rng, use_new_gen_method)

            context_params = []
            context_pickle_vars = []
            for c, conjunctive_contexts, context_seed in it.product(args.num_context, args.conjunctive_contexts, args.context_seeds):
                context_param, context_pickle_var = generate_contexts_and_data(
                    n,
                    model,
                    c,
                    args.num_pos,
                    args.num_neg,
                    args.neg_type,
                    param,
                    rng,
                    context_seed,
                    conjunctive_contexts
                )
                context_params.append(context_param)
                context_pickle_vars.append(context_pickle_var)

            if not any([cont_param is None for cont_param in context_params]):
                bar.update(len(context_params))
                # Write out ground-truth model
                pickle_var = {}
                pickle_var["true_model"] = model
                if not os.path.exists("pickles/target_models"):
                    os.makedirs("pickles/target_models")
                pickle.dump(pickle_var, open("pickles/target_models/" + param + ".pickle", "wb"))
                for i in range(len(context_params)):
                    cont_pickle_var = context_pickle_vars[i]
                    if not os.path.exists("pickles/contexts_and_data"):
                        os.makedirs("pickles/contexts_and_data")
                    pickle.dump(
                        cont_pickle_var, open("pickles/contexts_and_data/" + context_params[i] + ".pickle", "wb")
                    )
                success = True
            else:
                print("Failed to generate model + data, trying again\n")

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
    CLI.add_argument("--conjunctive_contexts", nargs="*", type=int, default=[0])
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
    CLI.add_argument("--use_new_generation_method", nargs="*", type=int, default=1)

    args = CLI.parse_args()

    if args.function == "generate":
        generate(args)
