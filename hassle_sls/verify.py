import numpy as np
from pysdd.sdd import Vtree, SddManager

from .pysat_solver import solve_weighted_max_sat, get_value, label_instance
from .type_def import MaxSatModel, Context, Clause


def find_solutions_rec(weights, selected, available, budget):
    if budget <= min(weights) / 10:
        yield selected

    for i in available:
        s = selected | {i}
        b = budget - weights[i]
        a = [j for j in available if j > i and weights[j] <= b + min(weights) / 10]
        yield from find_solutions_rec(weights, s, a, b)


def find_weight_assignments(weights, budget):
    weights = np.array(weights)
    selected = set()
    available = [i for i in range(len(weights)) if weights[i] <= budget]
    return find_solutions_rec(weights, selected, available, budget)


def clause_to_sdd(clause: Clause, manager: SddManager):
    result = manager.false()
    for i in clause:
        if i > 0:
            result |= manager.literal(i)
        else:
            result |= ~manager.literal(abs(i))
    return result


def get_sdd_manager(n: int):
    vtree = Vtree(n, list(range(1, n + 1)), "balanced")
    return SddManager.from_vtree(vtree)


def convert_to_logic(manager: SddManager, n: int, model: MaxSatModel, context: Context):
    best_solution, cst = solve_weighted_max_sat(n, model, context, 1)

    if cst is -1:
        return None
    else:
        value = get_value(model, best_solution, context)
        hard_constraints = [c for w, c in model if not w]

        hard_result = manager.true()
        for clause in hard_constraints:
            hard_result &= clause_to_sdd(clause, manager)

        soft_constraints = [(w, c) for w, c in model if w is not None]
        if len(soft_constraints) > 0:
            weights = [t[0] for t in soft_constraints]
            assignments = find_weight_assignments(weights, value)
            soft_result = manager.false()
            for assignment in assignments:
                assignment_result = manager.true()
                for i in assignment:
                    assignment_result &= clause_to_sdd(soft_constraints[i][1], manager)
                soft_result |= assignment_result
                if assignment == {i for i in range(len(soft_constraints))}:
                    break
            return hard_result & soft_result
        return hard_result


def count_solutions(n: int, model: MaxSatModel, context: Context):
    # TODO What we actually want: optimal across all possible contexts!

    manager = get_sdd_manager(n)
    logic = convert_to_logic(manager, n, model, context)
    if logic is None:
        return 0

    return logic.global_model_count()


def get_recall_precision_wmc(
    n: int, true_model: MaxSatModel, learned_model: MaxSatModel, context: Context
):
    manager = get_sdd_manager(n)
    true_logic = convert_to_logic(manager, n, true_model, context)
    learned_logic = convert_to_logic(manager, n, learned_model, context)
    combined = true_logic & learned_logic

    true_count, learned_count, combined_count = (
        l.global_model_count() for l in (true_logic, learned_logic, combined)
    )
    if learned_count == 0:
        return -1, -1, -1
    TN = pow(2, n) - (true_count + learned_count - combined_count)
    accuracy = (TN + combined_count) * 100 / pow(2, n)
    recall = combined_count * 100 / true_count
    precision = combined_count * 100 / learned_count
    return recall, precision, accuracy


def get_infeasibility_wmc(
    n: int, true_model: MaxSatModel, learned_model: MaxSatModel, context: Context
):
    manager = get_sdd_manager(n)
    true_feasible_model = []
    for w, clause in true_model:
        if w is None:
            true_feasible_model.append((w, clause))
    true_logic = convert_to_logic(manager, n, true_feasible_model, context)
    learned_logic = convert_to_logic(manager, n, learned_model, context)
    combined = true_logic & learned_logic

    learned_count, combined_count = (
        l.global_model_count() for l in (learned_logic, combined)
    )
    inf = learned_count - combined_count
    if learned_count == 0:
        return -1
    return inf * 100 / learned_count


def get_recall_precision_sampling(
    n,
    true_model: MaxSatModel,
    learned_model: MaxSatModel,
    context: Context,
    sample_size,
    seed,
):
    recall = get_tp_percentage_sampling(
        n, true_model, learned_model, context, sample_size, seed
    )
    precision = get_tp_percentage_sampling(
        n, learned_model, true_model, context, sample_size, seed
    )

    rng = np.random.RandomState(seed)
    acc = 0
    for _ in range(sample_size):
        instance = rng.rand(n) > 0.5
        for i in rng.choice(list(context), 1):
            instance[abs(i) - 1] = i > 0
        if label_instance(true_model, instance, context) == label_instance(
            learned_model, instance, context
        ):
            acc += 1

    accuracy = acc * 100 / sample_size

    return recall, precision, accuracy


def get_tp_percentage_sampling(
    n,
    true_model: MaxSatModel,
    learned_model: MaxSatModel,
    context: Context,
    sample_size,
    seed,
):
    rng = np.random.RandomState(seed)
    tp = 0
    sample = []
    tmp_data, cst = solve_weighted_max_sat(n, true_model, context, sample_size * 10)
    if len(tmp_data) > sample_size:
        indices = list(rng.choice(range(len(tmp_data)), sample_size, replace=False))
        for i in indices:
            sample.append(tmp_data[i])
    else:
        sample = tmp_data
    for example in sample:
        if label_instance(learned_model, example, context):
            tp += 1
    percent = tp * 100 / len(sample)
    return percent


def simple():
    n = 2
    true_model = [(1, {1}), (1, {2})]
    learned_model = [(5, {1})]
    r, p = get_recall_precision_wmc(n, true_model, learned_model, set())
    print(r, p)


if __name__ == "__main__":
    simple()
