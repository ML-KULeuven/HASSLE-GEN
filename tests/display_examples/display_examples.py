import pickle
from os import listdir, path

if __name__ == "__main__":
    pathname = path.dirname(path.realpath(__file__))
    pathname_target_model = pathname + "/pickles/contexts_and_data/"
    for f in listdir(pathname_target_model):
        with open(pathname_target_model+f, "rb") as fp:
            pickle_var = pickle.load(fp)
        contexts, instances, labels = (
            [{int(str(lit)) for lit in context} for context in pickle_var["contexts"]],
            pickle_var["data"],
            pickle_var["labels"],
        )

        # Print the instances
        print(f)
        print(len(instances))
        for i in range(len(instances)):
            instance = instances[i]
            print(i, list(zip(range(1, len(instances[i]) + 1), instances[i])), contexts[i], labels[i])
        print("\n\n\n\n\n")

        # Make sure no intext+context combination occurs twice
        print()
        print(f)
        a_list = [(tuple(instance), frozenset(context)) for (instance, context) in zip(instances, contexts)]
        a_set = set()
        for (instance,context) in zip(instances, contexts):
            a_set.add((tuple(instance), frozenset(context)))
        print(len(a_list))
        print(len(a_set))

        # Print label proportions
        print()
        print(f)
        print(f"Total number of examples: {len(instances)}")
        print(f"Proportion solutions: {len([instances[i] for i in range(len(instances)) if labels[i] == 1])/len(instances)}")
        print(f"Proportion suboptimal: {len([instances[i] for i in range(len(instances)) if labels[i] == 0])/len(instances)}")
        print(f"Proportion infeasibile: {len([instances[i] for i in range(len(instances)) if labels[i] == -1])/len(instances)}")
        #
        # # Print how often instances re-occur in different contexts per label
        solution_instance_set = set()
        infeasible_instance_set = set()
        suboptimal_instance_set = set()
        for i in range(len(instances)):
            if labels[i] == 1:
                solution_instance_set.add(tuple(instances[i]))
            if labels[i] == -1:
                infeasible_instance_set.add(tuple(instances[i]))
            if labels[i] == 0:
                suboptimal_instance_set.add(tuple(instances[i]))
        num_solutions = len([instances[i] for i in range(len(instances)) if labels[i] == 1])
        num_infeasible = len([instances[i] for i in range(len(instances)) if labels[i] == -1])
        num_suboptimal = len([instances[i] for i in range(len(instances)) if labels[i] == 0])
        print()
        print(f"The {num_solutions} solution examples contain {len(solution_instance_set)} different instances. That is {(len(solution_instance_set) * 100)/num_solutions}%")
        print(f"The {num_infeasible} infeasible examples contain {len(infeasible_instance_set)} different instances. That is {(len(infeasible_instance_set) * 100) / num_infeasible}%")
        print(f"The {num_suboptimal} suboptimal examples contain {len(suboptimal_instance_set)} different instances. That is {(len(suboptimal_instance_set) * 100) / num_suboptimal}%")
