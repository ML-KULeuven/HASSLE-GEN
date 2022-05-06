from pyddlib.bdd import BDD
from pyddlib.add import ADD
import operator
from functools import reduce


def clause_to_BDD(clause, variable_nodes):
    """
    Converts a clause into a binary decision diagram
    :param clause: The clause
    :param variable_nodes: A list of BDD variable nodes, one for each variable
    :return: The binary decision diagram that represents the clause
    """
    return reduce(operator.or_, [variable_nodes[abs(i) - 1] if i > 0 else ~variable_nodes[abs(i) - 1] for i in clause])


def model_to_ADD(model, number_of_variables):
    """
    Converts a MAX-SAT model into an algebraic decision diagram
    :param model: The MAX-SAT model
    :param number_of_variables: The number of variables
    :return: A tuple containing (i) the algebraic decision diagram and (ii) a Boolean that denotes whether MAX-SAT model
    contains soft constraints
    """
    # We make one node for each variable. From these nodes, we will construct the BDDs corresponding to the clauses
    # in the model.

    if len(model) == 0:
        return ADD.constant(0), False

    variable_nodes = [BDD.variable(i) for i in range(1, number_of_variables+1)]

    hard_constraint_BDDs = []
    soft_constraint_ADDs = []

    for constraint in model:
        if not constraint[1]:
            if constraint[0] is None:
                # Simply return a model with a single 0 node if the current constraint is a hard constraint and is empty
                return ADD.constant(0), False
            else:
                # Continue to next iteration if the current constraint is a soft constraint and is empty
                continue

        # Construct a BDD that represents the current constraint (disjunctive clause)
        bdd = clause_to_BDD(constraint[1], variable_nodes)

        if constraint[0] is None:
            # Hard constraint
            hard_constraint_BDDs.append(bdd)
        else:
            # Soft constraint
            # Multiply the BDD with an ADD that contains a single node with as value the soft constraint's weight
            soft_constraint_ADDs.append(ADD.constant(constraint[0]) * bdd)

    # The full model, represented in an ADD, is the conjunction of the hard constraints' BDDs, multiplied with
    # the sum of the soft constraints' ADDs.
    if len(hard_constraint_BDDs) > 0:
        conjunction_of_hard_constraint_BDDs = reduce(operator.and_, hard_constraint_BDDs)
    else:
        conjunction_of_hard_constraint_BDDs = BDD.one()

    if len(soft_constraint_ADDs) > 0:
        sum_of_soft_constraint_ADDs = reduce(operator.add, soft_constraint_ADDs)
        # First thing that gets returned is the diagram, the second denotes whether the diagram 'contains' soft
        # constraints
        return ADD.constant(1) * conjunction_of_hard_constraint_BDDs * sum_of_soft_constraint_ADDs, True
    else:
        # First thing that gets returned is the diagram, the second denotes whether the diagram 'contains' soft
        # constraints
        return ADD.constant(1) * conjunction_of_hard_constraint_BDDs, False


def best_value(add: ADD, has_soft_constraints):
    """
    Compute the best value achievable in a given algebraic decision diagram. This diagram could represent a MAX-SAT
    model, or a MAX-SAT model + context combination
    :param add: The algebraic decision diagram
    :param has_soft_constraints: A Boolean denoting whether the MAX-SAT model that the algebraic decision diagram
    represents contains soft constraints
    :return: A tuple containing (i) the best value achievable and (ii) an instance or partial instance that achieves
    that value. Note: when the represented model contains no soft constraints, the best value returned is 0 when the
    hard constraints are satisfied, and None when the hard constraints are not satisfied.
    """
    # Could alter the ADD code so that the maximal value is tracked during ADD construction, and does not have to be
    # searched for in the end. Alternatively, a list of the ADDs terminal nodes can be tracked. Either way,
    # to get a time benefit, this could be be optimised. Note: there seems to be an efficient way of calculating the
    # best_value for a model in a given context which does not involve calculating the ADD of the model+context,
    # and does not involve this method, so, optimising this method is perhaps not worth the time investment.
    if has_soft_constraints:
        return best_value_helper(add)
    else:
        the_best_value, the_best_value_instance = best_value_helper(add)
        if the_best_value == 0:
            # If there are no soft constraints, and the hard constraints are violated, the best value is None
            return None, the_best_value_instance
        elif the_best_value == 1:
            # If there are no soft constraints, and the hard constraints are satisfied, the best value is 0
            return 0, the_best_value_instance
        else:
            raise Exception("When there are no soft constraints in a diagram, the best value w.r.t. the diagram should"
                            "be either 0 or 1.")


def best_value_helper(add: ADD):
    """
    Recursive helper function to best_value
    :param add: The algebraic decision diagram
    :return: A tuple containing (i) the best value achievable and (ii) an instance or partial instance that achieves
    that value.
    """
    if add.is_terminal():
        return add.value, [[]]
    else:
        neg_branch_value, neg_branch_instance_signatures = best_value_helper(add._low)
        pos_branch_value, pos_branch_instance_signatures = best_value_helper(add._high)
        if neg_branch_value > pos_branch_value:
            return neg_branch_value, [[-add.index] + neg_branch_instance_signature for neg_branch_instance_signature in neg_branch_instance_signatures]
        elif pos_branch_value > neg_branch_value:
            return pos_branch_value, [[add.index] + pos_branch_instance_signature for pos_branch_instance_signature in pos_branch_instance_signatures]
        else:
            return neg_branch_value, [[-add.index] + neg_branch_instance_signature for neg_branch_instance_signature in neg_branch_instance_signatures]\
                   + [[add.index] + pos_branch_instance_signature for pos_branch_instance_signature in pos_branch_instance_signatures]


def best_value_repeated_restrict(model: ADD, context: BDD, has_soft_constraints, conjunctive_contexts=0):
    """
    Alternative way of finding best value of model in a certain context.
    In this approach, we do not use the ADD resulting from the multiplication of the model ADD and context BDD,
    and calculate ts best value. Instead, for each literal in the given context, we do a restrict-operation on the
    model ADD which sets that literal to the value dictated by the context, and we calculate the best value of the
    resulting diagrams. We then return the maximum of those calculated best values.
    :param model: An algebraic decision diagram representing a MAX-SAT model
    :param: context: A binary decision diagram representing a context
    :param: has_soft_constraints: A Boolean denoting whether the MAX-SAT model that the algebraic decision diagram
    represents contains soft constraints
    :param conjunctive_contexts: A Boolean that denotes whether the context should be
    interpreted as a conjunction. If False, the context is interpreted as a disjunction.
    :return: The best value achievable. Note: the best value returned is 0 when the
    hard constraints are satisfied, and None when the hard constraints are not satisfied.
    """
    if has_soft_constraints:
        if not conjunctive_contexts:
            return max([best_value(model.restrict({abs(i): i > 0}), has_soft_constraints)[0] for i in context])
        else:
            return best_value(model.restrict({abs(i): i > 0 for i in context}), has_soft_constraints)[0]
    else:
        if not conjunctive_contexts:
            best_value_list = [best_value(model.restrict({abs(i): i > 0}), has_soft_constraints)[0] for i in context]
        else:
            best_value_list = [best_value(model.restrict({abs(i): i > 0 for i in context}), has_soft_constraints)[0]]

        if 0 in best_value_list:
            # If there are no soft constraints, and the hard constraints are violated, the best value is None
            return 0
        elif all(value is None for value in best_value_list):
            return None
        else:
            raise Exception("When there are no soft constraints in a diagram, the repeated restrict should result in "
                            "0s and/or Nones")


def best_value_repeated_efficient_restrict(add: ADD, context, has_soft_constraints, conjunctive_contexts=0):
    """
    Yet another alternative way of finding the best value of a model in a certain context.
    We 'imagine' a restrict on the context in the best-value calculation, and simply don't calculate the best-value of
    branches that violate the context in the recursive process. For a disjunctive context, we do this separately for
    every literal set by the context, and take the maximum of the resulting best values.
    This gives us the same result as a regular restrict, except that no new diagram had to be computed.
    :param add: An algebraic decision diagram representing a MAX-SAT model
    :param: context: A binary decision diagram representing a context
    :param: has_soft_constraints: A Boolean denoting whether the MAX-SAT model that the algebraic decision diagram
    represents contains soft constraints
    :param conjunctive_contexts: A Boolean that denotes whether the context should be
    interpreted as a conjunction. If False, the context is interpreted as a disjunction.
    :return: The best value achievable. Note: the best value returned is 0 when the
    hard constraints are satisfied, and None when the hard constraints are not satisfied.
    """
    if has_soft_constraints:
        if not conjunctive_contexts:
            return max([best_value_restricted_disjunctive_contexts(add, abs(i), (i > 0)) for i in context])
        else:
            return best_value_restricted_conjunctive_contexts(add, [abs(i) for i in context], [(i > 0) for i in context])
    else:
        if not conjunctive_contexts:
            best_value = max([best_value_restricted_disjunctive_contexts(add, abs(i), (i > 0)) for i in context])
        else:
            best_value = best_value_restricted_conjunctive_contexts(add, [abs(i) for i in context], [(i > 0) for i in context])

        if best_value == 0:
            # If there are no soft constraints, and the hard constraints are violated, the best value is None
            return None
        elif best_value == 1:
            # If there are no soft constraints, and the hard constraints are satisfied, the best value is 0
            return 0
        else:
            raise Exception("When there are no soft constraints in a diagram, the best value w.r.t. the diagram should"
                            "be either 0 or 1.")


def best_value_restricted_disjunctive_contexts(add: ADD, restrict_literal, restrict_value):
    """
    Recursive helper function to best_value_repeated_efficient_restrict for disjunctive contexts
    We 'imagine' a restrict on one literal in the best-value calculation, and simply don't calculate the best value of
    branches that violate the restrict in the recursive process.
    :param add: An algebraic decision diagram (shrinks every recursive call)
    :param restrict_literal: The number of the variable which is being restricted (integer)
    :param restrict_value: The value to restrict the variable to (True or False)
    :return The best value achievable
    """
    if add.is_terminal():
        return add.value
    else:
        if add.index == restrict_literal:
            if restrict_value:
                return best_value_restricted_disjunctive_contexts(add._high, restrict_literal, restrict_value)
            else:
                return best_value_restricted_disjunctive_contexts(add._low, restrict_literal, restrict_value)
        return max(best_value_restricted_disjunctive_contexts(add._low, restrict_literal, restrict_value),
                   best_value_restricted_disjunctive_contexts(add._high, restrict_literal, restrict_value))


def best_value_restricted_conjunctive_contexts(add: ADD, restrict_literals, restrict_values):
    """
    Recursive helper function to best_value_repeated_efficient_restrict for conjunctive contexts
    We 'imagine' a restrict on the entire context in the best-value calculation, and simply don't calculate the
    best value of branches that violate the restrict in the recursive process.
    :param add: An algebraic decision diagram (shrinks every recursive call)
    :param restrict_literals: A list of numbers of variables which are being restricted (list of integers)
    :param restrict_values: A list of respective values to restrict the variables to (list of Booleans)
    :return The best value achievable
    """
    if add.is_terminal():
        return add.value
    else:
        if add.index in restrict_literals:
            index = restrict_literals.index(add.index)
            restrict_value = restrict_values[index]
            if restrict_value:
                return best_value_restricted_conjunctive_contexts(add._high, restrict_literals, restrict_values)
            else:
                return best_value_restricted_conjunctive_contexts(add._low, restrict_literals, restrict_values)
        return max(best_value_restricted_conjunctive_contexts(add._low, restrict_literals, restrict_values),
                   best_value_restricted_conjunctive_contexts(add._high, restrict_literals, restrict_values))


def get_value(add: ADD, instance, has_soft_constraints):
    """
    Takes a MAX-SAT model represented as an ADD and an instance, and calculates the value of that instance with respect
    to the model's soft constraints. None is returned if the instance is not a complete assignment but only
    a partial assignment. Zero is returned if the instance is infeasible with respect to the model's hard constraints.
    :param add: An algebraic decision diagram representing a MAX-SAT model
    :param instance: An instance
    :param has_soft_constraints: A Boolean denoting whether the MAX-SAT model that the algebraic decision diagram
    represents contains soft constraints
    :return The value that the instance achieves with respect to the MAX-SAT model's soft constraints
    """
    valuation = {}
    for i in range(1, len(instance) + 1):
        valuation[i] = instance[i-1]
    value = add.restrict(valuation).value
    if has_soft_constraints:
        return value
    else:
        if value == 0:
            # If there are no soft constraints, and the hard constraints are violated, the value is None
            return None
        elif value == 1:
            # If there are no soft constraints, and the hard constraints are satisfied, the value is 0
            return 0
        else:
            raise Exception("When there are no soft constraints in a diagram, the value w.r.t. the diagram should"
                            "be either 0 or 1.")
