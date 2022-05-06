from hassle_sls.pysat_solver import label_instance_with_cache
from knowledge_compilation import model_to_ADD, best_value, clause_to_BDD, best_value_repeated_restrict, best_value_repeated_efficient_restrict, get_value
from pyddlib.bdd import BDD


def instance_signature_satisfies_context(instance_signature, context, conjunctive_contexts=0):
    """
    Computes whether a given instance signature satisfied a given context.
    An 'instance signature' is to be understood in the following way:
    instance signature [1, 3, -5] represents all instances that are supersets of this collection, i.e. all instances
    that have a positive occurrence of the first variable and third variable, and a negative occurrence of the fifth
    variable. Therefore, in the case of a disjunctive context, an instance signature violates the context only if it
    explicitly contains the negations of the literals occurring in that disjunctive context.
    In case of a conjunctive context, an instance signature violates the context only if it explicitly contains one or
    more negations of literals occurring in that conjunctive context.
    :param instance_signature: The instance signature
    :param context: The context
    :param conjunctive_contexts: A Boolean that denotes whether the context should be
    interpreted as a conjunction. If False, the context is interpreted as a disjunction.
    :return A Boolean that denotes whether the given instance signature satisfies the given context
    """
    if len(context) == 0:
        return True
    if not conjunctive_contexts:
        for l1 in context:
            if l1 in instance_signature or -l1 not in instance_signature:
                # if a literal from the context explicitly occurs in the instance signature, or its negation does not occur
                # in the instance signature, the context is definitely satisfied
                return True
        return False
    else:
        for l1 in context:
            if -l1 in instance_signature:
                return False
        return True


def evaluate(model, examples,
             use_knowledge_compilation_caching=False,
             knowledge_compilation_variant=1,
             use_diagram_for_instance_evaluation=False,
             conjunctive_contexts=0):
    """
    Evaluates the accuracy of a given MAX-SAT model on a set of context-specific labeled examples. This method
    takes two flags: use_knowledge_compilation_caching and use_diagram_for_instance_evaluation. These flags are included
    for the purpose of easily comparing various setups in experiments.
    :param model: The MAX-SAT model (individual) to evaluate
    :param examples: The collection of context-specific labeled examples to use in the evaluation
    :param use_knowledge_compilation_caching: A flag that denotes whether knowledge-compilation based caching is used
    for the purpose of computing the best achievable value of the model+context.
    :param knowledge_compilation_variant This parameter only takes effect if use_knowledge_compilation_based_caching
    has value True. It takes integer values 1, 2, 3 or 4. These denote which variant of knowledge-
    compilation based caching should be used.
    Value 1 means that the model's ADD will be computed once and
    will each context's BDD will be combined (multiplied) with each context's BDD, which is also computed.
    The resulting ADD can be scanned in order to find its maximal terminal node, which is the best attainable value.
    Value 2 means that an alternative approach is used, in which the ADD of the model+context is not computed, but
    merely the ADD of the model is computed. Then, a separate restrict operation is done on the model ADD, one for each
    literal in the context, setting that literal to the value dictated by the context. Then, the best value is searched
    for in the ADD resulting from the restrict. Finally, the maximum of these computed best values is used as the best
    value of the model+context.
    Value 3 means that yet another alternative approach is used, which is supposed to be a more efficient variant of the
    previous one. Here, instead of repeated calculating the result of a restrict operation on the model's ADD, and then
    looking for the best value in the resulting ADD, we look for the best value in the model's ADD, whilst 'imagining'
    that the restrict took place. Concretely, this means that in the scan through the model's ADD for the best value,
    we ignore branches that are not in accordance with the restrict. We again do this for each literal in the context,
    and take the maximum best value as the best value of the model+context.
    Value 4 is the same as value 3, except that the ADD of the model without any context is precomputed, as well as all
    instance signatures that achieve the optimal value in this diagram. Then, when the optimal value in a specific
    context is to be computed, we first check whether any of the precomputed instance signatures satisfies the context.
    If this is the case, we immediately know that the optimal value in said context is the same as the precomputed
    optimal value in the MAX-SAT model without any context. If no instance signature satisfies the context, approach 3
    is used.
    :param use_diagram_for_instance_evaluation: This parameter only takes effect if
    use_knowledge_compilation_based_caching has value True. This parameter takes values True or False.
    If this value is set to False, the knowledge-compilation approach is only used for calculating the best value of
    each model+context combination. If this value is set to True, the model's ADD is also used for calculating the
    actual value of each instance with respect to the model's soft constraints. In this case, the model's ADD is used
    for this, as opposed to the result of multiplying the model's ADD and the context's BDD. This is a valid thing to do
    , as long as the example satisfies the corresponding context, which we assume is the case.
    :param conjunctive_contexts: A Boolean that denotes whether the contexts occurring in the examples should be
    interpreted as conjunctions. If False, the contexts are interpreted as disjunctions.
    :return: The model's accuracy on the labeled examples, as well as a bitvector with as many entries as there are
    examples, where every entry denotes whether the model correctly labeled the corresponding example
    """
    score = 0
    correctly_predicted_bitvector = []
    cache = dict()

    if use_knowledge_compilation_caching:
        (model_as_ADD, has_soft_constraints) = model_to_ADD(model, len(examples[0][1]))
        if knowledge_compilation_variant == 4:
            best_value_model, best_instance_signatures_model = best_value(model_as_ADD, has_soft_constraints)
        variable_nodes = [BDD.variable(i) for i in range(1, len(examples[0][1]) + 1)]
        for context, *_ in examples:
            context_as_tuple = tuple(context)
            if context_as_tuple not in cache:
                if knowledge_compilation_variant == 1:
                    if not conjunctive_contexts:
                        cache[context_as_tuple] = best_value(model_as_ADD * clause_to_BDD(context, variable_nodes),
                                                             has_soft_constraints)[0]
                    else:
                        model_with_contexts = model_as_ADD
                        for i in context:
                            model_with_contexts = model_with_contexts * clause_to_BDD({i}, variable_nodes)
                        cache[context_as_tuple] = best_value(model_with_contexts, has_soft_constraints)[0]

                elif knowledge_compilation_variant == 2:
                    # Alternative version of calculating best value of model + context using repeated restrict
                    cache[context_as_tuple] = best_value_repeated_restrict(model_as_ADD, context, has_soft_constraints,
                                                                           conjunctive_contexts=conjunctive_contexts)

                elif knowledge_compilation_variant == 3:
                    # More efficient version of the alternative version mentioned above
                    cache[context_as_tuple] = best_value_repeated_efficient_restrict(model_as_ADD, context,
                                                                                     has_soft_constraints,
                                                                                     conjunctive_contexts=conjunctive_contexts)
                elif knowledge_compilation_variant == 4:
                    added_to_cache = False
                    for a_best_instance_signature in best_instance_signatures_model:
                        if instance_signature_satisfies_context(a_best_instance_signature, context, conjunctive_contexts=conjunctive_contexts):
                            cache[context_as_tuple] = best_value_model
                            added_to_cache = True
                            break
                    if not added_to_cache:
                        cache[context_as_tuple] = best_value_repeated_efficient_restrict(model_as_ADD, context,
                                                                                         has_soft_constraints,
                                                                                         conjunctive_contexts=conjunctive_contexts)
                else:
                    raise ValueError("knowledge_compilation_variant only takes values 1, 2, 3 or 4")

    for i in range(len(examples)):
        context, instance, label = examples[i]
        cached_best_value = None
        context_as_tuple = tuple(context)
        if context_as_tuple in cache:
            cached_best_value = cache[context_as_tuple]
        if use_knowledge_compilation_caching and use_diagram_for_instance_evaluation:
            value_of_instance = get_value(model_as_ADD, instance, has_soft_constraints)
            result = label_instance_with_cache(model, instance, context,
                                               cached_best_value=cached_best_value, value_of_instance=value_of_instance,
                                               conjunctive_contexts=conjunctive_contexts)
        else:
            result = label_instance_with_cache(model, instance, context,
                                               cached_best_value=cached_best_value, conjunctive_contexts=conjunctive_contexts)
        learned_label = result[0]
        # Don't update cache when using cached value, because then this update would be redundant.
        if cached_best_value is None and result[1] is not None:
            cache[context_as_tuple] = result[1]
        if ((learned_label == 0 or learned_label == -1) and (label == 0 or label == -1)) or (learned_label == 1 and label == 1):
            score += 1
            correctly_predicted_bitvector.append(1)
        else:
            correctly_predicted_bitvector.append(0)

    return score / len(examples), correctly_predicted_bitvector


def evaluate_use_infeasibility(model, examples,
             use_knowledge_compilation_caching=False,
             knowledge_compilation_variant=1,
             use_diagram_for_instance_evaluation=False,
             conjunctive_contexts=0, inf=None):
    """
    Evaluates the accuracy of a given MAX-SAT model on a set of context-specific labeled examples.
    The difference with the evaluate function is that here, the distinction between infeasibility versus suboptimality
    matters. This complicates knowledge compilation, as in a straightforward translation from a MAX-SAT model to an ADD,
    no distinction can be made between an instance that is infeasibile and an instance that is feasible
    but satisfies 0 weight in soft constraints.
    The inf parameter is only used by HASSLE-SLS, as it uses a different way of representing labels.
    :param model: The MAX-SAT model (individual) to evaluate
    :param examples: The collection of context-specific labeled examples to use in the evaluation
    :param use_knowledge_compilation_caching: A flag that denotes whether knowledge-compilation based caching is used
    for the purpose of computing the best achievable value of the model+context.
    :param knowledge_compilation_variant This parameter only takes effect if use_knowledge_compilation_based_caching
    has value True. It takes integer values 1, 2, 3 or 4. These denote which variant of knowledge-
    compilation based caching should be used.
    Value 1 means that the model's ADD will be computed once and
    will each context's BDD will be combined (multiplied) with each context's BDD, which is also computed.
    The resulting ADD can be scanned in order to find its maximal terminal node, which is the best attainable value.
    Value 2 means that an alternative approach is used, in which the ADD of the model+context is not computed, but
    merely the ADD of the model is computed. Then, a separate restrict operation is done on the model ADD, one for each
    literal in the context, setting that literal to the value dictated by the context. Then, the best value is searched
    for in the ADD resulting from the restrict. Finally, the maximum of these computed best values is used as the best
    value of the model+context.
    Value 3 means that yet another alternative approach is used, which is supposed to be a more efficient variant of the
    previous one. Here, instead of repeated calculating the result of a restrict operation on the model's ADD, and then
    looking for the best value in the resulting ADD, we look for the best value in the model's ADD, whilst 'imagining'
    that the restrict took place. Concretely, this means that in the scan through the model's ADD for the best value,
    we ignore branches that are not in accordance with the restrict. We again do this for each literal in the context,
    and take the maximum best value as the best value of the model+context.
    Value 4 is the same as value 3, except that the ADD of the model without any context is precomputed, as well as all
    instance signatures that achieve the optimal value in this diagram. Then, when the optimal value in a specific
    context is to be computed, we first check whether any of the precomputed instance signatures satisfies the context.
    If this is the case, we immediately know that the optimal value in said context is the same as the precomputed
    optimal value in the MAX-SAT model without any context. If no instance signature satisfies the context, approach 3
    is used.
    :param use_diagram_for_instance_evaluation: This parameter only takes effect if
    use_knowledge_compilation_based_caching has value True. This parameter takes values True or False.
    If this value is set to False, the knowledge-compilation approach is only used for calculating the best value of
    each model+context combination. If this value is set to True, the model's ADD is also used for calculating the
    actual value of each instance with respect to the model's soft constraints. In this case, the model's ADD is used
    for this, as opposed to the result of multiplying the model's ADD and the context's BDD. This is a valid thing to do
    , as long as the example satisfies the corresponding context, which we assume is the case.
    :param conjunctive_contexts: A Boolean that denotes whether the contexts occurring in the examples should be
    interpreted as conjunctions. If False, the contexts are interpreted as disjunctions.
    :param inf: Only relevant when this function is used in HASSLE-SLS. A list that contains as many entries as there
    are examples. Every entry is a Boolean that denotes whether the corresponding example is infeasible or not.
    :return: The model's accuracy on the labeled examples, as well as a bitvector with as many entries as there are
    examples, where every entry denotes whether the model correctly labeled the corresponding example
    """
    score = 0
    correctly_predicted_bitvector = []
    cache = dict()

    if use_knowledge_compilation_caching:
        hard_model = [constraint for constraint in model if constraint[0] is None]
        hard_model_as_ADD, _ = model_to_ADD(hard_model, len(examples[0][1]))
        full_model_as_ADD, has_soft_constraints = model_to_ADD(model, len(examples[0][1]))

        if knowledge_compilation_variant == 4:
            best_value_model, best_instance_signatures_model = best_value(full_model_as_ADD, has_soft_constraints)
        variable_nodes = [BDD.variable(i) for i in range(1, len(examples[0][1]) + 1)]
        for context, *_ in examples:
            context_as_tuple = tuple(context)
            if context_as_tuple not in cache:
                if knowledge_compilation_variant == 1:
                    if not conjunctive_contexts:
                        the_best_value = best_value(full_model_as_ADD * clause_to_BDD(context, variable_nodes),
                                                             has_soft_constraints)[0]
                        if the_best_value != 0:
                            cache[context_as_tuple] = the_best_value
                        else:
                            cache[context_as_tuple] = best_value(hard_model_as_ADD * clause_to_BDD(context, variable_nodes),
                                                             False)[0]
                    else:
                        model_with_contexts = full_model_as_ADD
                        for i in context:
                            model_with_contexts = model_with_contexts * clause_to_BDD({i}, variable_nodes)
                        the_best_value = best_value(model_with_contexts, has_soft_constraints)[0]
                        if the_best_value != 0:
                            cache[context_as_tuple] = the_best_value
                        else:
                            model_with_contexts = hard_model_as_ADD
                            for i in context:
                                model_with_contexts = model_with_contexts * clause_to_BDD({i}, variable_nodes)
                            cache[context_as_tuple] = best_value(model_with_contexts, False)[0]

                elif knowledge_compilation_variant == 2:
                    # Alternative version of calculating best value of model + context using repeated restrict
                    the_best_value = best_value_repeated_restrict(full_model_as_ADD, context, has_soft_constraints,
                                                                           conjunctive_contexts=conjunctive_contexts)
                    if the_best_value != 0:
                        cache[context_as_tuple] = the_best_value
                    else:
                        cache[context_as_tuple] = best_value_repeated_restrict(hard_model_as_ADD, context, False,
                                                                           conjunctive_contexts=conjunctive_contexts)

                elif knowledge_compilation_variant == 3:
                    # More efficient version of the alternative version mentioned above
                    the_best_value = best_value_repeated_efficient_restrict(full_model_as_ADD, context,
                                                                                     has_soft_constraints,
                                                                                     conjunctive_contexts=conjunctive_contexts)
                    if the_best_value != 0:
                        cache[context_as_tuple] = the_best_value
                    else:
                        cache[context_as_tuple] = best_value_repeated_efficient_restrict(hard_model_as_ADD, context,
                                                                                     False,
                                                                                     conjunctive_contexts=conjunctive_contexts)
                elif knowledge_compilation_variant == 4:
                    added_to_cache = False
                    for a_best_instance_signature in best_instance_signatures_model:
                        if instance_signature_satisfies_context(a_best_instance_signature, context, conjunctive_contexts=conjunctive_contexts):
                            cache[context_as_tuple] = best_value_model
                            added_to_cache = True
                            break
                    if not added_to_cache:
                        the_best_value = best_value_repeated_efficient_restrict(full_model_as_ADD, context,
                                                                                         has_soft_constraints,
                                                                                         conjunctive_contexts=conjunctive_contexts)
                        if the_best_value != 0:
                            cache[context_as_tuple] = the_best_value
                        else:
                            cache[context_as_tuple] = best_value_repeated_efficient_restrict(hard_model_as_ADD, context,
                                                                                         False,
                                                                                         conjunctive_contexts=conjunctive_contexts)
                else:
                    raise ValueError("knowledge_compilation_variant only takes values 1, 2, 3 or 4")

    for i in range(len(examples)):
        context, instance, label = examples[i]
        cached_best_value = None
        context_as_tuple = tuple(context)
        if context_as_tuple in cache:
            cached_best_value = cache[context_as_tuple]
        if use_knowledge_compilation_caching and use_diagram_for_instance_evaluation:
            value_of_instance = get_value(full_model_as_ADD, instance, has_soft_constraints)
            if value_of_instance == 0:
                value_of_instance = get_value(hard_model_as_ADD, instance, False)
            result = label_instance_with_cache(model, instance, context,
                                               cached_best_value=cached_best_value, value_of_instance=value_of_instance,
                                               conjunctive_contexts=conjunctive_contexts)
        else:
            result = label_instance_with_cache(model, instance, context,
                                               cached_best_value=cached_best_value, conjunctive_contexts=conjunctive_contexts)
        learned_label = result[0]
        # Don't update cache when using cached value, because then this update would be redundant.
        if cached_best_value is None and result[1] is not None:
            cache[context_as_tuple] = result[1]
        if inf is None:
            if learned_label == label:
                score += 1
                correctly_predicted_bitvector.append(1)
            else:
                correctly_predicted_bitvector.append(0)
        else:
            if (learned_label == -1 and label == False and inf[i] == True) or \
                    (learned_label == 0 and label == False and inf[i] == False) or \
                    (learned_label == 1 and label == True):
                score += 1
                correctly_predicted_bitvector.append(1)
            else:
                correctly_predicted_bitvector.append(0)
    return score / len(examples), correctly_predicted_bitvector


def evaluate_knowledge_compilation_based_dispatch(model, examples, knowledge_compilation_variant=4, use_diagram_for_instance_evaluation=True, conjunctive_contexts=0, inf=None):
    """
    ONLY FOR HASSLE-SLS!
    Dispatches the evaluate call along to the right evaluation function, depending on whether the distinction between
    infeasible and suboptimal negative examples should be made.
    :param model: The MAX-SAT model (individual) to evaluate
    :param examples: The collection of context-specific labeled examples to use in the evaluation
    :param use_knowledge_compilation_caching: A flag that denotes whether knowledge-compilation based caching is used
    for the purpose of computing the best achievable value of the model+context.
    :param knowledge_compilation_variant This parameter only takes effect if use_knowledge_compilation_based_caching
    has value True. It takes integer values 1, 2, 3 or 4. These denote which variant of knowledge-
    compilation based caching should be used.
    Value 1 means that the model's ADD will be computed once and
    will each context's BDD will be combined (multiplied) with each context's BDD, which is also computed.
    The resulting ADD can be scanned in order to find its maximal terminal node, which is the best attainable value.
    Value 2 means that an alternative approach is used, in which the ADD of the model+context is not computed, but
    merely the ADD of the model is computed. Then, a separate restrict operation is done on the model ADD, one for each
    literal in the context, setting that literal to the value dictated by the context. Then, the best value is searched
    for in the ADD resulting from the restrict. Finally, the maximum of these computed best values is used as the best
    value of the model+context.
    Value 3 means that yet another alternative approach is used, which is supposed to be a more efficient variant of the
    previous one. Here, instead of repeated calculating the result of a restrict operation on the model's ADD, and then
    looking for the best value in the resulting ADD, we look for the best value in the model's ADD, whilst 'imagining'
    that the restrict took place. Concretely, this means that in the scan through the model's ADD for the best value,
    we ignore branches that are not in accordance with the restrict. We again do this for each literal in the context,
    and take the maximum best value as the best value of the model+context.
    Value 4 is the same as value 3, except that the ADD of the model without any context is precomputed, as well as all
    instance signatures that achieve the optimal value in this diagram. Then, when the optimal value in a specific
    context is to be computed, we first check whether any of the precomputed instance signatures satisfies the context.
    If this is the case, we immediately know that the optimal value in said context is the same as the precomputed
    optimal value in the MAX-SAT model without any context. If no instance signature satisfies the context, approach 3
    is used.
    :param use_diagram_for_instance_evaluation: This parameter only takes effect if
    use_knowledge_compilation_based_caching has value True. This parameter takes values True or False.
    If this value is set to False, the knowledge-compilation approach is only used for calculating the best value of
    each model+context combination. If this value is set to True, the model's ADD is also used for calculating the
    actual value of each instance with respect to the model's soft constraints. In this case, the model's ADD is used
    for this, as opposed to the result of multiplying the model's ADD and the context's BDD. This is a valid thing to do
    , as long as the example satisfies the corresponding context, which we assume is the case.
    :param conjunctive_contexts: A Boolean that denotes whether the contexts occurring in the examples should be
    interpreted as conjunctions. If False, the contexts are interpreted as disjunctions.
    :param inf: Only relevant when this function is used in HASSLE-SLS. A list that contains as many entries as there
    are examples. Every entry is a Boolean that denotes whether the corresponding example is infeasible or not.
    :return: The model's accuracy on the labeled examples, as well as a bitvector with as many entries as there are
    examples, where every entry denotes whether the model correctly labeled the corresponding example

    """
    if inf is None:
        return evaluate(model, examples, use_knowledge_compilation_caching=True, knowledge_compilation_variant=knowledge_compilation_variant,
                        use_diagram_for_instance_evaluation=use_diagram_for_instance_evaluation, conjunctive_contexts=conjunctive_contexts)
    else:
        return evaluate_use_infeasibility(model, examples, use_knowledge_compilation_caching=use_diagram_for_instance_evaluation,
                                          knowledge_compilation_variant=knowledge_compilation_variant,
                                          use_diagram_for_instance_evaluation=use_diagram_for_instance_evaluation,
                                          conjunctive_contexts=conjunctive_contexts, inf=inf)