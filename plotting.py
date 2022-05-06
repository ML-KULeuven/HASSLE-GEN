import pickle
import itertools
import numpy as np
from math import sqrt
import matplotlib.markers as mark
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns
from hassle_sls.experiments.synthetic import evaluate_statistics

"""
This module contains classes and methods useful for plotting (aggregates of) data contained within observers.
"""


class ScatterData:
    colors = []
    markers = ["o", "v", "x"]

    def __init__(self, title, plot_options):
        self.title = title
        self.data = []
        self.limits = None, None
        self.plot_options = plot_options

    def add_data(self, name, x_data, y_data, error=None):
        self.data.append((name, x_data, y_data, error))
        return self

    @property
    def size(self):
        return len(self.data)

    def x_lim(self, limits):
        self.limits = limits, self.limits[1]

    def y_lim(self, limits):
        self.limits = self.limits[0], limits

    def gen_colors(self):
        if len(self.data) <= len(self.colors):
            return self.colors[:len(self.data)]
        iterator = iter(cm.get_cmap("rainbow")(np.linspace(0, 1, len(self.data))))
        return [next(iterator) for _ in range(len(self.data))]

    def gen_markers(self):
        if len(self.data) <= len(self.markers):
            return self.markers[:len(self.data)]
        iterator = itertools.cycle(mark.MarkerStyle.filled_markers)
        return [next(iterator) for _ in range(len(self.data))]

    def render(self, ax, lines=True, log_x=True, log_y=True, label_x=None, label_y=None, legend_pos=None,
               x_ticks=None, y_ticks=None, show_x_tick_labels=True, show_y_tick_labels=True):

        plots = []
        colors = self.gen_colors()
        markers = self.gen_markers()

        show_legend = True
        if legend_pos is None:
            legend_pos = "lower right"

        plot_diagonal = False
        plot_extra = None
        plot_format = "scatter"
        show_error = True
        steps_x = None
        steps_y = None
        y_tick_every_unit = False

        cache = None
        for plot_option in self.plot_options or ():
            if cache is None:
                if plot_option == "diagonal":
                    plot_diagonal = True
                else:
                    cache = plot_option
            else:
                if cache == "format":
                    plot_format = plot_option
                elif cache == "error":
                    show_error = int(plot_option)
                elif cache == "legend":
                    if plot_option != "none":
                        legend_pos = plot_option
                elif cache == "show_legend":
                    if plot_option == "False" or plot_option == "false":
                        show_legend = False
                elif cache == "lx":
                    label_x = plot_option
                elif cache == "ly":
                    label_y = plot_option
                elif cache == "steps_x":
                    steps_x = int(plot_option)
                elif cache == "steps_y":
                    steps_y = int(plot_option)
                elif cache == "y_tick_every_unit":
                    if plot_option == "True" or plot_option == "true":
                        y_tick_every_unit = True
                elif cache == "plot_extra":
                    plot_extra = plot_option
                elif cache == "x_lim":
                    parts = plot_option.split(":")
                    limits = (float(parts[0]), float(parts[1]))
                    self.x_lim(limits)
                elif cache == "y_lim":
                    parts = plot_option.split(":")
                    limits = (float(parts[0]), float(parts[1]))
                    self.y_lim(limits)
                cache = None

        min_x, max_x, min_y, max_y = np.infty, -np.infty, np.infty, -np.infty
        for i in range(self.size):
            name, x_data, y_data, error = self.data[i]
            try:
                min_x = min(min_x, np.min(x_data))
                min_y = min(min_y, np.min(y_data))
                max_x = max(max_x, np.max(x_data))
                max_y = max(max_y, np.max(y_data))
            except TypeError:
                pass

            if plot_format == "scatter":
                plots.append(ax.scatter(x_data, y_data, color=colors[i], marker=markers[i], s=1))
                if lines:
                    ax.plot(x_data, y_data, color=colors[i], linewidth=1)
                if show_error == 1 and error is not None:
                    ax.fill_between(x_data, y_data - error, y_data + error, color=colors[i], alpha=0.35, linewidth=0)
                elif show_error == 2 and error is not None:
                    ax.errorbar(x_data, y_data, error, linestyle='None', capsize=2, capthick=0.75, elinewidth=0.75, color=colors[i])
            elif plot_format == "bar":
                plots.append(ax.bar(x_data, y_data, color=colors[i]))
            else:
                raise ValueError("Unknown plot format")

            if i == self.size - 1:
                if plot_extra and plot_extra == "1/x":
                    ax.plot(x_data, 1 / x_data, linestyle="--")

        if plot_diagonal:
            ax.plot(np.array([min_x, max_x]), np.array([min_y, max_y]), linestyle="--")

        ax.grid(True)
        legend_names = list(t[0] for t in self.data)
        # legend_names = ["No mixing - DT", "No mixing - RF", "Mixing - DT", "Mixing - RF"]
        # legend_names = ["No formulas", "Formulas"]
        # legend_names = []
        if show_legend and (15 >= len(self.data) == len(legend_names)) and legend_pos:
            ax.legend(plots, legend_names, loc=legend_pos, prop={'size': 9}, bbox_to_anchor=(1.05, 1))

        if log_x:
            ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')

        x_lim, y_lim = self.limits
        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)

        if label_y is not None:
            ax.set_ylabel(label_y, fontsize=4)
        if label_x is not None:
            ax.set_xlabel(label_x, fontsize=4)

        if steps_x is not None:
            x_ticks = np.linspace(min_x, max_x, steps_x)
        if steps_y is not None:
            y_ticks = np.linspace(min_y, max_y, steps_y)
        if y_tick_every_unit:
            y_ticks = np.linspace(0, int(np.ceil(max_y)), int(np.ceil(max_y)) + 1)
            y_ticks = [int(y_tick) for y_tick in y_ticks]
        # x_ticks = [1, 2, 3]
        if x_ticks is not None:
            ax.xaxis.set_ticks(x_ticks)
            ax.set_xticklabels(x_ticks if show_x_tick_labels else "", fontsize=3)
        if y_ticks is not None:
            ax.yaxis.set_ticks(y_ticks)
            ax.set_yticklabels(y_ticks if show_y_tick_labels else "", fontsize=4)

    def plot(self, filename=None, size=None, **kwargs):
        fig = plt.figure()
        if size is not None:
            fig.set_size_inches(*size)
        self.render(fig.gca(), **kwargs)
        if filename is None:
            plt.show(block=True)
        else:
            plt.savefig(filename, format="png", bbox_inches="tight", pad_inches=0.01, dpi=600)
            plt.close()


def plot_example_coverage_heatmap(observer, pathname):
    """
    Takes a single evolutionary algorithm observer and plots a heatmap showing the number of individuals
    covering each example throughout the generations
    """
    data = np.transpose(observer.get_array("example_coverage_vector"))
    pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax = sns.heatmap(data)
    plt.xlabel("Generations")
    plt.ylabel("Examples")
    plt.savefig(pathname, format="png", bbox_inches="tight", pad_inches=0.08, dpi=600)
    plt.close()


def plot_average_example_coverage_heatmap(observer_list, pathname):
    """
        Takes a list of evolutionary algorithm observers and for each computes a heatmap showing
        the proportion of individuals covering each example throughout the generations. From this, it then computes an
        'average heatmap', where the average is taken over the different observers/runs.
    """
    min_number_of_iterations = min([len(observer_list[i].get_array("example_coverage_vector")) for i in range(len(observer_list))])
    data = np.transpose(observer_list[0].get_array("example_coverage_vector")[:min_number_of_iterations])
    for i in range(1, len(observer_list)):
         data += np.transpose(observer_list[i].get_array("example_coverage_vector")[:min_number_of_iterations])
    data = data / len(observer_list)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax = sns.heatmap(data)
    plt.xlabel("Generations")
    plt.ylabel("Examples")
    plt.savefig(pathname, format="png", bbox_inches="tight", pad_inches=0.08, dpi=600)
    plt.close()


def plot_sls_best_model_example_coverage_heatmap(observer, pathname):
    """
        Takes a single HASSLE-SLS observer and computes a heatmap showing, for each example, whether
        that example is covered by the best model found. The iteration varies over the horizontal axis.
    """
    data = np.transpose(observer.get_array("best_model_correct_examples"))
    pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax = sns.heatmap(data)
    plt.xlabel("Iterations")
    plt.ylabel("Examples")
    plt.savefig(pathname, format="png", bbox_inches="tight", pad_inches=0.08, dpi=600)
    plt.close()


def plot_sls_average_best_model_example_coverage_heatmap(observer_list, pathname):
    """
        Takes a list of HASSLE-SLS observers and computes, for each observer, a heatmap showing,
        for each example, whether that example is covered by the best model found.
        The iteration varies over the horizontal axis.
        It then computes an 'average' heatmap, where the average is taken over the different observers/runs.
        In case the number of iterations differs between the runs, the smallest of these used.
    """
    min_number_of_iterations = min([len(observer_list[i].get_array("best_model_correct_examples")) for i in range(len(observer_list))])
    data = np.transpose(observer_list[0].get_array("best_model_correct_examples")[:min_number_of_iterations])
    for i in range(1, len(observer_list)):
        data += np.transpose(observer_list[i].get_array("best_model_correct_examples")[:min_number_of_iterations])
    data = data / len(observer_list)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax = sns.heatmap(data)
    plt.xlabel("Generations")
    plt.ylabel("Examples")
    plt.savefig(pathname, format="png", bbox_inches="tight", pad_inches=0.08, dpi=600)
    plt.close()


def plot_sls_current_model_example_coverage_heatmap(observer, pathname):
    """
    Takes a single HASSLE-SLS observer and computes a heatmap showing, for each example, whether
    that example is covered by the currently considered model. The iteration varies over the horizontal axis.
    """
    data = np.transpose(observer.get_array("current_model_correct_examples"))
    pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax = sns.heatmap(data)
    plt.xlabel("Iterations")
    plt.ylabel("Examples")
    plt.savefig(pathname, format="png", bbox_inches="tight", pad_inches=0.08, dpi=600)
    plt.close()


def plot_sls_average_current_model_example_coverage_heatmap(observer_list, pathname):
    """
    Takes a list of HASSLE-SLS observers and computes for each observer a heatmap showing,
    for each example, whether that example is covered by the currently considered model.
    The iteration varies over the horizontal axis.
    It then computes an 'average' heatmap, where the average is taken over the different observers/runs.
    In case the number of iterations differs between the runs, the smallest of these used.
    """
    min_number_of_iterations = min([len(observer_list[i].get_array("current_model_correct_examples")) for i in range(len(observer_list))])
    data = np.transpose(observer_list[0].get_array("current_model_correct_examples")[:min_number_of_iterations])
    for i in range(1, len(observer_list)):
        data += np.transpose(observer_list[i].get_array("current_model_correct_examples")[:min_number_of_iterations])
    data = data / len(observer_list)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax = sns.heatmap(data)
    plt.xlabel("Generations")
    plt.ylabel("Examples")
    plt.savefig(pathname, format="png", bbox_inches="tight", pad_inches=0.08, dpi=600)
    plt.close()


def plot_iterations_per_second_barplot(observer_lists, pathname, cutoff_time):
    """
    Takes a list of lists of observers. For each inner list of observers, it computes the average number of iterations
    executed per second. This is then displayed in a bar plot, in which every bar is associated with each inner list.
    """
    labels = [observer_list[0].legend_entry for observer_list in observer_lists]
    avg_num_iterations_list = []
    for observer_list in observer_lists:
        num_iterations_list = []
        for observer in observer_list:
            num_iterations_list.append(observer.get_last_log_entry()["gen_count"])
        avg_num_iterations_list.append(sum(num_iterations_list)/len(num_iterations_list))
    avg_iterations_per_second_list = [x/cutoff_time for x in avg_num_iterations_list]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    plt.bar(x, avg_iterations_per_second_list, width, color="#A1CF67", tick_label=labels)
    plt.ylabel("Iterations per second")
    plt.title(f"Iterations per second over a cutoff time of {cutoff_time} seconds")
    plt.savefig(pathname, format="png", bbox_inches="tight", pad_inches=0.08, dpi=600)
    plt.close()


def plot_averages_over_observers(observer_lists, *args, pathname, time_interval=None, cutoff_time=None,
                                 in_unit_interval=True, plot_errors=True, y_label=None, size=None, show_legend=True, legend_entries=None):
    """
    Takes a list of one or more lists of observers and plots the data contained in these observers, as
    dictated by the passed arguments (args), by taking the averages of those data over all observers, per list.
    As an example, consider the following situation:
    The passed list of lists of observers is [[observer1, observer2]]
    observer1.log = [{gen_count = 0, best_score = 0.1, gen_duration = 0.05}
                     {gen_count = 1, best_score = 0.4, gen_duration = 0.23},
                     {gen_count = 2, best_score = 0.7, gen_duration = 0.21}]
    observer2.log = [{gen_count = 0, best_score = 0.3, gen_duration = 0.04}
                     {gen_count = 1, best_score = 0.6, gen_duration = 0.24},
                     {gen_count = 2, best_score = 0.9, gen_duration = 0.21}]
    Performing plot_averages_over_observers on these two observers, without specifying a time_interval,
    would generate a plot displaying at generation 0 a best_score of 0.2,at generation 1 a best_score of 0.5
    and at generation 2 a best score of 0.8.
    Performing this function with a specified time interval instead, such as 10, for example, would generate a plot
    displaying at time 0 a best_score of 0, at time 10 a best_score of 0.2, at time 20 a best score of 0.2,
    at time 30 a best score of 0.5, at time 40 a best score of 0.5 and at time 50 a best score of 0.8.
    If a cutoff_time is provided in addition to a time interval, that time is used to limit the horizontal axis.
    If observers are provided in multiple lists contained within the observer_lists parameter, one average is computed
    and plotted per inner list. For example, if the best_score argument is asked to be plotted, then one such plot
    will be displayed per inner list.
    """
    if time_interval is None:
        scatter = ScatterData("Evolution", ["error", 2, "lx", "Generation", "ly", y_label if y_label is not None else None, "show_legend", str(show_legend)])
    else:
        scatter = ScatterData("Evolution", ["error", 2, "lx", "Seconds", "ly", y_label if y_label is not None else None, "show_legend", str(show_legend)])
        if cutoff_time is not None:
            scatter.x_lim((0, cutoff_time))
    if in_unit_interval:
        scatter.y_lim((0.45, 1))

    for i in range(len(observer_lists)):
        observer_list = observer_lists[i]
        if time_interval is None:
            x = max(observer_list, key=lambda o: len(o.get_array("gen_count"))).get_array("gen_count")
            for key in args:
                list_of_observers_arrays = [observer.get_array(key) for observer in observer_list]
                average_array = np.array(element_wise_average_variable_length_arrays(list_of_observers_arrays))
                error_array = None
                if plot_errors:
                    error_array = np.array(
                        element_wise_error_variable_length_arrays(list_of_observers_arrays, average_array))
                if legend_entries is None:
                    if observer_list[0].legend_entry is not None:
                        scatter.add_data(f"{observer_list[0].legend_entry}, {key}", x, average_array, error_array)
                    else:
                        scatter.add_data(f"Configuration {i}, {key}", x, average_array, error_array)
                else:
                    scatter.add_data(f"{legend_entries[i]}", x, average_array, error_array)

                # print(f"Array with averages of {key}: {average_array}")
            scatter.plot(log_x=False, log_y=False, size=size, filename=pathname)
        else:
            longest = max(observer_list, key=lambda o: sum(o.get_array("gen_duration")))
            duration_longest = sum(longest.get_array("gen_duration"))
            x = [k * time_interval for k in range(int(duration_longest // time_interval + 2))]
            list_of_observers_duration_arrays = [observer.get_array("gen_duration") for observer in observer_list]
            for key in args:
                list_of_observers_arrays = [observer.get_array(key) for observer in observer_list]
                list_of_transformed_observers_arrays = \
                    [transform_array_time_intervals(list_of_observers_arrays[i],
                                                    list_of_observers_duration_arrays[i],
                                                    time_interval) for i in
                     range(len(list_of_observers_duration_arrays))]
                average_array = np.array(
                    element_wise_average_variable_length_arrays(list_of_transformed_observers_arrays))
                error_array = None
                if plot_errors:
                    error_array = np.array(
                        element_wise_error_variable_length_arrays(list_of_transformed_observers_arrays, average_array))

                # index = x.index(120) + 1
                # x = x[1:index]
                # average_array = average_array[1:index]
                # error_array = error_array[1:index]

                if legend_entries is None:
                    if observer_list[0].legend_entry is not None:
                        scatter.add_data(f"{observer_list[0].legend_entry}, {key}", x, average_array, error_array)
                    else:
                        scatter.add_data(f"Configuration {i}, {key}", x, average_array, error_array)
                else:
                    scatter.add_data(f"{legend_entries[i]}", x, average_array, error_array)
                # print(f"Array with averages of {key}: {average_array}")
            scatter.plot(log_x=False, log_y=False, size=size, filename=pathname)
    plt.close()


def plot_time_relations_over_multiple_patterns(all_observer_lists,
                                               pathname,
                                               cutoff_time,
                                               varied_parameter,
                                               values_of_varied_parameter,
                                               size=None,
                                               show_legend=True,
                                               show_y_label=True,
                                               y_lim=None,
                                               y_ticks=None,
                                               y_tick_every_unit=True):
    """
    Takes a list of lists of lists observers. Each innermost list is associated with an algorithm configuration.
    Each list one level higher is associated with a problem configuration (e.g. problems with 10 variables and 10
    constraints) and contains one or more lists of observers that have observed runs of problems of that configuration.
    This function then plots the runtime per iteration relation between the different algorithm configurations, over the
    different problem configurations.
    It first normalizes the average number of iterations per second of the first algorithm configuration to 1 for every
    problem configuration. Then, for every problem configuration, it computes and plots how fast all other algorithm
    configurations were in relation to the first.
    In the resulting figure, the horizontal axis shows the different problem configurations (which are supposed to
    differ only in a single parameter (varied_parameter)). The vertical axis shows the relative speed-up with respect
    to the first algorithm configuration. The figure contains various plots: one for each algorithm configuration.
    An example of such a figure can be seen in examples/figures/output_of_plot_time_relations_over_multiple_patterns.png
    """
    scatter = ScatterData("Evolution", ["error", 2, "lx", varied_parameter, "ly", "Speed-up factor" if show_y_label else "", "y_tick_every_unit", y_tick_every_unit, "show_legend", str(show_legend)])
    scatter.y_lim(y_lim)

    num_of_configurations = len(all_observer_lists[0])
    x = values_of_varied_parameter

    # Find average evaluations per second without knowledge compilation
    averages_without_KC = []
    for j in range(len(all_observer_lists)):
        current_observer_list = all_observer_lists[j][0]
        evaluation_counts = [observer.get_last_log_entry()["number_of_evaluations"] for observer in current_observer_list]
        evaluations_per_second_list = [evaluation_count / cutoff_time for evaluation_count in evaluation_counts]
        average_evaluations_per_second = sum(evaluations_per_second_list) / len(evaluations_per_second_list)
        averages_without_KC.append(average_evaluations_per_second)
    scatter.add_data(f"{current_observer_list[0].legend_entry}", x, [1]*len(x), [0]*len(x))

    for i in range(1, num_of_configurations):
        averages_this_configuration = []
        standard_errors_this_configuration = []

        for j in range(len(all_observer_lists)):
            current_observer_list = all_observer_lists[j][i]
            evaluation_counts = [observer.get_last_log_entry()["number_of_evaluations"] for observer in current_observer_list]
            evaluations_per_second_list = [evaluation_count/cutoff_time for evaluation_count in evaluation_counts]
            average_evaluations_per_second = sum(evaluations_per_second_list)/len(evaluations_per_second_list)
            std_evaluations_per_second = np.std(evaluations_per_second_list)
            standard_error_evaluations_per_second = std_evaluations_per_second/sqrt(len(evaluations_per_second_list))
            averages_this_configuration.append(average_evaluations_per_second/averages_without_KC[j])
            standard_errors_this_configuration.append(standard_error_evaluations_per_second/averages_without_KC[j])
        scatter.add_data(f"{current_observer_list[0].legend_entry}", x, averages_this_configuration, standard_errors_this_configuration)
    scatter.plot(log_x=False, log_y=False, filename=pathname, size=size, x_ticks=values_of_varied_parameter, y_ticks=y_ticks)
    plt.close()


def plot_average_final_entry_over_multiple_patterns(all_observer_lists,
                                                    pathname,
                                                    key,
                                                    varied_parameter,
                                                    values_of_varied_parameter,
                                                    y_label=None,
                                                    size=None,
                                                    show_legend=True,
                                                    legend_entries=None,
                                                    y_lim=None,
                                                    y_ticks=None,
                                                    show_x_tick_labels=True,
                                                    show_y_tick_labels=True
                                                    ):
    """
    Takes a list of lists of lists observers. Each innermost list is associated with an algorithm configuration.
    Each list one level higher is associated with a problem configuration (e.g. problems with 10 variables and 10
    constraints) and contains one or more lists of observers that have observed runs of problems of that configuration.
    For every problem configuration, this function computes and plots the average final entry of a specified key
    (e.g. best_score) in the observer log.
    In the resulting figure, the horizontal axis shows the different problem configurations (which are supposed to
    differ only in a single parameter (varied_parameter)). The vertical axis shows the final entry of the requested key
    The figure contains various plots: one for each algorithm configuration.
    An example of such a figure can be seen in
    examples/figures/output_of_plot_average_final_entry_over_multiple_patterns.png
    """
    scatter = ScatterData("Evolution", ["error", 2, "lx", varied_parameter, "ly", y_label if y_label is not None else key, "show_legend", str(show_legend)])
    scatter.y_lim(y_lim)

    num_of_configurations = len(all_observer_lists[0])
    if legend_entries is not None and len(legend_entries) != num_of_configurations:
        raise Exception("There should be as many legend entries as there are configurations in the observer lists")
    x = values_of_varied_parameter

    for i in range(num_of_configurations):
        averages_this_configuration = []
        standard_errors_this_configuration = []
        for j in range(len(all_observer_lists)):
            current_observer_list = all_observer_lists[j][i]
            key_values = [observer.get_last_log_entry()[key] for observer in current_observer_list]
            average_key_value = sum(key_values) / len(key_values)
            std_key_values = np.std(key_values)
            standard_error_key_values = std_key_values / sqrt(len(key_values))
            averages_this_configuration.append(average_key_value)
            standard_errors_this_configuration.append(standard_error_key_values)
        if legend_entries is None:
            scatter.add_data(f"{current_observer_list[0].legend_entry}", x, averages_this_configuration, standard_errors_this_configuration)
        else:
            scatter.add_data(f"{legend_entries[i]}", x, averages_this_configuration, standard_errors_this_configuration)
    scatter.plot(log_x=False, log_y=False, filename=pathname, size=size, x_ticks=values_of_varied_parameter, y_ticks=y_ticks,
                 show_x_tick_labels=show_x_tick_labels, show_y_tick_labels=show_y_tick_labels)
    plt.close()


def plot_average_evaluation_statistics_over_multiple_patterns(all_learned_model_lists, all_target_model_lists, all_number_of_variables_lists,
                                                              pathname, varied_parameter, values_of_varied_parameter, size=None):
    """
    Takes a list of lists of lists of learned models and a list of lists of lists of respective target models.
    Each innermost list is associated with an algorithm configuration.
    Each list one level higher is associated with a problem configuration (e.g. problems with 10 variables and 10
    constraints) and contains one or more lists of models that have been learned in algorithm runs on problems of that
    configuration.
    For each learned model target model combination, this function computes the accuracy, precision, recall,
    infeasibility and regret on all possible instances (of as many variables as there are in the models) in an empty
    context. These are then plotted.
    In the resulting figures, the horizontal axis shows the different problem configurations (which are supposed to
    differ only in a single parameter (varied_parameter)). The vertical axis shows one of the metrics computed (differs
    for the different figures generated). Each figure contains various plots: one for each algorithm configuration.
    An example of such a figure can be seen in
    examples/figures/output_of_plot_average_evaluation_statistics_over_multiple_patterns.png
    """
    scatter_acc = ScatterData("Evolution", ["error", 2, "lx", varied_parameter, "ly", "Accuracy", "show_legend", "False"])
    scatter_prec = ScatterData("Evolution", ["error", 2, "lx", varied_parameter, "ly", "Precision", "show_legend", "False"])
    scatter_rec = ScatterData("Evolution", ["error", 2, "lx", varied_parameter, "ly", "Recall", "show_legend", "False"])
    scatter_inf = ScatterData("Evolution", ["error", 2, "lx", varied_parameter, "ly", "Infeasibility", "show_legend", "False"])
    scatter_reg = ScatterData("Evolution", ["error", 2, "lx", varied_parameter, "ly", "Regret", "show_legend", "False"])


    num_of_configurations = len(all_learned_model_lists[0])
    x = values_of_varied_parameter
    for i in range(num_of_configurations):
        averages_acc_this_configuration = []
        standard_errors_acc_this_configuration = []

        averages_prec_this_configuration = []
        standard_errors_prec_this_configuration = []

        averages_rec_this_configuration = []
        standard_errors_rec_this_configuration = []

        averages_inf_this_configuration = []
        standard_errors_inf_this_configuration = []

        averages_reg_this_configuration = []
        standard_errors_reg_this_configuration = []

        for j in range(len(all_learned_model_lists)):
            current_learned_model_list = all_learned_model_lists[j][i]
            current_target_model_list = all_target_model_lists[j][i]
            current_number_of_variables_list = all_number_of_variables_lists[j][i]
            acc_values = []
            prec_values = []
            rec_values = []
            inf_values = []
            reg_values = []
            for k in range(len(current_learned_model_list)):
                print(f"i is: {i} out of {range(num_of_configurations)}, j is {j} out of {range(len(all_learned_model_lists))}, "
                      f"k is {k} out of {range(len(current_learned_model_list))}")
                current_target_model = current_target_model_list[k]
                current_learned_model = current_learned_model_list[k]
                current_number_of_variables = current_number_of_variables_list[k]
                recall, precision, accuracy, reg, infeasiblity = evaluate_statistics(current_number_of_variables, current_target_model, current_learned_model, set())
                acc_values.append(accuracy/100)
                prec_values.append(precision/100)
                rec_values.append(recall/100)
                inf_values.append(infeasiblity/100)
                reg_values.append(reg/100)

            averages_acc_this_configuration.append(sum(acc_values) / len(acc_values))
            std_acc_values = np.std(acc_values)
            standard_errors_acc_this_configuration.append(std_acc_values / sqrt(len(acc_values)))

            averages_prec_this_configuration.append(sum(prec_values) / len(prec_values))
            std_prec_values = np.std(prec_values)
            standard_errors_prec_this_configuration.append(std_prec_values / sqrt(len(prec_values)))

            averages_rec_this_configuration.append(sum(rec_values) / len(rec_values))
            std_rec_values = np.std(rec_values)
            standard_errors_rec_this_configuration.append(std_rec_values / sqrt(len(rec_values)))

            averages_inf_this_configuration.append(sum(inf_values) / len(inf_values))
            std_inf_values = np.std(inf_values)
            standard_errors_inf_this_configuration.append(std_inf_values / sqrt(len(inf_values)))

            averages_reg_this_configuration.append(sum(reg_values) / len(reg_values))
            std_reg_values = np.std(reg_values)
            standard_errors_reg_this_configuration.append(std_reg_values / sqrt(len(reg_values)))

        scatter_acc.add_data("a", x, averages_acc_this_configuration, standard_errors_acc_this_configuration)
        scatter_prec.add_data("a", x, averages_prec_this_configuration, standard_errors_prec_this_configuration)
        scatter_rec.add_data("a", x, averages_rec_this_configuration, standard_errors_rec_this_configuration)
        scatter_inf.add_data("a", x, averages_inf_this_configuration, standard_errors_inf_this_configuration)
        scatter_reg.add_data("a", x, averages_reg_this_configuration, standard_errors_reg_this_configuration)

    scatter_acc.plot(log_x=False, log_y=False, filename=pathname+"accuracy_plot", size=size, x_ticks=values_of_varied_parameter)
    plt.close()
    scatter_prec.plot(log_x=False, log_y=False, filename=pathname + "precision_plot", size=size, x_ticks=values_of_varied_parameter)
    plt.close()
    scatter_rec.plot(log_x=False, log_y=False, filename=pathname + "recall_plot", size=size, x_ticks=values_of_varied_parameter)
    plt.close()
    scatter_inf.plot(log_x=False, log_y=False, filename=pathname + "infeasibility_plot", size=size, x_ticks=values_of_varied_parameter)
    plt.close()
    scatter_reg.plot(log_x=False, log_y=False, filename=pathname + "regret_plot", size=size, x_ticks=values_of_varied_parameter)
    plt.close()


def compute_average_evaluation_statistics_over_multiple_patterns(all_learned_model_lists, all_target_model_lists, all_number_of_variables_lists,
                                                              pathname):
    """
    Does the same as plot_average_evaluation_statistics_over_multiple_patterns, but saves the results in a pickle file,
    instead of plotting them.
    """
    num_of_configurations = len(all_learned_model_lists[0])

    results_list = []

    for i in range(num_of_configurations):
        averages_acc_this_configuration = []
        standard_errors_acc_this_configuration = []

        averages_prec_this_configuration = []
        standard_errors_prec_this_configuration = []

        averages_rec_this_configuration = []
        standard_errors_rec_this_configuration = []

        averages_inf_this_configuration = []
        standard_errors_inf_this_configuration = []

        averages_reg_this_configuration = []
        standard_errors_reg_this_configuration = []

        for j in range(len(all_learned_model_lists)):
            current_learned_model_list = all_learned_model_lists[j][i]
            current_target_model_list = all_target_model_lists[j][i]
            current_number_of_variables_list = all_number_of_variables_lists[j][i]
            acc_values = []
            prec_values = []
            rec_values = []
            inf_values = []
            reg_values = []
            for k in range(len(current_learned_model_list)):
                print(
                    f"i is: {i} out of {range(num_of_configurations)}, j is {j} out of {range(len(all_learned_model_lists))}, "
                    f"k is {k} out of {range(len(current_learned_model_list))}")
                current_target_model = current_target_model_list[k]
                current_learned_model = current_learned_model_list[k]
                current_number_of_variables = current_number_of_variables_list[k]
                recall, precision, accuracy, reg, infeasiblity = evaluate_statistics(current_number_of_variables,
                                                                                     current_target_model,
                                                                                     current_learned_model, set())
                acc_values.append(accuracy / 100)
                prec_values.append(precision / 100)
                rec_values.append(recall / 100)
                inf_values.append(infeasiblity / 100)
                reg_values.append(reg / 100)

            averages_acc_this_configuration.append(sum(acc_values) / len(acc_values))
            std_acc_values = np.std(acc_values)
            standard_errors_acc_this_configuration.append(std_acc_values / sqrt(len(acc_values)))

            averages_prec_this_configuration.append(sum(prec_values) / len(prec_values))
            std_prec_values = np.std(prec_values)
            standard_errors_prec_this_configuration.append(std_prec_values / sqrt(len(prec_values)))

            averages_rec_this_configuration.append(sum(rec_values) / len(rec_values))
            std_rec_values = np.std(rec_values)
            standard_errors_rec_this_configuration.append(std_rec_values / sqrt(len(rec_values)))

            averages_inf_this_configuration.append(sum(inf_values) / len(inf_values))
            std_inf_values = np.std(inf_values)
            standard_errors_inf_this_configuration.append(std_inf_values / sqrt(len(inf_values)))

            averages_reg_this_configuration.append(sum(reg_values) / len(reg_values))
            std_reg_values = np.std(reg_values)
            standard_errors_reg_this_configuration.append(std_reg_values / sqrt(len(reg_values)))
        results_list.append((averages_acc_this_configuration, standard_errors_acc_this_configuration,
                             averages_prec_this_configuration, standard_errors_prec_this_configuration,
                             averages_rec_this_configuration, standard_errors_rec_this_configuration,
                             averages_inf_this_configuration, standard_errors_inf_this_configuration,
                             averages_reg_this_configuration, standard_errors_reg_this_configuration))
        with open(pathname + "/evaluation_results.pickle", 'wb') as handle:
            pickle.dump(results_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


def plot_precomputed_average_evaluation_statistics_over_multiple_patterns(pathname,
                                                                          varied_parameter,
                                                                          values_of_varied_parameter,
                                                                          size=None,
                                                                          acc_y_lim=None,
                                                                          prec_y_lim=None,
                                                                          rec_y_lim=None,
                                                                          inf_y_lim=None,
                                                                          reg_y_lim=None,
                                                                          acc_y_ticks=None,
                                                                          prec_y_ticks=None,
                                                                          rec_y_ticks=None,
                                                                          inf_y_ticks=None,
                                                                          reg_y_ticks=None,
                                                                          acc_show_x_label=True,
                                                                          acc_show_y_label=True,
                                                                          prec_show_x_label=True,
                                                                          prec_show_y_label=True,
                                                                          rec_show_x_label=True,
                                                                          rec_show_y_label=True,
                                                                          inf_show_x_label=True,
                                                                          inf_show_y_label=True,
                                                                          reg_show_x_label=True,
                                                                          reg_show_y_label=True,
                                                                          acc_show_x_tick_labels=True,
                                                                          acc_show_y_tick_labels=True,
                                                                          prec_show_x_tick_labels=True,
                                                                          prec_show_y_tick_labels=True,
                                                                          rec_show_x_tick_labels=True,
                                                                          rec_show_y_tick_labels=True,
                                                                          inf_show_x_tick_labels=True,
                                                                          inf_show_y_tick_labels=True,
                                                                          reg_show_x_tick_labels=True,
                                                                          reg_show_y_tick_labels=True,
                                                                          ):
    """
    Does the same as plot_average_evaluation_statistics_over_multiple_patterns, but starts from precomptued evaluation
    statistics, saved in a pickle file,
    instead of plotting them.
    """
    scatter_acc = ScatterData("Evolution", ["error", 2, "lx", varied_parameter if acc_show_x_label else "", "ly", "Accuracy" if acc_show_y_label else "", "show_legend", "False"])
    scatter_prec = ScatterData("Evolution", ["error", 2, "lx", varied_parameter if prec_show_x_label else "", "ly", "Precision" if prec_show_y_label else "", "show_legend", "False"])
    scatter_rec = ScatterData("Evolution", ["error", 2, "lx", varied_parameter if rec_show_x_label else "", "ly", "Recall" if rec_show_y_label else "", "show_legend", "False"])
    scatter_inf = ScatterData("Evolution", ["error", 2, "lx", varied_parameter if inf_show_x_label else "", "ly", "Infeasibility" if inf_show_y_label else "", "show_legend", "False"])
    scatter_reg = ScatterData("Evolution", ["error", 2, "lx", varied_parameter if reg_show_x_label else "", "ly", "Regret" if reg_show_y_label else "", "show_legend", "False"])

    scatter_acc.y_lim(acc_y_lim)
    scatter_prec.y_lim(prec_y_lim)
    scatter_rec.y_lim(rec_y_lim)
    scatter_inf.y_lim(inf_y_lim)
    scatter_reg.y_lim(reg_y_lim)

    with open(pathname+"/evaluation_results.pickle", 'rb') as handle:
        evaluation_results = pickle.load(handle)

    for config_results in evaluation_results:
        scatter_acc.add_data("a", values_of_varied_parameter, config_results[0], config_results[1])
        scatter_prec.add_data("a", values_of_varied_parameter, config_results[2], config_results[3])
        scatter_rec.add_data("a", values_of_varied_parameter, config_results[4], config_results[5])
        scatter_inf.add_data("a", values_of_varied_parameter, config_results[6], config_results[7])
        scatter_reg.add_data("a", values_of_varied_parameter, config_results[8], config_results[9])
    scatter_acc.plot(log_x=False, log_y=False, filename=pathname+"/accuracy_plot", size=size, x_ticks=values_of_varied_parameter, y_ticks=acc_y_ticks, show_x_tick_labels=acc_show_x_tick_labels, show_y_tick_labels=acc_show_y_tick_labels)
    plt.close()
    scatter_prec.plot(log_x=False, log_y=False, filename=pathname + "/precision_plot", size=size, x_ticks=values_of_varied_parameter, y_ticks=prec_y_ticks, show_x_tick_labels=prec_show_x_tick_labels, show_y_tick_labels=prec_show_y_tick_labels)
    plt.close()
    scatter_rec.plot(log_x=False, log_y=False, filename=pathname + "/recall_plot", size=size, x_ticks=values_of_varied_parameter, y_ticks=rec_y_ticks, show_x_tick_labels=rec_show_x_tick_labels, show_y_tick_labels=rec_show_y_tick_labels)
    plt.close()
    scatter_inf.plot(log_x=False, log_y=False, filename=pathname + "/infeasibility_plot", size=size, x_ticks=values_of_varied_parameter, y_ticks=inf_y_ticks, show_x_tick_labels=inf_show_x_tick_labels, show_y_tick_labels=inf_show_y_tick_labels)
    plt.close()
    scatter_reg.plot(log_x=False, log_y=False, filename=pathname + "/regret_plot", size=size, x_ticks=values_of_varied_parameter, y_ticks=reg_y_ticks, show_x_tick_labels=reg_show_x_tick_labels, show_y_tick_labels=reg_show_y_tick_labels)
    plt.close()


def element_wise_average_variable_length_arrays(arrays):
    """
    Computes and outputs an array in which every element is the average of all elements at that index
    in the supplied (variable-length) arrays.
    Consider the following example:
    arrays =    [[1, 3, 5]
                 [3, 1, 7]
                 [2, 1]]
    output_array = [2, 1.66667, 6]
    """
    list_of_averages = []
    max_length = max([len(array) for array in arrays])
    for i in range(max_length):
        average = 0
        number_of_arrays_with_this_index = 0
        for array in arrays:
            if i < len(array):
                average += array[i]
                number_of_arrays_with_this_index += 1
        average = average / number_of_arrays_with_this_index
        list_of_averages.append(average)
    return list_of_averages


def element_wise_error_variable_length_arrays(arrays, provided_average_array=None):
    """
    Works similarly to element_wise_average_variable_length_arrays, except that here it is not averages, but standard
    errors which are calculated. The standard error is defined as the standard deviation divided by the square root of
    the number of runs/arrays. Optionally, this function takes a pre-computed array of averages that should be used in
    the standard error computation.
    """
    if provided_average_array is not None:
        average_array = provided_average_array
    else:
        average_array = []
        max_length = max([len(array) for array in arrays])
        for i in range(max_length):
            average = 0
            number_of_arrays_with_this_index = 0
            for array in arrays:
                if i < len(array):
                    average += array[i]
                    number_of_arrays_with_this_index += 1
            average = average / number_of_arrays_with_this_index
            average_array.append(average)

    list_of_errors = []
    max_length = max([len(array) for array in arrays])
    for i in range(max_length):
        standard_deviation = 0
        number_of_arrays_with_this_index = 0
        for array in arrays:
            if i < len(array):
                standard_deviation += (array[i] - average_array[i]) ** 2
                number_of_arrays_with_this_index += 1
        standard_deviation = sqrt(standard_deviation/number_of_arrays_with_this_index)
        standard_error = standard_deviation / sqrt(number_of_arrays_with_this_index)
        list_of_errors.append(standard_error)
    return list_of_errors


def transform_array_time_intervals(array, duration_array, time_interval, default_value=0):
    """
    Takes two arrays of the same length, a specified time interval and a default value.
    The first array contains datapoints and the second contains the durations of the generations from which the
    datapoints stem. The output of this method is an array that contains the most recent datapoint, for points in time
    that are multiples of the given time interval. For time points at which there is no most recent value yet, the
    provided default is used.
    Consider the following example:
    array = [1, 3, 4]
    duration_array = [5, 7, 3]
    time_interval = 2
    default_value = 0
    output_array = [0, 0, 0, 1, 1, 1, 3, 3, 4]
    """
    if len(array) != len(duration_array):
        raise Exception("The two provided arrays are not of the same length")
    output_array = []
    cumulative_duration_array = transform_to_cumulative(duration_array)
    i = -1
    k = 0
    while i != len(array) - 1:
        current_time_point = k * time_interval
        while i != len(array) - 1 and cumulative_duration_array[i + 1] <= current_time_point:
            i += 1
        if i == -1:
            output_array.append(default_value)
        else:
            output_array.append(array[i])
        k += 1
    return output_array


def transform_to_cumulative(duration_array):
    """
    Transforms an array of durations to a cumulative array of durations.
    Consider the following example:
    duration_array = [1, 2, 3]
    output_array = [1, 3, 6]
    """
    output_array = []
    cumulative_value = 0
    for element in duration_array:
        cumulative_value += element
        output_array.append(cumulative_value)
    return output_array
