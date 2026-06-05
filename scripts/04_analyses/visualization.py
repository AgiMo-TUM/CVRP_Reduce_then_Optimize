"""Plot-generating functions extracted from generate_samples_for_viz.py."""
import os
import gzip
import pickle as pkl
import numpy as np
import gurobipy as gp
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pyvrp import Model as HGS_Model
from pyvrp.stop import MaxRuntime as HGS_MaxRuntime
import re
import math
import random

from core.utils.cvrp import CVRP_node
from core.utils.cvrp import CVRP
from core.cvrp_solvers.ip_grb import cvrp
from core.cvrp_solvers.ip_grb import sol_vals
from core.cvrp_solvers.heuristics import Clark_Wright_heuristic
from core.evaluation.benchmarking_utils import get_mip_gap_by_method
from core.evaluation.benchmarking_utils import get_num_edges_by_method
from core.evaluation.benchmarking_utils import get_optgaps_by_method
from core.evaluation.benchmarking_utils import get_objval_by_method
from core.evaluation.benchmarking_utils import get_runtimes_by_method
from core.evaluation.benchmarking_utils import get_solver_runtimes_by_method
from core.evaluation.benchmarking_utils import get_num_arcs_used_by_method
# `get_missing_arcs_by_method` is imported lazily inside `generate_missing_arc_plot`
# because the utility is not defined in `core/evaluation/benchmarking_utils.py`
# yet; the rest of the module must still load.
# `get_optgaps_by_method_missing_arcs` is imported lazily inside
# `generate_missing_arc_plot` for the same reason as `get_missing_arcs_by_method`.

from core.data_processing.data_utils import dict_to_instance


from core.fctp_heuristics_julia.python_wrapper import Frank_Wolfe_regularisation_env

from core.cvrp_solvers.ip_grb import cvrp_via_VRP_Easy

from core.utils.additional_features import MST_feature
from core.utils.additional_features import compute_voronoi_adjacency
from core.utils.additional_features import compute_Clark_Wright_savings


from core.cvrp_solvers.heuristics import heu_solve_HGS_VRPTW
from core.data_processing.data_utils import load_instance
# `load_with_numpy_fix` was unused; dropped.

from collections import defaultdict


def generate_benchmark_plot(solution_dir):
    """Plots, for each benchmark method, its per-instance optimality gap percentage as a stacked grid of line subplots."""
    opt_gap_data = get_optgaps_by_method(solution_dir)


    # Use the actual per-method instance count rather than a hard-coded 52.
    x = range(len(next(iter(opt_gap_data.values()))))

    num_methods = len(opt_gap_data)
    fig, axes = plt.subplots(num_methods, 1, figsize=(10, 6), sharex=True)

    if num_methods == 1:
        axes = [axes]

    for ax, (method, values) in zip(axes, opt_gap_data.items()):
        ax.plot(x, values[:len(next(iter(opt_gap_data.values())))], marker='o', linestyle='-', label=method)
        ax.set_title(f"Method: {method}", fontsize=10)
        ax.set_ylabel("Opt_gap_value %")
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=8)

    # Final touches
    axes[-1].set_xlabel("Instance index")
    plt.tight_layout()
    plt.show()


def generate_runtime_plot(best_instance_path, instances_path, reduced_instance_path):
    """Compares HGS vs reduced-method gap-vs-runtime curves on one line plot, one curve per reduction threshold."""

    best_score_dict = {}

    for instance_file in os.listdir(best_instance_path):
            with gzip.open(os.path.join(best_instance_path, instance_file), "rb") as f:
                result_dict = pkl.load(f)

            instance, instance_solution  = dict_to_instance(result_dict)

            exact_objective_value = sum(instance_solution[(u,v)] * instance.arc_costs[k]
                                 for k, (u,v) in
                             enumerate(zip(instance.arc_index[0],
                                            instance.arc_index[1])))
            print(instance_file, " with exact_objective_value : ", exact_objective_value)
            m = re.search(r"(\d+)(?=\.pkl\.gz$)", instance_file)
            extract_key = int(m.group(1))
            best_score_dict[extract_key] = exact_objective_value
    print("best_score_dict : ", best_score_dict)
    HGS_runtime_dict = {5000 : 0}

    for folder in os.listdir(instances_path):

        folder_path = os.path.join(instances_path, folder)
        m = re.search(r"shorter_(\d+)", folder)
        extract_name = int(m.group(1))

        best_values_list = []
        for instance_file in os.listdir(folder_path):
            with gzip.open(os.path.join(folder_path, instance_file), "rb") as f:
                result_dict = pkl.load(f)

            m = re.search(r"(\d+)(?=\.pkl\.gz$)", instance_file)
            extract_key = int(m.group(1))
            best_values_gap = (result_dict["optimal_value"]- best_score_dict[extract_key])/best_score_dict[extract_key]
            print("optimal_value : " , result_dict["optimal_value"])
            print("best gap = ", best_values_gap)
            best_values_list.append(best_values_gap)

        HGS_runtime_dict[extract_name] = sum(best_values_list)/len(best_values_list)
        print(HGS_runtime_dict)

    reduced_runtime_dict = {}
    for folder in os.listdir(reduced_instance_path):

        folder_path = os.path.join(reduced_instance_path, folder)
        m = re.findall(r"(\d+)", folder)
        extract_name_1 = int(m[-1])
        opt_gap_data = get_optgaps_by_method(folder_path)

        for (method, values) in opt_gap_data.items():
            print("method : ", method)
            m = re.search(r"(\d+(?:\.\d+)?)$", method)
            extract_name_2 = float(m.group(1))
            if extract_name_2 not in reduced_runtime_dict:
                reduced_runtime_dict[extract_name_2] = {}
            reduced_runtime_dict[extract_name_2][extract_name_1] = sum(values) / len(values)
            print(float(m.group(1)), " with reduced_value_mean = ", sum(values)/len(values))

    plt.figure(figsize=(10, 6))

    X = sorted(HGS_runtime_dict.keys())
    y1 = HGS_runtime_dict
    y_list = reduced_runtime_dict

    y_hgs = [y1[x] for x in X]
    plt.plot(X, y_hgs, marker="o", linewidth=2.5, label="HGS")


    for reduced_key in sorted(y_list.keys()):
        subdict = y_list[reduced_key]

        # Ensure alignment: missing values become None
        y_vals = [subdict.get(x, None) for x in X]

        plt.plot(
            X,
            y_vals,
            marker="o",
            linestyle="--",
            linewidth=1.8,
            label=str(reduced_key)
        )

    plt.xlabel("Runtime")
    plt.ylabel("Gap Value")
    plt.title("HGS vs Reduced Methods Runtime Comparison")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="Method", fontsize=10)

    plt.tight_layout()
    plt.show()


def generate_exact_plot(best_instance_path, reduced_instance_path):
    """Scatters mean optimality gap versus mean runtime for each reduction threshold, annotating each point with its threshold."""

    best_score_dict = {}

    for instance_file in os.listdir(best_instance_path):
            with gzip.open(os.path.join(best_instance_path, instance_file), "rb") as f:
                result_dict = pkl.load(f)

            instance, instance_solution  = dict_to_instance(result_dict)

            exact_objective_value = sum(instance_solution[(u,v)] * instance.arc_costs[k]
                                 for k, (u,v) in
                             enumerate(zip(instance.arc_index[0],
                                            instance.arc_index[1])))
            print(instance_file, " with exact_objective_value : ", exact_objective_value)
            m = re.search(r"(\d+)(?=\.pkl\.gz$)", instance_file)
            extract_key = int(m.group(1))
            best_score_dict[extract_key] = exact_objective_value
    HGS_runtime_dict = {1000 : 0}

    reduced_runtime_dict = {}
    for folder in os.listdir(reduced_instance_path):

        folder_path = os.path.join(reduced_instance_path, folder)
        m = re.findall(r"(\d+)", folder)
        extract_name_1 = int(m[-1])
        opt_gap_data = get_optgaps_by_method(folder_path)
        obj_val_data = get_objval_by_method(folder_path)
        runtime =get_runtimes_by_method(folder_path)
        print("runtime : ", runtime)
        print("obj_value : ", obj_val_data)

        for (method, values) in opt_gap_data.items():
            print("method : ", method)
            m = re.search(r"(\d+(?:\.\d+)?)$", method)
            extract_name_2 = float(m.group(1))
            if extract_name_2 not in reduced_runtime_dict:
                reduced_runtime_dict[extract_name_2] = {}
            reduced_runtime_dict[extract_name_2] = (sum(runtime[method])/len(runtime[method]), sum(values) / len(values))
        print("reduced_runtime_dict : ", reduced_runtime_dict)

    plt.figure(figsize=(8,5))
    for key, (runtime, objective) in reduced_runtime_dict.items():
         plt.scatter(runtime, objective, label=str(key))
         plt.text(runtime, objective, f" {key}", fontsize=9, va='center')

    plt.xlabel("Runtime (s)")
    plt.ylabel("Optimal gap HGS 1000s")
    plt.title("Optimal gap vs Runtime for each threshold")
    plt.legend(title="redution coefficient")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    #     subdict = y_list[reduced_key]

    #     y_vals = [subdict.get(x, None) for x in X]

    #     plt.plot(
    #         X,
    #         y_vals,
    #         marker="o",
    #         linestyle="--",
    #         linewidth=1.8,
    #         label=str(reduced_key)
    #     )


def generate_missing_arc_plot(benchmark_missing_arc_path, original_path):
    """Scatter plot of optimality gap versus number of missing arcs, with one color per reduction runtime."""
    from core.evaluation.benchmarking_utils import get_missing_arcs_by_method  # noqa: WPS433
    from core.evaluation.benchmarking_utils import get_optgaps_by_method_missing_arcs  # noqa: WPS433

    missing_arcs = get_missing_arcs_by_method(benchmark_missing_arc_path)
    print(missing_arcs)
    plot_by_runtime_dict = {}
    i = 0
    for folder in os.listdir(original_path):

        folder_path = os.path.join(original_path, folder)
        m = re.findall(r"(\d+)", folder)
        extract_name_1 = int(m[-1])
        opt_gap_data = get_optgaps_by_method_missing_arcs(folder_path)
        print(opt_gap_data)

        if extract_name_1 not in plot_by_runtime_dict:
            plot_by_runtime_dict[extract_name_1] = []

        for (method, values) in opt_gap_data.items():
            print("method : ", method)
            m = re.search(r"(\d+(?:\.\d+)?)$", method)
            extract_name_2 = float(m.group(1))

            for key in values.keys():

                if key in missing_arcs[extract_name_2]:
                    plot_by_runtime_dict[extract_name_1].append((missing_arcs[extract_name_2][key], values[key]))

    print(plot_by_runtime_dict)

    keys = list(plot_by_runtime_dict.keys())
    colors = plt.cm.tab10(range(len(keys))) # tab10 gives 10 distinct colors
    plt.figure(figsize=(8, 6))
    for color, key in zip(colors, keys):
         points = plot_by_runtime_dict[key]
         xs = [p[0] for p in points]
         ys = [p[1] for p in points]
         plt.scatter(xs, ys, color=color, label=f"Runtime reduced {key}")
    plt.xlabel("number missing arcs")
    plt.ylabel("Gap best-known HGS solution")
    plt.title("Missing arcs versus Opt Gap")
    plt.legend()
    plt.grid(True)
    plt.show()


def _generate_runtime_missing_arc_plot_impl(benchmarking_path, with_bars=False):
    """Aggregates benchmarking results into per-threshold mean/median value-gap and runtime curves with shaded std bands, then draws histograms of gap and runtime per threshold."""
    performance_dict = {}
    better_runtime_dict = {}
    out_runtime_limit_dict = {}
    # Build the cost dict on the fly from each result pickle's
    # `exact_objective_value`. Keys are the raw file names so we no longer rely
    # on the legacy `sol_instance_(\d+_\d+).pkl.gz` regex or an external
    # `cost_dict.pkl` aggregator.
    cost_dict = {}
    # iterate over subfolders inside the benchmarking directory
    for folder_name in os.listdir(benchmarking_path):
        folder_path = os.path.join(benchmarking_path, folder_name)
        if not os.path.isdir(folder_path):
            continue


        # iterate over files inside each folder
        count_instances = 0
        for file_name in os.listdir(folder_path):

            count_instances += 1

            if file_name not in performance_dict.keys():
                performance_dict[file_name] = []
            file_path = os.path.join(folder_path, file_name)


            with gzip.open(file_path, "rb") as f:
                result_dict = pkl.load(f)

            # Populate per-instance baseline cost from the result dict itself.
            if file_name not in cost_dict and "exact_objective_value" in result_dict:
                cost_dict[file_name] = result_dict["exact_objective_value"]

            if result_dict["solver_value"] > 0 :
                if folder_name != "HGS" and folder_name != "HGS_80" :
                    if folder_name  == -0.5 or folder_name ==1:
                        performance_dict[file_name].append(
                            (float(folder_name),(result_dict["solver_value"],
                            result_dict["solver_runtime"],
                            result_dict["num_arcs_pred"],
                            result_dict["exact_objective_value"],
                            result_dict["num_arcs_enriched"]-result_dict["num_arcs_pred"]))
                        )

                    else:
                        performance_dict[file_name].append(
                            (float(folder_name),(result_dict["solver_value"],
                            result_dict["runtime"],
                            result_dict["num_arcs_pred"],
                            result_dict["exact_objective_value"],
                            result_dict["num_arcs_enriched"]-result_dict["num_arcs_pred"]))
                        )
                # else :
                #         performance_dict[file_name].append(
                #             (0.9,(result_dict["solver_value"],
                #             result_dict["solver_runtime"],
                #             result_dict["num_missing_arcs"]))
                #         )
                #     else :
                #         performance_dict[file_name].append(
                #             (0.8,(result_dict["solver_value"],
                #             result_dict["solver_runtime"],
                #             result_dict["num_missing_arcs"]))
                #         )
            else:
                print(file_name, " ", (float(folder_name),(result_dict["solver_value"],
                    result_dict["solver_runtime"],
                    result_dict["num_missing_arcs"])))
                if float(folder_name) == 1.0 or float(folder_name)==-0.5:
                    out_runtime_limit_dict[file_name] = 1
        print("count instances = ", count_instances)
    print("number of out_runtime_limit files : ", len(out_runtime_limit_dict.keys()))


    new_performance_dict = {}
    renew_performance_dict = {}
    better_runtime_count = 0
    better_runtime_bool  = False
    best_runtime_value  = 5001
    for key in performance_dict.keys():
        # `cost_dict` is keyed by file name (built from each result pickle's
        # `exact_objective_value` above); no regex parsing needed.
        optimal_value = cost_dict.get(key)
        if optimal_value is None:
            # Skip instances with no recorded baseline.
            continue
        optimal_runtime= 5000
        new_performance_dict[key] = []
        renew_performance_dict[key] = []
        found_threshold = False
        HGS_runtime = 1000
        for threshold in performance_dict[key]:
            if threshold[0] == 1.0 or threshold[0] == -0.5:
                found_threshold  = True
                optimal_runtime = threshold[1][1]
                HGS_runtime =  80
                break
        for threshold in performance_dict[key]:
            new_performance_dict[key].append((threshold[0], ((threshold[1][0]- optimal_value)*100/optimal_value,
                                                    (threshold[1][1]-optimal_runtime)*100/optimal_runtime , threshold[1][2]/5050, threshold[1][4])))
        if not found_threshold :
            #         if not better_runtime_bool:
            #                 best_runtime_value = threshold[1][1]
            #             better_runtime_count +=1
            #             better_runtime_bool = True
            #             better_runtime_threshold = threshold[0]
            #                                         (threshold[1][1]-5000)/5000 , threshold[1][2])))

            for threshold in performance_dict[key]:
                if threshold[0] != 1 and threshold[0] != -0.5:
                    optimal_value = threshold[1][3]
                    renew_performance_dict[key].append((threshold[0], ((threshold[1][0]- threshold[1][3])*100/threshold[1][3],
                                                    threshold[1][1] , threshold[1][2]/5050)))
        better_runtime_bool = False
        if best_runtime_value < 5001:
            better_runtime_dict[key] = (best_runtime_value, better_runtime_threshold)
            best_runtime_value  = 5001
    print("better_runtime_count : ", better_runtime_count)
    print("better_runtime_dict : ", better_runtime_dict)


    #     handles.append(plt.Line2D([], [], marker=marker_map[p], linestyle='None', markersize=10))
    #     labels.append(f"threshold = {p}")


    #     for p, triple in values:
    #         grouped_value[p].append(triple[0])     # tuple[1][0]
    #         grouped_runtime[p].append(triple[1])   # tuple[1][1]


    #     x = list(range(len(grouped_value[p])))

    #     ax1.plot(
    #         x, grouped_value[p],
    #         color=colors[idx],
    #         marker='o',
    #         label=f"value p={p}"
    #     )

    #     ax2.plot(
    #         x, grouped_runtime[p],
    #         color=colors[idx],
    #         marker='x',
    #         linestyle='--',
    #         label=f"runtime p={p}"
    #     )


    # Extract all unique parameter values
    param_values = sorted({entry[0] for values in new_performance_dict.values() for entry in values})

        # Remove threshold -0.5 everywhere


    # Prepare accumulators
    value_acc = {p: [] for p in param_values}
    runtime_acc = {p: [] for p in param_values}
    pred_arcs_acc = {p: [] for p in param_values}

    # Fill accumulators
    for instance, values in new_performance_dict.items():
        for p, triple in values:
            value_acc[p].append(triple[0])     # tuple[1][0]
            runtime_acc[p].append(triple[1])   # tuple[1][1]
            pred_arcs_acc[p].append(triple[2])

    filtered_param_values = [p for p in param_values if p != -0.5]

    value_acc = {p: value_acc[p] for p in filtered_param_values}
    pred_arcs_acc = {p: pred_arcs_acc[p] for p in filtered_param_values}
    runtime_acc = {p: runtime_acc[p] for p in filtered_param_values}

    param_values = filtered_param_values

    # Compute averages and standard deviations
    avg_values = np.array([np.mean(value_acc[p]) for p in param_values])
    std_values = np.array([np.std(value_acc[p]) for p in param_values])

    avg_pred_arcs  = np.array([np.mean(pred_arcs_acc[p]) for p in param_values])
    std_pred_arcs = np.array([np.std(pred_arcs_acc[p]) for p in param_values])

    avg_runtimes = []
    std_runtimes = []

    for p in param_values:
        arr = np.array(runtime_acc[p])

        # Compute 95th percentile threshold
        cutoff = np.percentile(arr, 99)

        trimmed = arr[arr <= cutoff]

        # Compute mean and std on trimmed data
        avg_runtimes.append(np.mean(trimmed))
        std_runtimes.append(np.std(trimmed))

    avg_runtimes = np.array(avg_runtimes)
    std_runtimes = np.array(std_runtimes)


    # Median value gap
    median_values = np.array([np.median(value_acc[p]) for p in param_values])
    median_runtimes = np.array([np.median(runtime_acc[p]) for p in param_values])


    plt.rcParams.update({
        "font.size": 24,          # base font size
        "axes.titlesize": 26,     # title
        "axes.labelsize": 24,     # x/y labels
        "xtick.labelsize": 22,    # x tick labels
        "ytick.labelsize": 22,    # y tick labels
        "legend.fontsize": 18     # legend
    })

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Left axis: averaged values

    # Create equally spaced positions
    x_positions = np.arange(len(param_values))

    # Plot using x_positions instead of param_values

    # Set equally spaced ticks with threshold labels
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([str(p) for p in param_values], rotation=45)

    ax1.set_xlabel("Threshold")

    ax1.plot(x_positions, avg_values, marker='o', color='tab:blue', label="Average RVG gap %")
    ax1.fill_between(x_positions,
                     avg_values +std_values ,
                     avg_values - std_values,
                     color='tab:blue', alpha=0.2)

    ax1.plot(
        x_positions, median_values,
        linestyle='--', linewidth=2,
        color='navy',
        label="Median RVG gap %"
    )

    # Right axis: averaged runtimes
    ax2.plot(x_positions, avg_runtimes, marker='x', linestyle='--', color='tab:red', label="Average RRG gap % (Quantile 99%)")
    ax2.fill_between(x_positions,
                     avg_runtimes +std_runtimes ,
                     avg_runtimes -std_runtimes,
                     color='tab:red', alpha=0.2)

    ax2.plot(
        x_positions, median_runtimes,
        linestyle=':',
        linewidth=2,
        color='darkred',
        label="Median RRG gap %"
    )

    # X-axis: show only parameter values

    ax1.set_xlabel("Size-threshold")
    ax1.set_ylabel("Average RVG gap %", color='tab:blue')
    ax2.set_ylabel("Average RRG gap %", color='tab:red')

    ax1.set_title("Average RVG and RGG % per Threshold")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    all_lines  = lines1 + lines2
    all_labels = labels1 + labels2

    # Split into two halves
    mid = len(all_lines) // 2

    legend1 = ax1.legend(
        all_lines[:mid],
        all_labels[:mid],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0)
    )

    legend2 = ax1.legend(
        all_lines[mid:],
        all_labels[mid:],
        loc="upper left",
        bbox_to_anchor=(0.39, 1.0)   # adjust horizontal spacing if needed
    )

    ax1.add_artist(legend1)
    ax1.add_artist(legend2)


    ax1.grid(True)
    plt.tight_layout()
    plt.show()


    #         labels.append("0")
    #     else:
    #         labels.append(str(p))


    #     x_positions - bar_width/2,
    #     avg_pred_arcs,
    #     width=bar_width,
    #     color="tab:green",
    #     alpha=0.6,
    #     label="Average predicted arcs (%)"
    # )

    #     ax3.text(
    #         x_positions[i] - bar_width/2,
    #         val,
    #         f"{val*100:.1f}%",
    #         ha="center",
    #         va="bottom",
    #         fontsize=20,
    #         color="tab:green"
    #     )

    #     x_positions, avg_values,
    #     marker='o', color='tab:blue',
    #     label="Average RVG gap %"
    # )
    #     x_positions,
    #     avg_values + std_values,
    #     avg_values - std_values,
    #     color='tab:blue', alpha=0.2
    # )

    #     x_positions, median_values,
    #     linestyle='--', linewidth=2,
    #     color='navy',
    #     label="Median RVG gap %"
    # )


    #     x_positions, avg_runtimes,
    #     marker='x', linestyle='--',
    #     color='tab:red',
    #     label="Average RRG gap % (Quantile 99%)"
    # )
    #     x_positions,
    #     avg_runtimes + std_runtimes,
    #     avg_runtimes - std_runtimes,
    #     color='tab:red', alpha=0.2
    # )

    #     x_positions, median_runtimes,
    #     linestyle=':',
    #     linewidth=2,
    #     color='darkred',
    #     label="Median RRG gap %"
    # )


    #     all_lines[:mid],
    #     all_labels[:mid],
    #     loc="upper left",
    #     bbox_to_anchor=(0.61, 1.0)
    # )

    #     all_lines[mid:],
    #     all_labels[mid:],
    #     loc="upper left",
    #     bbox_to_anchor=(0.0, 1.0)   # adjust horizontal spacing if needed
    # )


    param_values = sorted({entry[0] for values in renew_performance_dict.values() for entry in values})

    # Prepare accumulators
    value_acc = {p: [] for p in param_values}
    runtime_acc = {p: [] for p in param_values}

    # Fill accumulators
    for instance, values in renew_performance_dict.items():
        for p, triple in values:
            value_acc[p].append(triple[0])     # tuple[1][0]
            runtime_acc[p].append(triple[1])   # tuple[1][1]

    # Compute averages and standard deviations
    avg_values = np.array([np.mean(value_acc[p]) for p in param_values])
    std_values = np.array([np.std(value_acc[p]) for p in param_values])


    avg_runtimes = []
    std_runtimes = []

    for p in param_values:
        arr = np.array(runtime_acc[p])

        # Compute 95th percentile threshold
        cutoff = np.percentile(arr, 100)

        trimmed = arr[arr <= cutoff]

        # Compute mean and std on trimmed data
        avg_runtimes.append(np.mean(trimmed))
        std_runtimes.append(np.std(trimmed))

    avg_runtimes = np.array(avg_runtimes)
    std_runtimes = np.array(std_runtimes)


    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Left axis: averaged values
    ax1.plot(param_values, avg_values, marker='o', color='tab:blue', label="Average complete relative gap")
    ax1.fill_between(param_values,
                     avg_values +std_values ,
                     avg_values - std_values,
                     color='tab:blue', alpha=0.2)

    # Right axis: averaged runtimes
    ax2.plot(param_values, avg_runtimes, marker='x', linestyle='--', color='tab:red', label="Mean complete relative runtime")
    ax2.fill_between(param_values,
                     avg_runtimes +std_runtimes ,
                     avg_runtimes -std_runtimes,
                     color='tab:red', alpha=0.2)

    # X-axis: show only parameter values
    ax1.set_xticks(param_values)
    ax1.set_xticklabels([str(p) for p in param_values], rotation=45)

    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Average complete relative gap", color='tab:blue')
    ax2.set_ylabel("Mean complete relative runtime", color='tab:red')

    ax1.set_title("Average complete relative Gap and mean relative Runtime per Threshold for unsolved CVRPTW")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.grid(True)
    plt.tight_layout()
    plt.show()

  # Extract all threshold values
    param_values = sorted({
        entry[0]
        for values in new_performance_dict.values()
        for entry in values
    })

    # Remove unwanted parameters
    excluded = {0.266667, 1.0}
    param_values = [p for p in param_values if p not in excluded]

    # Prepare accumulators
    value_acc = {p: [] for p in param_values}
    runtime_acc = {p: [] for p in param_values}

    # Fill accumulators
    for instance, values in new_performance_dict.items():
        for p, triple in values:
            if p in value_acc:
                value_acc[p].append(triple[0])
                runtime_acc[p].append(triple[1])

    # Create figure
    num_params = len(param_values)
    fig, axes = plt.subplots(2, num_params, figsize=(5 * num_params, 10))

    if num_params == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    # Loop over thresholds
    for i, p in enumerate(param_values):

        # GAP subplot
        ax_gap = axes[0, i]
        gaps = value_acc[p]
        ax_gap.hist(gaps, bins=20, color="tab:blue", alpha=0.7)
        if p==0.9:
            ax_gap.set_title(f"HGS — Gap")
        else :
            ax_gap.set_title(f"Threshold {p} — Gap %")

        ax_gap.text(
            0.98, 0.95,
            f"{len(gaps)} instances",
            transform=ax_gap.transAxes,
            ha="right", va="top",
            fontsize=10
        )

        # RUNTIME subplot
        ax_rt = axes[1, i]
        runtimes = np.array(runtime_acc[p])

        # Clip only the extreme positive tail
        low = np.min(runtimes)
        high = np.percentile(runtimes, 99)

        clipped = np.clip(runtimes, low, high)
        num_clipped = np.sum(runtimes > high)
        num_uncut = len(runtimes) - num_clipped

        ax_rt.hist(clipped, bins=30, color="tab:red", alpha=0.7)
        ax_rt.set_xlim(low, high)

        ax_rt.set_xlabel("Relative Runtime %")

        ax_rt.text(
            0.98, 0.95,
            f"{num_uncut} instances",
            transform=ax_rt.transAxes,
            ha="right", va="top",
            fontsize=10
        )

    plt.tight_layout()
    plt.show()


    excluded = {1, 0.266667}
    param_values = [p for p in param_values if p not in excluded]

    # Prepare accumulator
    value_acc = {p: [] for p in param_values}

    # Fill accumulator
    for instance, values in new_performance_dict.items():
        for p, triple in values:
            if p in value_acc:
                value_acc[p].append(triple[0])   # triple[0] = relative gap

    # Number of thresholds
    num_params = len(param_values)

    # Compute number of columns (2 rows)
    ncols = int(np.ceil(num_params / 2))
    nrows = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 10))

    # Flatten axes for easy indexing
    axes = axes.flatten()

    # Loop over thresholds
    for i, p in enumerate(param_values):
        ax = axes[i]
        gaps = value_acc[p]

        ax.hist(gaps, bins=20, color="tab:blue", alpha=0.7)
        if p == 0.9:
            ax.set_title(f"HGS(200s) — Gap")
        else:
            ax.set_title(f"Threshold {p} — Gap")

        ax.text(
            0.98, 0.95,
            f"{len(gaps)} instances",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=10
        )

    # Hide unused subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
    # Extract all threshold values
    param_values = sorted({
        entry[0]
        for values in renew_performance_dict.values()
        for entry in values
    })

    # Remove unwanted parameters
    excluded = {1, 0.266667}
    param_values = [p for p in param_values if p not in excluded]

    # Prepare accumulators
    value_acc = {p: [] for p in param_values}
    runtime_acc = {p: [] for p in param_values}

    # Fill accumulators
    for instance, values in renew_performance_dict.items():
        for p, triple in values:
            if p in value_acc:
                value_acc[p].append(triple[0])
                runtime_acc[p].append(triple[1])

    # Create figure
    num_params = len(param_values)
    fig, axes = plt.subplots(2, num_params, figsize=(5 * num_params, 10))

    if num_params == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    # Loop over thresholds
    for i, p in enumerate(param_values):

        # GAP subplot
        ax_gap = axes[0, i]
        gaps = value_acc[p]
        ax_gap.hist(gaps, bins=20, color="tab:blue", alpha=0.7)
        if p ==0.9 :
            ax_gap.set_title(f"HGS(200s) — Gap")
        else:
            ax_gap.set_title(f"Threshold {p} — Gap")

        ax_gap.text(
            0.98, 0.95,
            f"{len(gaps)} instances",
            transform=ax_gap.transAxes,
            ha="right", va="top",
            fontsize=10
        )

        # RUNTIME subplot
        ax_rt = axes[1, i]
        runtimes = np.array(runtime_acc[p])

        # Clip only the extreme positive tail
        low = np.min(runtimes)
        high = np.percentile(runtimes, 99)

        clipped = np.clip(runtimes, low, high)

        ax_rt.hist(clipped, bins=30, color="tab:red", alpha=0.7)
        ax_rt.set_xlim(low, high)

        ax_rt.set_xlabel("Runtime")

        ax_rt.text(
            0.98, 0.95,
            f"{len(runtimes)} instances",
            transform=ax_rt.transAxes,
            ha="right", va="top",
            fontsize=10
        )

    plt.tight_layout()
    plt.show()


    return performance_dict


def generate_runtime_missing_arc_plot(benchmarking_path):
    """Runtime vs missing-arc plot — line/area version (no bars)."""
    return _generate_runtime_missing_arc_plot_impl(benchmarking_path, with_bars=False)


def generate_runtime_missing_arc_plot_with_bars(benchmarking_path):
    """Runtime vs missing-arc plot — variant that also renders bar groups.

    Use this when you want the per-threshold bar breakdown overlaid on top of
    the line/area chart produced by ``generate_runtime_missing_arc_plot``.
    Internally calls the shared implementation with ``with_bars=True``; the
    bar rendering blocks inside ``_generate_runtime_missing_arc_plot_impl``
    are gated on that flag.
    """
    return _generate_runtime_missing_arc_plot_impl(benchmarking_path, with_bars=True)


def generate_HGS_threshold_plot(benchmarking_path):
    """Plots per-instance value/runtime curves and per-threshold mean gap/runtime curves with std shading, plus per-threshold gap and runtime histograms."""
    performance_dict = {}
    better_runtime_dict = {}
    out_runtime_limit_dict = {}
    #     cost_dict = pkl.load(f)
    # iterate over subfolders inside the benchmarking directory
    for folder_name in os.listdir(benchmarking_path):
        folder_path = os.path.join(benchmarking_path, folder_name)
        if not os.path.isdir(folder_path):
            continue


        # iterate over files inside each folder
        count_instances = 0
        for file_name in os.listdir(folder_path):

            count_instances += 1

            if file_name not in performance_dict.keys():
                performance_dict[file_name] = []
            file_path = os.path.join(folder_path, file_name)


            with gzip.open(file_path, "rb") as f:
                result_dict = pkl.load(f)

            if result_dict["solver_value"] > 0 :
                performance_dict[file_name].append(
                        (float(folder_name),(result_dict["solver_value"],
                        result_dict["solver_runtime"],
                        result_dict["num_missing_arcs"], result_dict["exact_objective_value"]))
                )
            else:
                print(file_name, " ", (float(folder_name),(result_dict["solver_value"],
                    result_dict["solver_runtime"],
                    result_dict["num_missing_arcs"])))
                out_runtime_limit_dict[file_name] = 1
        print("count instances = ", count_instances)
    print("number of out_runtime_limit files : ", len(out_runtime_limit_dict.keys()))
    new_performance_dict = {}
    renew_performance_dict = {}
    better_runtime_count = 0
    better_runtime_bool  = False
    best_runtime_value  = 5001
    for key in performance_dict.keys():
        new_performance_dict[key] = []
        HGS_runtime = 1000
        for threshold in performance_dict[key]:
            new_performance_dict[key].append((threshold[0], ((threshold[1][0]- threshold[1][3])/threshold[1][3],
                                                    (threshold[1][1]-HGS_runtime)/HGS_runtime , threshold[1][2])))


    # Extract all unique parameter values (tuple[0])
    param_values = sorted({entry[0] for values in new_performance_dict.values() for entry in values})

    grouped_value = {p: [] for p in param_values}
    grouped_runtime = {p: [] for p in param_values}

    # Fill the grouped structures
    for instance, values in new_performance_dict.items():
        for p, triple in values:
            grouped_value[p].append(triple[0])     # tuple[1][0]
            grouped_runtime[p].append(triple[1])   # tuple[1][1]

    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Colors for consistency
    colors = plt.cm.tab10(range(len(param_values)))

    for idx, p in enumerate(param_values):

        x = list(range(len(grouped_value[p])))

        # Left axis: tuple[1][0]
        ax1.plot(
            x, grouped_value[p],
            color=colors[idx],
            marker='o',
            label=f"value p={p}"
        )

        # Right axis: tuple[1][1]
        ax2.plot(
            x, grouped_runtime[p],
            color=colors[idx],
            marker='x',
            linestyle='--',
            label=f"runtime p={p}"
        )

    ax1.set_xlabel("Instance index")
    ax1.set_ylabel("optimal relative gap")
    ax2.set_ylabel("runtime")

    ax1.set_title("Value and Runtime across instances for each threshold")

    # Build a combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.grid(True)
    plt.tight_layout()
    plt.show()


    # Extract all unique parameter values
    param_values = sorted({entry[0] for values in new_performance_dict.values() for entry in values})

    # Prepare accumulators
    value_acc = {p: [] for p in param_values}
    runtime_acc = {p: [] for p in param_values}

    # Fill accumulators
    for instance, values in new_performance_dict.items():
        for p, triple in values:
            value_acc[p].append(triple[0])     # tuple[1][0]
            runtime_acc[p].append(triple[1])   # tuple[1][1]

    # Compute averages and standard deviations
    avg_values = np.array([np.mean(value_acc[p]) for p in param_values])
    std_values = np.array([np.std(value_acc[p]) for p in param_values])


    avg_runtimes = []
    std_runtimes = []

    for p in param_values:
        arr = np.array(runtime_acc[p])

        # Compute 95th percentile threshold
        cutoff = np.percentile(arr, 95)

        trimmed = arr[arr <= cutoff]

        # Compute mean and std on trimmed data
        avg_runtimes.append(np.mean(trimmed))
        std_runtimes.append(np.std(trimmed))

    avg_runtimes = np.array(avg_runtimes)
    std_runtimes = np.array(std_runtimes)


    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Left axis: averaged values
    ax1.plot(param_values, avg_values, marker='o', color='tab:blue', label="Average complete relative gap")
    ax1.fill_between(param_values,
                     avg_values +std_values ,
                     avg_values - std_values,
                     color='tab:blue', alpha=0.2)

    # Right axis: averaged runtimes
    ax2.plot(param_values, avg_runtimes, marker='x', linestyle='--', color='tab:red', label="Mean complete relative runtime (Quantil 95%)")
    ax2.fill_between(param_values,
                     avg_runtimes +std_runtimes ,
                     avg_runtimes -std_runtimes,
                     color='tab:red', alpha=0.2)

    # X-axis: show only parameter values
    ax1.set_xticks(param_values)
    ax1.set_xticklabels([str(p) for p in param_values], rotation=45)

    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Average complete relative gap", color='tab:blue')
    ax2.set_ylabel("Mean(95%) complete relative runtime", color='tab:red')

    ax1.set_title("Average complete relative Gap and mean (quant 95%) relative Runtime per Threshold")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.grid(True)
    plt.tight_layout()
    plt.show()


    param_values = sorted({entry[0] for values in renew_performance_dict.values() for entry in values})

    # Prepare accumulators
    value_acc = {p: [] for p in param_values}
    runtime_acc = {p: [] for p in param_values}

    # Fill accumulators
    for instance, values in renew_performance_dict.items():
        for p, triple in values:
            value_acc[p].append(triple[0])     # tuple[1][0]
            runtime_acc[p].append(triple[1])   # tuple[1][1]

    # Compute averages and standard deviations
    avg_values = np.array([np.mean(value_acc[p]) for p in param_values])
    std_values = np.array([np.std(value_acc[p]) for p in param_values])


    avg_runtimes = []
    std_runtimes = []

    for p in param_values:
        arr = np.array(runtime_acc[p])

        # Compute 95th percentile threshold
        cutoff = np.percentile(arr, 95)

        trimmed = arr[arr <= cutoff]

        # Compute mean and std on trimmed data
        avg_runtimes.append(np.mean(trimmed))
        std_runtimes.append(np.std(trimmed))

    avg_runtimes = np.array(avg_runtimes)
    std_runtimes = np.array(std_runtimes)


    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Left axis: averaged values
    ax1.plot(param_values, avg_values, marker='o', color='tab:blue', label="Average complete relative gap")
    ax1.fill_between(param_values,
                     avg_values +std_values ,
                     avg_values - std_values,
                     color='tab:blue', alpha=0.2)

    # Right axis: averaged runtimes
    ax2.plot(param_values, avg_runtimes, marker='x', linestyle='--', color='tab:red', label="Median complete relative runtime")
    ax2.fill_between(param_values,
                     avg_runtimes +std_runtimes ,
                     avg_runtimes -std_runtimes,
                     color='tab:red', alpha=0.2)

    # X-axis: show only parameter values
    ax1.set_xticks(param_values)
    ax1.set_xticklabels([str(p) for p in param_values], rotation=45)

    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Average complete relative gap", color='tab:blue')
    ax2.set_ylabel("Mean complete relative runtime", color='tab:red')

    ax1.set_title("Average complete relative Gap and mean relative Runtime per Threshold")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.grid(True)
    plt.tight_layout()
    plt.show()

  # Extract all threshold values
    param_values = sorted({
        entry[0]
        for values in new_performance_dict.values()
        for entry in values
    })

    # Remove unwanted parameters
    excluded = {1, 0.266667}
    param_values = [p for p in param_values if p not in excluded]

    # Prepare accumulators
    value_acc = {p: [] for p in param_values}
    runtime_acc = {p: [] for p in param_values}

    # Fill accumulators
    for instance, values in new_performance_dict.items():
        for p, triple in values:
            if p in value_acc:
                value_acc[p].append(triple[0])
                runtime_acc[p].append(triple[1])

    # Create figure
    num_params = len(param_values)
    fig, axes = plt.subplots(2, num_params, figsize=(5 * num_params, 10))

    if num_params == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    # Loop over thresholds
    for i, p in enumerate(param_values):

        # GAP subplot
        ax_gap = axes[0, i]
        gaps = value_acc[p]
        ax_gap.hist(gaps, bins=20, color="tab:blue", alpha=0.7)
        ax_gap.set_title(f"Threshold {p} — Gap")

        ax_gap.text(
            0.98, 0.95,
            f"{len(gaps)} instances",
            transform=ax_gap.transAxes,
            ha="right", va="top",
            fontsize=10
        )

        # RUNTIME subplot
        ax_rt = axes[1, i]
        runtimes = np.array(runtime_acc[p])

        # Clip only the extreme positive tail
        low = np.min(runtimes)
        high = np.percentile(runtimes, 99)

        clipped = np.clip(runtimes, low, high)

        ax_rt.hist(clipped, bins=30, color="tab:red", alpha=0.7)
        ax_rt.set_xlim(low, high)

        ax_rt.set_xlabel("Relative Runtime")

        ax_rt.text(
            0.98, 0.95,
            f"{len(runtimes)} instances",
            transform=ax_rt.transAxes,
            ha="right", va="top",
            fontsize=10
        )

    plt.tight_layout()
    plt.show()

    # Extract all threshold values
    param_values = sorted({
        entry[0]
        for values in renew_performance_dict.values()
        for entry in values
    })

    # Remove unwanted parameters
    excluded = {1, 0.266667}
    param_values = [p for p in param_values if p not in excluded]

    # Prepare accumulators
    value_acc = {p: [] for p in param_values}
    runtime_acc = {p: [] for p in param_values}

    # Fill accumulators
    for instance, values in renew_performance_dict.items():
        for p, triple in values:
            if p in value_acc:
                value_acc[p].append(triple[0])
                runtime_acc[p].append(triple[1])

    # Create figure
    num_params = len(param_values)
    fig, axes = plt.subplots(2, num_params, figsize=(5 * num_params, 10))

    if num_params == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    # Loop over thresholds
    for i, p in enumerate(param_values):

        # GAP subplot
        ax_gap = axes[0, i]
        gaps = value_acc[p]
        ax_gap.hist(gaps, bins=20, color="tab:blue", alpha=0.7)
        ax_gap.set_title(f"Threshold {p} — Gap")

        ax_gap.text(
            0.98, 0.95,
            f"{len(gaps)} instances",
            transform=ax_gap.transAxes,
            ha="right", va="top",
            fontsize=10
        )

        # RUNTIME subplot
        ax_rt = axes[1, i]
        runtimes = np.array(runtime_acc[p])

        # Clip only the extreme positive tail
        low = np.min(runtimes)
        high = np.percentile(runtimes, 99)

        clipped = np.clip(runtimes, low, high)

        ax_rt.hist(clipped, bins=30, color="tab:red", alpha=0.7)
        ax_rt.set_xlim(low, high)

        ax_rt.set_xlabel("Runtime")

        ax_rt.text(
            0.98, 0.95,
            f"{len(runtimes)} instances",
            transform=ax_rt.transAxes,
            ha="right", va="top",
            fontsize=10
        )

    plt.tight_layout()
    plt.show()


    return performance_dict


def generate_HGS_threshold__missing_arc_model_comparaison_plot(benchmarking_path_1, benchmarking_path_2):
    """Compares two trained models by plotting per-threshold relative-value-difference and missing-edge-difference curves on a dual-axis chart."""
    performance_dict = {}
    better_runtime_dict = {}
    out_runtime_limit_dict = {}
    #     cost_dict = pkl.load(f)
    # iterate over subfolders inside the benchmarking directory
    for folder_name in os.listdir(benchmarking_path_1):
        folder_path = os.path.join(benchmarking_path_1, folder_name)
        if not os.path.isdir(folder_path):
            continue


        # iterate over files inside each folder
        count_instances = 0
        for file_name in os.listdir(folder_path):

            count_instances += 1

            if file_name not in performance_dict.keys():
                performance_dict[file_name] = []
            file_path = os.path.join(folder_path, file_name)


            with gzip.open(file_path, "rb") as f:
                result_dict = pkl.load(f)

            if result_dict["solver_value"] > 0 :
                performance_dict[file_name].append(
                        (float(folder_name),(result_dict["solver_value"],
                        result_dict["solver_runtime"],
                        result_dict["num_missing_arcs"], result_dict["exact_objective_value"]))
                )
            else:
                print(file_name, " ", (float(folder_name),(result_dict["solver_value"],
                    result_dict["solver_runtime"],
                    result_dict["num_missing_arcs"])))
                out_runtime_limit_dict[file_name] = 1
        print("count instances = ", count_instances)
    for folder_name in os.listdir(benchmarking_path_2):
        folder_path = os.path.join(benchmarking_path_2, folder_name)
        if not os.path.isdir(folder_path):
            continue


        # iterate over files inside each folder
        count_instances = 0
        for file_name in os.listdir(folder_path):

            count_instances += 1

            if file_name not in performance_dict.keys():
                performance_dict[file_name] = []
                print("filename : ", file_name)
            file_path = os.path.join(folder_path, file_name)


            with gzip.open(file_path, "rb") as f:
                result_dict = pkl.load(f)

            if result_dict["solver_value"] > 0 :
                performance_dict[file_name].append(
                        (float(folder_name),(result_dict["solver_value"],
                        result_dict["solver_runtime"],
                        result_dict["num_missing_arcs"], result_dict["exact_objective_value"]))
                )
            else:
                print(file_name, " ", (float(folder_name),(result_dict["solver_value"],
                    result_dict["solver_runtime"],
                    result_dict["num_missing_arcs"])))
                out_runtime_limit_dict[file_name] = 1
        print("count instances = ", count_instances)

    from collections import defaultdict

    # performances_dict is your full dictionary

    # ---------------------------------------------------------
    # 1. Group data by threshold across all instances
    # ---------------------------------------------------------

    # Structure:
    grouped = defaultdict(list)

    for instance, entries in performance_dict.items():
        # Temporary grouping per instance
        per_thr = defaultdict(list)

        for thr, data in entries:
            per_thr[thr].append(data)

        # Each threshold must have exactly 2 entries
        for thr, lst in per_thr.items():
            if len(lst) != 2:
                print(f"Warning: instance {instance}, threshold {thr} has {len(lst)} entries")
                continue

            # Store the pair for global aggregation
            grouped[thr].append((lst[0], lst[1]))

    # ---------------------------------------------------------
    # 2. Compute differences per threshold
    # ---------------------------------------------------------
    thresholds = sorted(grouped.keys())

    value_diff = []
    missing_arc_diff = []

    for thr in thresholds:
        diffs_value = []
        diffs_missing = []

        for g1, g2 in grouped[thr]:

            # You asked: "difference with the second coordinate of the second tuple"
            # → second coordinate = runtime = index 1
            diffs_value.append((g1[0] - g2[0])/g1[0])

            # Third coordinate = missing_arc = index 2
            diffs_missing.append(g1[2] - g2[2])

        # Average across all instances
        print("diffs_mising : ", diffs_missing)
        value_diff.append(np.mean(diffs_value))
        missing_arc_diff.append(np.mean(diffs_missing))

    value_diff = np.array(value_diff)
    missing_arc_diff = np.array(missing_arc_diff)

    # ---------------------------------------------------------
    # 3. Plot both curves with two y-axes
    # ---------------------------------------------------------

    fig, ax1 = plt.subplots(figsize=(10, 6))

        # First y-axis: value_diff
    # Convert thresholds to sorted list
    thresholds = sorted(grouped.keys())

    # Create equally spaced x positions
    x = np.arange(len(thresholds))

    # Plot using x instead of thresholds
    ax1.plot(x, value_diff, marker='o', color='tab:blue',
            label="Relative value difference")

    # Second axis
    ax2 = ax1.twinx()
    ax2.plot(x, missing_arc_diff, marker='x', linestyle='--',
            color='tab:red', label="Missing edge difference")

    # Set x-ticks to equal spacing but label them with thresholds
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(t) for t in thresholds])


    plt.title("Differences between big-trained and small-trained per threshold over big instances")
    ax1.grid(True)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.show()


def generate_runtime_knn_plot(benchmarking_path, benchmarking_path_2):
    """For each (k-NN, GNN-threshold) pair, plots the mean and median relative-value gap and relative-runtime gap with shaded std bands on a dual-axis chart."""
    performance_dict = {}
    better_runtime_dict = {}
    out_runtime_limit_dict = {}
    with open("cost_dict.pkl", "rb") as f:
        cost_dict = pkl.load(f)
    # iterate over subfolders inside the benchmarking directory
    for folder_name in os.listdir(benchmarking_path):
        folder_path = os.path.join(benchmarking_path, folder_name)
        if not os.path.isdir(folder_path):
            continue


        # iterate over files inside each folder
        count_instances = 0
        for file_name in os.listdir(folder_path):

            count_instances += 1

            if file_name not in performance_dict.keys():
                performance_dict[file_name] = []
            file_path = os.path.join(folder_path, file_name)


            with gzip.open(file_path, "rb") as f:
                result_dict = pkl.load(f)

            if result_dict["solver_value"] > 0 :
                if folder_name != "HGS" and folder_name != "HGS_80" :
                    performance_dict[file_name].append(
                        (float(folder_name),(result_dict["solver_value"],
                        result_dict["runtime"]-9,
                        result_dict["num_missing_arcs"]))
                    )
                else :
                    if folder_name =="HGS":
                        performance_dict[file_name].append(
                            (0.9,(result_dict["solver_value"],
                            result_dict["solver_runtime"],
                            result_dict["num_missing_arcs"]))
                        )
                    else :
                        performance_dict[file_name].append(
                            (0.8,(result_dict["solver_value"],
                            result_dict["solver_runtime"],
                            result_dict["num_missing_arcs"]))
                        )
            else:
                print(file_name, " ", (float(folder_name),(result_dict["solver_value"],
                    result_dict["solver_runtime"],
                    result_dict["num_missing_arcs"])))
                if float(folder_name) == 1.0:
                    out_runtime_limit_dict[file_name] = 1
        print("count instances = ", count_instances)
    print("number of out_runtime_limit files : ", len(out_runtime_limit_dict.keys()))

    for folder_name in os.listdir(benchmarking_path_2):
        folder_path = os.path.join(benchmarking_path_2, folder_name)
        if not os.path.isdir(folder_path):
            continue


        # iterate over files inside each folder
        count_instances = 0
        for file_name in os.listdir(folder_path):

            count_instances += 1

            if file_name not in performance_dict.keys():
                continue
            if performance_dict[file_name]==[]:
                continue
            file_path = os.path.join(folder_path, file_name)


            with gzip.open(file_path, "rb") as f:
                result_dict = pkl.load(f)

            if result_dict["solver_value"] > 0 :
                if folder_name != "HGS" and folder_name != "HGS_80" :
                    if folder_name == 1:
                        performance_dict[file_name].append(
                            (float(folder_name),(result_dict["solver_value"],
                            result_dict["solver_runtime"],
                            result_dict["num_missing_arcs"]))
                        )
                    else:
                        performance_dict[file_name].append(
                            (float(folder_name),(result_dict["solver_value"],
                            result_dict["runtime"]-9,
                            result_dict["num_missing_arcs"]))
                        )
                # else :

                    #     performance_dict[file_name].append(
                    #         (0.9,(result_dict["solver_value"],
                    #         result_dict["solver_runtime"],
                    #         result_dict["num_missing_arcs"]))
                    #     )
                    # else :
                    #     performance_dict[file_name].append(
                    #         (0.8,(result_dict["solver_value"],
                    #         result_dict["solver_runtime"],
                    #         result_dict["num_missing_arcs"]))
                    #     )
            else:
                print(file_name, " ", (float(folder_name),(result_dict["solver_value"],
                    result_dict["solver_runtime"],
                    result_dict["num_missing_arcs"])))
                if float(folder_name) == 1.0:
                    out_runtime_limit_dict[file_name] = 1
        print("count instances = ", count_instances)
    print("number of out_runtime_limit files : ", len(out_runtime_limit_dict.keys()))
    new_performance_dict = {}
    renew_performance_dict = {}
    better_runtime_count = 0
    better_runtime_bool  = False
    best_runtime_value  = 5001
    for key in performance_dict.keys():
        match = re.match(r"sol_instance_(\d+_\d+)\.pkl\.gz", key)
        optimal_value = cost_dict[match.group(1)]
        optimal_runtime= 5000
        new_performance_dict[key] = []
        renew_performance_dict[key] = []
        found_threshold = False
        HGS_runtime = 1000
        for threshold in performance_dict[key]:
            if threshold[0] == 1.0:
                found_threshold  = True
                HGS_value = threshold[1][0]
                optimal_runtime = threshold[1][1]
                HGS_runtime =  80
                break
        for threshold in performance_dict[key]:
            new_performance_dict[key].append((threshold[0], ((threshold[1][0]- optimal_value)*100/optimal_value,
                                                    (threshold[1][1]-optimal_runtime)*100/optimal_runtime , threshold[1][2])))
        if not found_threshold :
            #         if not better_runtime_bool:
            #                 best_runtime_value = threshold[1][1]
            #             better_runtime_count +=1
            #             better_runtime_bool = True
            #             better_runtime_threshold = threshold[0]
            for threshold in performance_dict[key]:
                if threshold[0] != 1:
                    renew_performance_dict[key].append((threshold[0], ((threshold[1][0]- optimal_value)/optimal_value,
                                                    threshold[1][1] , threshold[1][2])))
        better_runtime_bool = False

    #     (30, 0.33333),
    #     (20, 0.266667),
    #     (16, 0.2),
    #     (4, 0.05),
    #     (2, 0.02),
    #     (8, 0.1),
    # ]

    #     lookup = {thr: vals for thr, vals in lst}

    #     result_pairs = {}

    #     for big, small in pairs:
    #                 big = 0.9
    #         if big in lookup and small in lookup:

    #             t1 = lookup[float(big)]
    #             t2 = lookup[small]

    #             diff = (t2[0] - t1[0], t2[1] - t1[1], t2[2] - t1[2])

    #                 big = 1

    #             result_pairs[(big, small)] = diff

    #     new_dict[key] = result_pairs


    #     values = []

    #     for key in new_dict:
    #             values.append(new_dict[key][pair])

    #     arr = np.array(values, dtype=float)
    #     avg_results[pair] = arr.mean(axis=0)


    #     entry[0]
    #     for values in renew_performance_dict.values()
    #     for entry in values
    # })


    #     for p, triple in values:
    #         if p in value_acc:
    #             value_acc[p].append(triple[0])
    #             runtime_acc[p].append(triple[1])


    #     axes = np.array([[axes[0]], [axes[1]]])


    #     ax_gap = axes[0, i]
    #     gaps = value_acc[p]
    #     ax_gap.hist(gaps, bins=20, color="tab:blue", alpha=0.7)
    #     ax_gap.set_title(f"Threshold {p} — Gap")

    #     ax_gap.text(
    #         0.98, 0.95,
    #         f"{len(gaps)} instances",
    #         transform=ax_gap.transAxes,
    #         ha="right", va="top",
    #         fontsize=10
    #     )

    #     ax_rt = axes[1, i]
    #     runtimes = np.array(runtime_acc[p])

    #     low = np.min(runtimes)
    #     high = np.percentile(runtimes, 99)

    #     clipped = np.clip(runtimes, low, high)

    #     ax_rt.hist(clipped, bins=30, color="tab:red", alpha=0.7)
    #     ax_rt.set_xlim(low, high)

    #     ax_rt.set_xlabel("Runtime")

    #     ax_rt.text(
    #         0.98, 0.95,
    #         f"{len(runtimes)} instances",
    #         transform=ax_rt.transAxes,
    #         ha="right", va="top",
    #         fontsize=10
    #     )


    pairs = [
        (2, 0.02),
        (4, 0.05),
        (16, 0.2),
        (20, 0.266667),
        (30, 0.33333),
    ]

    #     "big":   [(gap, runtime), ...],
    #     "small": [(gap, runtime), ...]
    # }
    new_dict = {}

    for key, lst in new_performance_dict.items():
        lookup = {thr: vals for thr, vals in lst}  # thr → (gap, runtime, missing)

        pair_dict = {}

        for big, small in pairs:
            # HGS special case
            big_key = 0.9 if big == 1 else big

            if big_key in lookup and small in lookup:
                gap_big, rt_big, _ = lookup[big_key]
                gap_small, rt_small, _ = lookup[small]

                pair_dict[(big, small)] = {
                    "big":   (gap_big, rt_big),
                    "small": (gap_small, rt_small),
                }

        new_dict[key] = pair_dict
        big_gap_values = {pair: [] for pair in pairs}
        big_rt_values  = {pair: [] for pair in pairs}
        small_gap_values = {pair: [] for pair in pairs}
        small_rt_values  = {pair: [] for pair in pairs}

        for key, pair_dict in new_dict.items():
            for pair in pairs:
                if pair in pair_dict:
                    gap_big, rt_big = pair_dict[pair]["big"]
                    gap_small, rt_small = pair_dict[pair]["small"]

                    big_gap_values[pair].append(gap_big)
                    big_rt_values[pair].append(rt_big)
                    small_gap_values[pair].append(gap_small)
                    small_rt_values[pair].append(rt_small)

    def trimmed_stats(values, quantile):
        """Mean and std after trimming the top tail above the given percentile."""
        values = np.array(values, dtype=float)
        if len(values) == 0:
            return np.nan, np.nan
        cutoff = np.percentile(values, quantile)
        trimmed = values[values <= cutoff]
        return trimmed.mean(), trimmed.std()

    big_gap_mean, big_gap_std = [], []
    big_rt_mean, big_rt_std = [], []
    small_gap_mean, small_gap_std = [], []
    small_rt_mean, small_rt_std = [], []

    for pair in pairs:
        # big coordinate
        m, s = trimmed_stats(big_gap_values[pair],100)
        big_gap_mean.append(m)
        big_gap_std.append(s)

        m, s = trimmed_stats(big_rt_values[pair], 99)
        big_rt_mean.append(m)
        big_rt_std.append(s)

        # small coordinate
        m, s = trimmed_stats(small_gap_values[pair], 100)
        small_gap_mean.append(m)
        small_gap_std.append(s)

        m, s = trimmed_stats(small_rt_values[pair],99)
        small_rt_mean.append(m)
        small_rt_std.append(s)

    # ---- MEDIANS ----
    big_gap_median   = []
    big_rt_median    = []
    small_gap_median = []
    small_rt_median  = []

    for pair in pairs:
        # big coordinate
        big_gap_median.append(np.median(big_gap_values[pair]) if len(big_gap_values[pair]) else np.nan)
        big_rt_median.append(np.median(big_rt_values[pair]) if len(big_rt_values[pair]) else np.nan)

        # small coordinate
        small_gap_median.append(np.median(small_gap_values[pair]) if len(small_gap_values[pair]) else np.nan)
        small_rt_median.append(np.median(small_rt_values[pair]) if len(small_rt_values[pair]) else np.nan)


    plt.rcParams.update({
        "font.size": 20,          # base font size
        "axes.titlesize": 22,     # title
        "axes.labelsize": 20,     # x/y labels
        "xtick.labelsize": 18,    # x tick labels
        "ytick.labelsize": 18,    # y tick labels
        "legend.fontsize": 14     # legend
    })

    x_labels = [f"({p[0]}, {p[1]})" for p in pairs]
    x_pos = np.arange(len(pairs))

    fig, ax1 = plt.subplots(figsize=(14, 8))


    # -------------------------
    # LEFT AXIS — VALUE GAPS
    # -------------------------
    ax1.set_ylabel("Average RVG %")

    # First coordinate (big)
    ax1.plot(x_pos, big_gap_mean, marker="o", color="tab:blue", label="RVG (k-NN)")
    ax1.fill_between(
        x_pos,
        np.array(big_gap_mean) + np.array(big_gap_std),
        np.array(big_gap_mean) - np.array(big_gap_std),
        color="tab:blue",
        alpha=0.2
    )

    # Second coordinate (small)
    ax1.plot(x_pos, small_gap_mean, marker="s", color="tab:green", label="RVG (GNN)")
    ax1.fill_between(
        x_pos,
        np.array(small_gap_mean) + np.array(small_gap_std),
        np.array(small_gap_mean) - np.array(small_gap_std),
        color="tab:green",
        alpha=0.2
    )

    # Median value gap (big)
    ax1.plot(
        x_pos, big_gap_median,
        linestyle="--", linewidth=2,
        color="navy",
        label="Median RVG k-NN"
    )

    # Median value gap (small)
    ax1.plot(
        x_pos, small_gap_median,
        linestyle="--", linewidth=2,
        color="darkgreen",
        label="Median RVG GNN"
    )


    # -------------------------
    # RIGHT AXIS — RUNTIME GAPS
    # -------------------------
    ax2 = ax1.twinx()
    ax2.set_ylabel("Average RRG % (Quantile 99%)")

    # First coordinate (big)
    ax2.plot(x_pos, big_rt_mean, marker="^", color="tab:red", label="RRG % k-NN (Quantile 99%)")
    ax2.fill_between(
        x_pos,
        np.array(big_rt_mean) + np.array(big_rt_std),
        np.array(big_rt_mean) - np.array(big_rt_std),
        color="tab:red",
        alpha=0.2
    )

    # Second coordinate (small)
    ax2.plot(x_pos, small_rt_mean, marker="v", color="tab:purple", label="RRG % GNN (Quantile 99%)")
    ax2.fill_between(
        x_pos,
        np.array(small_rt_mean) + np.array(small_rt_std),
        np.array(small_rt_mean) - np.array(small_rt_std),
        color="tab:purple",
        alpha=0.2
    )

    # Median runtime gap (big)
    ax2.plot(
        x_pos, big_rt_median,
        linestyle=":", linewidth=2,
        color="darkred",
        label="Median RRG (k-NN)"
    )

    # Median runtime gap (small)
    ax2.plot(
        x_pos, small_rt_median,
        linestyle=":", linewidth=2,
        color="indigo",
        label="Median RRG (GNN)"
    )


    # -------------------------
    # X-axis formatting
    # -------------------------
    plt.xticks(x_pos, x_labels, rotation=45, ha="right")
    ax1.set_title("Average RVG and RVG % per threshold pair")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # -------------------------
    # Combined legend
    # -------------------------
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    all_lines  = lines1 + lines2
    all_labels = labels1 + labels2

    # Split in two halves
    mid = len(all_lines) // 2

    legend1 = ax1.legend(all_lines[:mid], all_labels[:mid],
                        loc="upper left", bbox_to_anchor=(0.0, 1.0))
    legend2 = ax1.legend(all_lines[mid:], all_labels[mid:],
                        loc="upper left", bbox_to_anchor=(0.245, 1.0))

    ax1.add_artist(legend1)
    ax1.add_artist(legend2)


    plt.tight_layout()
    plt.show()


    return performance_dict


def generate_new_HGS_reduction_histogram_plot(benchmarking_path_list, save_path):
    """For each HGS runtime, draws stacked histograms (one subplot per threshold) of the relative optimality gap distribution across instances."""

    HGS_time_limit_dict = {}

    for benchmarking_path in benchmarking_path_list:

        # Extract HGS runtime from the path (last integer)
        match = re.findall(r"(\d+)$", benchmarking_path)
        if not match:
            raise ValueError(f"Cannot extract HGS runtime from path: {benchmarking_path}")
        HGS_runtime = int(match[0])

        # threshold → list of gaps
        threshold_gap_dict = {}

        # Loop over folders inside the benchmarking path
        for folder_name in os.listdir(benchmarking_path):
            folder_path = os.path.join(benchmarking_path, folder_name)
            if not os.path.isdir(folder_path):
                continue

            threshold = float(folder_name)

            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                with gzip.open(file_path, "rb") as f:
                    result_dict = pkl.load(f)

                if result_dict["solver_value"] <= 0:
                    continue

                # Compute relative optimal gap
                rel_gap = (
                    result_dict["solver_value"] - result_dict["exact_objective_value"]
                ) / result_dict["exact_objective_value"]

                if threshold not in threshold_gap_dict:
                    threshold_gap_dict[threshold] = []

                threshold_gap_dict[threshold].append(rel_gap)

        # Store raw gaps under the HGS runtime key
        HGS_time_limit_dict[HGS_runtime] = threshold_gap_dict

        # -------------------------
        # PLOT HISTOGRAMS PER THRESHOLD
        # -------------------------
        thresholds = sorted(threshold_gap_dict.keys())
        n = len(thresholds)

        # Create subplots: one per threshold
        fig, axes = plt.subplots(
            nrows=n,
            ncols=1,
            figsize=(8, 3 * n),
            constrained_layout=True
        )

        # If only one threshold, axes is not a list
        if n == 1:
            axes = [axes]

        for ax, thr in zip(axes, thresholds):
            gaps = threshold_gap_dict[thr]

            ax.hist(gaps, bins=20, alpha=0.7, color="steelblue", edgecolor="black")
            ax.set_title(f"Threshold = {thr}")
            ax.set_xlabel("Relative Optimal Gap")
            ax.set_ylabel("Count")
            ax.grid(True, linestyle="--", alpha=0.4)

        fig.suptitle(f"HGS_time_limit_{HGS_runtime}", fontsize=16)
        plt.show()

    # Save the raw dictionary
    print("Saving aggregated results to:", save_path)
    with open(save_path, "wb") as f:
        pkl.dump(HGS_time_limit_dict, f)

    return HGS_time_limit_dict


def plot_HGS_reduction(HGS_time_limit_dict_path, title="Average Relative Optimal Gap vs Threshold"):
    """Loads aggregated HGS-runtime dict and plots one curve per HGS time-limit of average relative optimality gap as a function of reduction threshold."""

    with open(HGS_time_limit_dict_path, "rb") as f:
        HGS_time_limit_dict = pkl.load(f)
    plt.figure(figsize=(10, 6))

    all_thresholds = sorted({
        thr for d in HGS_time_limit_dict.values() for thr in d.keys()
    })

    # Sort HGS runtimes for consistent plotting order
    for HGS_runtime in sorted(HGS_time_limit_dict.keys()):
        if HGS_runtime != 10:
            threshold_gap_dict = HGS_time_limit_dict[HGS_runtime]

            # Sort thresholds on x-axis
            thresholds = sorted(threshold_gap_dict.keys())
            gaps = [threshold_gap_dict[t] for t in thresholds]

            plt.plot(
                thresholds,
                gaps,
                marker="o",
                linewidth=2,
                label=f"HGS runtime = {HGS_runtime}s"
            )
    plt.xticks(all_thresholds, rotation=45)
    plt.xlabel("Threshold")
    plt.ylabel("Average Relative Optimal Gap")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_HGS_threshold_vs_gap(dict_path):
    """For each HGS runtime, plots average relative gap versus threshold (one curve per graph size) and highlights each curve's minimum."""
    with open(dict_path, "rb") as f:
        HGS_time_limit_dict = pkl.load(f)

    graph_size_dict = {2001000 : 2000, 281625 : 750, 125250 : 500, 500500 : 1000, 1125750: 1500 }

    for HGS_runtime, threshold_dict in HGS_time_limit_dict.items():

        thresholds = sorted(threshold_dict.keys())

        # Create categorical x positions
        x_positions = list(range(len(thresholds)))

        # Build tick labels (replace 1 → PyVRP)
        tick_labels = ["PyVRP" if t == 1 else str(t) for t in thresholds]

        all_graph_sizes = sorted({
            gs for thr in thresholds for gs in threshold_dict[thr].keys()
        })

        plt.figure(figsize=(10, 6))

        for graph_size in all_graph_sizes:
            y_values = []

            for thr in thresholds:
                if graph_size in threshold_dict[thr]:
                    gaps = threshold_dict[thr][graph_size]
                    avg_gap = sum(gaps) / len(gaps)
                else:
                    avg_gap = None
                y_values.append(avg_gap)

            if not any(v is not None for v in y_values):
                continue

            # Plot using categorical x positions
            plt.plot(
                x_positions,
                y_values,
                marker="o",
                linewidth=2,
                label=f"Nb Customers = {graph_size_dict[graph_size]}"
            )

            # ---- Highlight minimum ----
            valid_indices = [i for i, v in enumerate(y_values) if v is not None]
            valid_values = [y_values[i] for i in valid_indices]

            min_idx_local = valid_indices[np.argmin(valid_values)]
            min_x = x_positions[min_idx_local]
            min_val = y_values[min_idx_local]

            plt.scatter(min_x, min_val, color="red", s=80, zorder=5)

            plt.text(
                min_x,
                min_val,
                f"{min_val:.3g}",
                fontsize=9,
                ha="left",
                va="bottom"
            )

        # Apply categorical ticks
        plt.xticks(x_positions, tick_labels, rotation=45)

        plt.xlabel("Threshold")
        plt.ylabel("Average Relative Gap")
        plt.title(f"HGS_runtime_{HGS_runtime}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()


def generate_new_HGS_3_reduction_plot(benchmarking_path_list, save_path):
    """Plots a histogram of signed solver-value differences between two thresholds across instances using a symmetric-log x-axis with log-spaced bins."""
    import re
    import gzip
    import pickle as pkl
    import os

    HGS_time_limit_dict_sol = {}

    for benchmarking_path in benchmarking_path_list:

        # Extract HGS runtime from the path (last integer)
        match = re.findall(r"(\d+)$", benchmarking_path)
        if not match:
            raise ValueError(f"Cannot extract HGS runtime from path: {benchmarking_path}")
        HGS_runtime = int(match[0])
        print(match)


        # Loop over threshold folders
        for folder_name in os.listdir(benchmarking_path):
            folder_path = os.path.join(benchmarking_path, folder_name)
            if not os.path.isdir(folder_path):
                continue

            threshold = float(folder_name)

            if threshold != 1 and threshold != 0.99 :
                continue


            # Loop over instance files
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                with gzip.open(file_path, "rb") as f:
                    result_dict = pkl.load(f)

                # Skip invalid runs
                if result_dict["solver_value"] <= 0:
                    continue

                if (file_name, len(result_dict["solution"])) not in HGS_time_limit_dict_sol.keys():
                    HGS_time_limit_dict_sol[(file_name,len(result_dict["solution"]))] = [result_dict["solver_value"]]
                else :
                    HGS_time_limit_dict_sol[(file_name,len(result_dict["solution"]))].append(result_dict["solver_value"])


    # Save the full structure
    new_new_HGS_dict = {}
    for key in HGS_time_limit_dict_sol.keys():
        if len(HGS_time_limit_dict_sol[key]) >1 :
            new_new_HGS_dict[key] = HGS_time_limit_dict_sol[key][1] - HGS_time_limit_dict_sol[key][0]
    print("Saving aggregated results to:", save_path)
    print(HGS_time_limit_dict_sol)
    print(new_new_HGS_dict)

    values = np.array(list(new_new_HGS_dict.values()))
    print("mean = ", np.mean(values))

    # Split positive and negative values
    pos = values[values > 0]
    neg = -values[values < 0]   # take absolute for binning

    # Build logarithmic bins for magnitudes
    all_magnitudes = np.concatenate([pos, neg])
    min_val = all_magnitudes[all_magnitudes > 0].min()
    max_val = all_magnitudes.max()

    # Log-spaced bins for magnitudes
    mag_bins = np.logspace(np.log10(min_val), np.log10(max_val), 40)

    # Convert magnitude bins into signed bins
    bins = np.concatenate([-mag_bins[::-1], mag_bins])

    plt.figure(figsize=(12, 6))
    plt.hist(values, bins=bins, color="steelblue", alpha=0.8)

    # Logarithmic x-axis that supports negative values
    plt.xscale("symlog", linthresh=10)

    plt.xlabel("Value (symlog scale)")
    plt.ylabel("Frequency")
    plt.title("Histogram of signed values with logarithmic x-axis")

    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    return HGS_time_limit_dict_sol


def plt_new_HGS_4(dict_path):
    """Plots threshold-vs-average-relative-cost-gap and threshold-vs-average-relative-runtime-gap curves per graph size with minima highlighted, then runtime-vs-cost trajectories per graph size."""

    with open(dict_path, "rb") as f:
        HGS_time_limit_dict_sol = pkl.load(f)

    new_relative_HGS_dict = {}

    graph_size_dict = {2001000 : 2000, 281625 : 750, 125250 : 500, 500500 : 1000, 1125750: 1500 }

    print(HGS_time_limit_dict_sol)

    complete_value = 99999999
    complete_time =  0
    for key in HGS_time_limit_dict_sol.keys():
        for i in range(len(HGS_time_limit_dict_sol[key])):
            d = HGS_time_limit_dict_sol[key][i][1]

            only_key = next(iter(d))
            if only_key == 1:
                filtered_list = [t for t in HGS_time_limit_dict_sol[key][i][1][1] if int(t[1].rstrip("s")) < 1000]
                complete_value = filtered_list[-1][2]
                complete_time = int(filtered_list[-1][1].rstrip("s"))
                break
        new_relative_HGS_dict[key] = []
        for i in range(len(HGS_time_limit_dict_sol[key])):

            vals = list(HGS_time_limit_dict_sol[key][i][1].values())
            only_key = next(iter(HGS_time_limit_dict_sol[key][i][1]))

            # since your dict has only one value, extract it
            only_list = vals[0]

            only_list = [t for t in only_list if int(t[1].rstrip("s")) < 1000]


            # take the last tuple inside that list
            last_iter,last_time, last_cost = only_list[-1]

            for j in range(len(only_list)):
                if only_list[j][2]< complete_value and only_key != 1:
                    last_time = only_list[j][1]
                    break
            if only_key ==1 :
                last_cost = complete_value
                last_time = str(complete_time) + "s"
            new_relative_HGS_dict[key].append((only_key, HGS_time_limit_dict_sol[key][i][0], last_iter,  (int(last_time.rstrip("s"))-complete_time)/complete_time,  complete_time, (last_cost-complete_value)/complete_value))


    # Step 1 — reorganize data by graph size
    grouped = defaultdict(lambda: defaultdict(list))

    for key, tuples in new_relative_HGS_dict.items():
        for (thr, gsize, _, _, _, gap) in tuples:
            grouped[gsize][thr].append(gap)

    # Step 2 — define the thresholds you want to show
    special_thresholds = [ 0.01, 0.0125, 0.015,0.0175, 0.02, 0.035,0.05,0.1, 1]

    # Step 3 — x positions and labels
    x_positions = np.arange(len(special_thresholds))
    x_labels = ["0.01", "0.0125", "0.015", "0.0175", "0.02","0.035","0.05", "0.1", "PyVRP"]

    plt.figure(figsize=(12, 7))

    # Step 4 — plot one curve per graph size
    for gsize, thr_dict in grouped.items():
        y_values = []

        for thr in special_thresholds:
            if thr in thr_dict:
                avg_gap = sum(thr_dict[thr]) / len(thr_dict[thr])
            else:
                avg_gap = None
            y_values.append(avg_gap)

        # Skip if no valid data
        if not any(v is not None for v in y_values):
            continue

        plt.plot(
            x_positions,
            y_values,
            marker="o",
            linewidth=2,
            label=f"Graph size = {graph_size_dict[gsize]}"
        )

        # Highlight minimum
        valid_indices = [i for i, v in enumerate(y_values) if v is not None]
        valid_values = [y_values[i] for i in valid_indices]

        min_idx = valid_indices[valid_values.index(min(valid_values))]
        min_x = x_positions[min_idx]
        min_y = y_values[min_idx]

        plt.scatter(min_x, min_y, color="red", s=90, zorder=5)
        plt.text(min_x, min_y, f"{min_y:.3g}", fontsize=9, ha="left", va="bottom")

    # Step 5 — format x-axis
    plt.xticks(x_positions, x_labels, rotation=45)

    plt.xlabel("Threshold")
    plt.ylabel("Average Relative Gap")
    plt.title("Threshold vs Average Relative Cost Gap (per graph size) 1000s")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


    grouped = defaultdict(lambda: defaultdict(list))

    for key, tuples in new_relative_HGS_dict.items():
        for (thr, gsize, _, runtime_gap, _, _) in tuples:
            grouped[gsize][thr].append(runtime_gap)

    # Step 2 — collect all thresholds in sorted order
    all_thresholds = sorted({thr for g in grouped.values() for thr in g.keys()})

    # Step 3 — categorical x positions
    x_positions = list(range(len(all_thresholds)))
    x_labels = [str(t) for t in all_thresholds]

    plt.figure(figsize=(12, 7))

    # Step 4 — plot one curve per graph size
    for gsize, thr_dict in grouped.items():
        y_values = []

        for thr in all_thresholds:
            if thr in thr_dict:
                avg_runtime_gap = sum(thr_dict[thr]) / len(thr_dict[thr])
            else:
                avg_runtime_gap = None
            y_values.append(avg_runtime_gap)

        if not any(v is not None for v in y_values):
            continue

        plt.plot(
            x_positions,
            y_values,
            marker="o",
            linewidth=2,
            label=f"Graph size = {graph_size_dict[gsize]}"
        )

        # Step 5 — highlight minimum
        valid_indices = [i for i, v in enumerate(y_values) if v is not None]
        valid_values = [y_values[i] for i in valid_indices]

        min_idx = valid_indices[valid_values.index(min(valid_values))]
        min_x = x_positions[min_idx]
        min_y = y_values[min_idx]

        plt.scatter(min_x, min_y, color="red", s=90, zorder=5)
        plt.text(min_x, min_y, f"{min_y:.3g}", fontsize=9, ha="left", va="bottom")

    # Step 6 — format x-axis
    plt.xticks(x_positions, x_labels, rotation=45)

    plt.xlabel("Threshold")
    plt.ylabel("Average Relative Runtime Gap")
    plt.title("Threshold vs Average Relative Runtime Gap (per graph size)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


    # graph_size → threshold → list of (runtime, cost)
    grouped = defaultdict(lambda: defaultdict(list))

    for key in HGS_time_limit_dict_sol.keys():
        for entry in HGS_time_limit_dict_sol[key]:
            gsize = entry[0]
            thr_dict = entry[1]

            for thr, traj in thr_dict.items():
                # traj is a list of (iter, "123s", cost)
                cleaned = []
                for (it, t_str, cost) in traj:
                    runtime = int(t_str.rstrip("s"))
                    cleaned.append((runtime, cost))

                grouped[gsize][thr] = cleaned
    for gsize, thr_dict in grouped.items():

        plt.figure(figsize=(10, 6))

        for thr, traj in thr_dict.items():
            # Extract runtime and cost
            runtimes = [t for (t, c) in traj]
            costs = [c for (t, c) in traj]

            plt.plot(runtimes, costs, marker="o", label=f"thr={thr}")

        plt.xlabel("Runtime (s)")
        plt.ylabel("Solution value")
        plt.title(f"Runtime vs Cost — Graph size {graph_size_dict[gsize]}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()


def plt_new_HGS_77(dict_path):
    """Aggregates per-(neighbourhood-size, k-NN-restriction) relative-iter / relative-runtime / relative-value metrics and plots their mean +/- std on a dual-axis errorbar chart, highlighting negative points."""

    with open(dict_path, "rb") as f:
        HGS_time_limit_dict_sol = pkl.load(f)

    print(HGS_time_limit_dict_sol)
    new_relative_HGS_dict = {}

    graph_size_dict = {2001000 : 2000, 281625 : 750, 125250 : 500, 500500 : 1000, 1125750: 1500 }

    for key in HGS_time_limit_dict_sol.keys():
        for key_2 in HGS_time_limit_dict_sol[key]:
            if key_2 == (1,1):
                filtered_list = [t for t in HGS_time_limit_dict_sol[key][key_2][1] if int(t[1].rstrip("s")) < 2000]
                complete_value = filtered_list[-1][2]
                complete_time = int(filtered_list[-1][1].rstrip("s"))
                complete_iteration = filtered_list[-1][0]
                new_relative_HGS_dict[key] = [(complete_iteration, complete_time, complete_value), []]
                break
        longest_iter = new_relative_HGS_dict[key][0][0]
        longest_time = new_relative_HGS_dict[key][0][1]
        longest_value = new_relative_HGS_dict[key][0][2]
        for key_2 in HGS_time_limit_dict_sol[key]:
            filtered_list = [t for t in HGS_time_limit_dict_sol[key][key_2][1] if int(t[1].rstrip("s")) < 2000]
            if(len(filtered_list)==0):
                print(key_2)
                print(key)
                continue
            complete_value = filtered_list[-1][2]
            complete_time = int(filtered_list[-1][1].rstrip("s"))
            for t in range(len(filtered_list)):
                if filtered_list[t][2] < longest_value:
                    complete_value = filtered_list[t][2]
                    complete_time = int(filtered_list[t][1].rstrip("s"))

            complete_iteration = filtered_list[-1][0]

            new_relative_HGS_dict[key][1].append({key_2 : ((complete_iteration-longest_iter)/longest_iter,
                                                 (complete_time-longest_time)/longest_time,
                                                 (complete_value-longest_value)/longest_value)})

    # ---------------------------------------------------------
    # 1. Aggregate triples by subkey
    # ---------------------------------------------------------
    agg = defaultdict(list)

    allowed_customers = [2000, 1000, 500, 750, 1500]
    for filename, (_, list_of_dicts) in new_relative_HGS_dict.items():


        # Extract number of customers from filename: sol_instance_750_336_45.pkl.gz
        match = re.search(r"sol_instance_(\d+)_", filename)
        if not match:
            continue

        num_customers = int(match.group(1))

        # Skip if not in allowed list
        if allowed_customers is not None and num_customers not in allowed_customers:
            continue
        for d in list_of_dicts:
            (subkey, triple), = d.items()   # unpack the single key-value pair
            agg[subkey].append(triple)

    # ---------------------------------------------------------
    # 2. Compute mean and std for each subkey
    # ---------------------------------------------------------
    stats = {}
    for subkey, triples in agg.items():
        arr = np.array(triples)  # shape (N, 3)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        stats[subkey] = (mean, std)

    # ---------------------------------------------------------
    # 3. Sort subkeys for plotting
    # ---------------------------------------------------------
    subkeys = sorted(stats.keys())
    x = np.arange(len(subkeys))

    means = np.array([stats[k][0] for k in subkeys])  # (K,3)
    stds  = np.array([stats[k][1] for k in subkeys])  # (K,3)

    # ---------------------------------------------------------
    # 4. Plot with dual y-axis
    # ---------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left axis: first two metrics
    labels_left = ["relative_num_iter", "relative_runtime"]
    colors_left = ["tab:blue", "tab:orange"]

    for i in range(2):
        ax1.errorbar(
            x, means[:, i], yerr=stds[:, i],
            marker="o", capsize=4,
            label=labels_left[i], color=colors_left[i]
        )

    ax1.set_xlabel("(Neighbourhood size, Graph k-NN restriction)")
    ax1.set_ylabel("relative_num_iter / relative_runtime")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(k) for k in subkeys], rotation=45)

    # ---------------------------------------------------------
    # 5. Right axis: third metric (relative_value)
    # ---------------------------------------------------------
    ax2 = ax1.twinx()

    # Plot the main curve
    ax2.errorbar(
        x, means[:, 2], yerr=stds[:, 2],
        marker="s", capsize=4,
        label="relative_value", color="tab:green"
    )

    # Highlight negative points
    neg_idx = np.where(means[:, 2] < 0)[0]
    if len(neg_idx) > 0:
        ax2.scatter(
            x[neg_idx], means[neg_idx, 2],
            color="red", s=60, zorder=5, label="negative values"
        )

        # Annotate each negative point in scientific notation
        for idx in neg_idx:
            val = means[idx, 2]
            ax2.text(
                x[idx], val,
                f"{val:.1e}",   # scientific notation with 3 significant digits
                color="red", fontsize=9,
                ha="center", va="bottom"
            )

    # Highlight negative points
    neg_idx = np.where(means[:, 1] < 0)[0]
    if len(neg_idx) > 0:
        ax1.scatter(
            x[neg_idx], means[neg_idx, 1],
            color="blue", s=60, zorder=5, label="negative runtime"
        )

        # Annotate each negative point in scientific notation
        for idx in neg_idx:
            val = means[idx, 1]
            ax1.text(
                x[idx], val,
                f"{val:.2f}",   # scientific notation with 3 significant digits
                color="blue", fontsize=9,
                ha="center", va="bottom"
            )

    ax2.set_ylabel("relative_value")

    # ---------------------------------------------------------
    # 6. Combined legend
    # ---------------------------------------------------------
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Mean ± Std of Metrics Across Instances (Dual Y-Axis, Negative Highlighted) for CVRPTW 2000s")
    plt.tight_layout()
    plt.show()


def plt_new_HGS_77_inf_r(dict_path):
    """Same as plt_new_HGS_77 but with completion-time inflation, restricted to a hand-picked set of (neighbourhood-size, k-NN) configurations and a 60s time horizon."""

    with open(dict_path, "rb") as f:
        HGS_time_limit_dict_sol = pkl.load(f)

    print(HGS_time_limit_dict_sol)
    new_relative_HGS_dict = {}

    graph_size_dict = {2001000 : 2000, 281625 : 750, 125250 : 500, 500500 : 1000, 1125750: 1500 }

    for key in HGS_time_limit_dict_sol.keys():
        if HGS_time_limit_dict_sol[key] == {}:
            continue
        for key_2 in HGS_time_limit_dict_sol[key]:
            if key_2 == (1,1):
                filtered_list = [t for t in HGS_time_limit_dict_sol[key][key_2][1] if int(t[1].rstrip("s")) < 60]
                complete_value = filtered_list[-1][2]
                print(HGS_time_limit_dict_sol[key][key_2][2][0])
                complete_time = int(filtered_list[-1][1].rstrip("s"))
                complete_iteration = filtered_list[-1][0]
                new_relative_HGS_dict[key] = [(complete_iteration, complete_time, complete_value), []]
                break
        longest_iter = new_relative_HGS_dict[key][0][0]
        longest_time = new_relative_HGS_dict[key][0][1]
        longest_value = new_relative_HGS_dict[key][0][2]
        for key_2 in HGS_time_limit_dict_sol[key]:
            print(key, ", ", key_2, ", ", HGS_time_limit_dict_sol[key][key_2][2][0])
            filtered_list = [t for t in HGS_time_limit_dict_sol[key][key_2][1] if int(t[1].rstrip("s")) < 60]
            if(len(filtered_list)==0):
                print(key_2)
                print(key)
                continue
            complete_value = filtered_list[-1][2]
            complete_time = int(filtered_list[-1][1].rstrip("s")) + HGS_time_limit_dict_sol[key][key_2][2][1]
            for t in range(len(filtered_list)):
                if filtered_list[t][2] < longest_value:
                    complete_value = filtered_list[t][2]
                    complete_time = int(filtered_list[t][1].rstrip("s")) + HGS_time_limit_dict_sol[key][key_2][2][1]
                    break

            complete_iteration = filtered_list[-1][0]

            new_relative_HGS_dict[key][1].append({key_2 : ((complete_iteration-longest_iter)*100/longest_iter,
                                                 (complete_time-longest_time)*100/longest_time,
                                                 (complete_value-longest_value)*100/longest_value)})


    plt.rcParams.update({
        "font.size": 20,          # base font size
        "axes.titlesize": 20,     # title
        "axes.labelsize": 18,     # x/y labels
        "xtick.labelsize": 16,    # x tick labels
        "ytick.labelsize": 16,    # y tick labels
        "legend.fontsize": 16     # legend
    })
    # ---------------------------------------------------------
    # 1. Aggregate triples by subkey
    # ---------------------------------------------------------
    agg = defaultdict(list)

    allowed_customers = [2000, 1000, 500, 750, 1500]
    for filename, (_, list_of_dicts) in new_relative_HGS_dict.items():


        # Extract number of customers from filename: sol_instance_750_336_45.pkl.gz
        match = re.search(r"sol_instance_(\d+)_", filename)
        if not match:
            continue

        num_customers = int(match.group(1))

        # Skip if not in allowed list
        if allowed_customers is not None and num_customers not in allowed_customers:
            continue
        for d in list_of_dicts:
            (subkey, triple), = d.items()   # unpack the single key-value pair
            agg[subkey].append(triple)

    # ---------------------------------------------------------
    # 2. Compute mean and std for each subkey
    # ---------------------------------------------------------
    stats = {}
    for subkey, triples in agg.items():
        arr = np.array(triples)  # shape (N, 3)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        stats[subkey] = (mean, std)

    selected = [
        (20, 100),
        (20, 200),
        (20, 300),
        (20, 500),
        (10, 100),
        (10, 200),
        (10, 300),
        (15, 100),
        (15, 200),
        (15, 300),
        (5, 100),
        (5,200),
        (25, 100),
        (25, 200),
        (25, 300),
        (30,100)
        # add more here
    ]

    stats = {k: v for k, v in stats.items() if k in selected}
    # ---------------------------------------------------------
    # 3. Sort subkeys for plotting
    # ---------------------------------------------------------
    subkeys = sorted(stats.keys())
    x = np.arange(len(subkeys))

    means = np.array([stats[k][0] for k in subkeys])  # (K,3)
    stds  = np.array([stats[k][1] for k in subkeys])  # (K,3)

    # ---------------------------------------------------------
    # 4. Plot with dual y-axis
    # ---------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left axis: first two metrics
    labels_left = ["relative_num_iter %", "relative_runtime %"]
    colors_left = ["tab:blue", "tab:orange"]

    for i in range(2):
        ax1.errorbar(
            x, means[:, i], yerr=stds[:, i],
            marker="o", capsize=4,
            label=labels_left[i], color=colors_left[i]
        )

    ax1.set_xlabel("Neighbourhood Candidate-Liste size")
    ax1.set_ylabel("relative_num_iter % / relative_runtime %")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(k[0]) for k in subkeys], rotation=45)


    # ---------------------------------------------------------
    # 5. Right axis: third metric (relative_value)
    # ---------------------------------------------------------
    ax2 = ax1.twinx()

    # Plot the main curve
    ax2.errorbar(
        x, means[:, 2], yerr=stds[:, 2],
        marker="s", capsize=4,
        label="relative_value %", color="tab:green"
    )

    # Highlight negative points
    neg_idx = np.where(means[:, 2] < 0)[0]
    if len(neg_idx) > 0:
        ax2.scatter(
            x[neg_idx], means[neg_idx, 2],
            color="red", s=60, zorder=5, label="negative values %"
        )

        # Annotate each negative point in scientific notation
        for idx in neg_idx:
            val = means[idx, 2]
            ax2.text(
                x[idx], val - 0.30 * abs(val),   # push further down
                f"{val:.0e}",
                color="red", fontsize=13,
                ha="center", va="top"
            )

    # Highlight negative points
    neg_idx = np.where(means[:, 1] < 0)[0]
    if len(neg_idx) > 0:
        ax1.scatter(
            x[neg_idx], means[neg_idx, 1],
            color="blue", s=60, zorder=5, label="negative runtime %"
        )

        # Annotate each negative point in scientific notation
        for idx in neg_idx:
            val = means[idx, 1]
            ax1.text(
                x[idx], val + 0.20 * abs(val),
                f"{val:.1f}",   # scientific notation with 3 significant digits
                color="blue", fontsize=13,
                ha="center", va="bottom"
            )

    ax2.set_ylabel("relative_value %")

    # ---------------------------------------------------------
    # 6. Combined legend
    # ---------------------------------------------------------
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    all_lines  = lines1 + lines2
    all_labels = labels1 + labels2

    # Split into two halves
    mid = len(all_lines) // 2

    legend1 = ax1.legend(
        all_lines[:mid],
        all_labels[:mid],
        loc="upper left",
        bbox_to_anchor=(0.28, 1.0)
    )

    legend2 = ax1.legend(
        all_lines[mid:],
        all_labels[mid:],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0)   # adjust horizontal spacing if needed
    )

    ax1.add_artist(legend1)
    ax1.add_artist(legend2)


    plt.title("Mean ± Std of Metrics Across Instances (Negative Highlighted) for CVRPTW 60s")
    plt.tight_layout()
    plt.show()


def plt_new_HGS_77_average_runtime(dict_path):
    """Plots, for each (neighbourhood-size, k-NN) configuration, the average relative-cost-gap-vs-runtime curve across instances, with the PyVRP baseline drawn in bold black."""

    with open(dict_path, "rb") as f:
        HGS_time_limit_dict_sol = pkl.load(f)

    new_relative_HGS_dict = {}

    for key in HGS_time_limit_dict_sol.keys():
        for key_2 in HGS_time_limit_dict_sol[key]:
            if key_2 == (1,1):
                filtered_list = [t for t in HGS_time_limit_dict_sol[key][key_2][1] if int(t[1].rstrip("s")) < 3000]
                complete_value = filtered_list[-1][2]
                complete_time = int(filtered_list[-1][1].rstrip("s"))
                complete_iteration = filtered_list[-1][0]
                new_relative_HGS_dict[key] = [(complete_iteration, complete_time, complete_value), []]
                break
        longest_iter = new_relative_HGS_dict[key][0][0]
        longest_time = new_relative_HGS_dict[key][0][1]
        longest_value = new_relative_HGS_dict[key][0][2]
        for key_2 in HGS_time_limit_dict_sol[key]:
            filtered_list = [t for t in HGS_time_limit_dict_sol[key][key_2][1] if int(t[1].rstrip("s")) < 3000]
            if (len(filtered_list)== 0):
                print(key)
                print(key_2)
                continue
            last_time=int(filtered_list[0][1].rstrip("s"))
            last_value=filtered_list[0][2]
            new_average_list= []
            for t in range(len(filtered_list)):
                compte_time_step = (int(filtered_list[t][1].rstrip("s"))-last_time)//5
                for i in range(compte_time_step):
                    new_average_list.append((last_value-longest_value)/longest_value)
                last_value=filtered_list[t][2]
                last_time=int(filtered_list[t][1].rstrip("s"))
            while len(new_average_list)<400:
                new_average_list.append(new_average_list[-1])

            new_relative_HGS_dict[key][1].append({key_2 : new_average_list})

    agg = defaultdict(list)

    for instance_key, (_, list_of_dicts) in new_relative_HGS_dict.items():
        for d in list_of_dicts:
            (subkey, values), = d.items()   # unpack the single key-value pair
            if(subkey != (5,100) and subkey != (5,200)):
                agg[subkey].append(values)

    # ---------------------------------------------------------
    # 2. Compute element-wise averages
    # ---------------------------------------------------------
    avg_curves = {}
    for subkey, list_of_lists in agg.items():
        arr = np.array(list_of_lists)   # shape (num_instances, T)
        avg = arr.mean(axis=0)
        avg_curves[subkey] = avg

    # ---------------------------------------------------------
    # 3. Build x-axis = runtime = index * 5 seconds
    # ---------------------------------------------------------
    # Assume all lists have same length
    T = len(next(iter(avg_curves.values())))
    x = np.arange(T) * 5   # runtime in seconds

    # ---------------------------------------------------------
    # 4. Plot one curve per subkey
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))

    for subkey, avg_list in sorted(avg_curves.items()):
        if subkey == (1, 1):
            # Highlight PyVRP curve
            plt.plot(
                x, avg_list,
                label="PyVRP",
                linewidth=3.0,
                color="black"
            )
        else:
            plt.plot(
                x, avg_list,
                label=str(subkey),
                linewidth=1.5
            )


    plt.xlabel("Runtime (seconds)")
    plt.ylabel("Average value")
    plt.title("Average curves per subkey across instances for CVRPTW")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plt_new_HGS_77_average_runtime_2(dict_path):
    """Like plt_new_HGS_77_average_runtime but shifts each curve's x-axis by its initial preprocessing time and plots only a hand-picked subset of (neighbourhood-size, k-NN) pairs as average RVG percent over runtime."""

    with open(dict_path, "rb") as f:
        HGS_time_limit_dict_sol = pkl.load(f)

    new_relative_HGS_dict = {}

    for key in HGS_time_limit_dict_sol.keys():
        for key_2 in HGS_time_limit_dict_sol[key]:
            if key_2 == (1,1):
                filtered_list = [t for t in HGS_time_limit_dict_sol[key][key_2][1] if int(t[1].rstrip("s")) < 3000]
                complete_value = filtered_list[-1][2]
                complete_time = int(filtered_list[-1][1].rstrip("s"))
                complete_iteration = filtered_list[-1][0]
                new_relative_HGS_dict[key] = [(complete_iteration, complete_time, complete_value), []]
                break
        longest_iter = new_relative_HGS_dict[key][0][0]
        longest_time = new_relative_HGS_dict[key][0][1]
        longest_value = new_relative_HGS_dict[key][0][2]
        for key_2 in HGS_time_limit_dict_sol[key]:
            filtered_list = [t for t in HGS_time_limit_dict_sol[key][key_2][1] if int(t[1].rstrip("s")) < 3000]
            if (len(filtered_list)== 0):
                print(key)
                print(key_2)
                continue
            last_time=int(filtered_list[0][1].rstrip("s"))
            last_value=filtered_list[0][2]
            new_average_list= []
            for t in range(len(filtered_list)):
                compte_time_step = (int(filtered_list[t][1].rstrip("s"))-last_time)//5
                for i in range(compte_time_step):
                    new_average_list.append((last_value-longest_value)*100/longest_value)
                last_value=filtered_list[t][2]
                last_time=int(filtered_list[t][1].rstrip("s"))
            while len(new_average_list)<400:
                new_average_list.append(new_average_list[-1])

            new_relative_HGS_dict[key][1].append({key_2 : [new_average_list, HGS_time_limit_dict_sol[key][key_2][2][0] -2]})


    agg = defaultdict(list)
    agg_2 = defaultdict(list)

    for instance_key, (_, list_of_dicts) in new_relative_HGS_dict.items():
        for d in list_of_dicts:
            (subkey, values), = d.items()   # unpack the single key-value pair
            if(subkey != (5,100) and subkey != (5,200)):
                agg[subkey].append(values[0])
                agg_2[subkey].append(values[1])

    # ---------------------------------------------------------
    # 2. Compute element-wise averages
    # ---------------------------------------------------------
    avg_curves = {}
    avg_curves_2 = {}
    for subkey, list_of_lists in agg.items():
        arr = np.array(list_of_lists)   # shape (num_instances, T)
        avg = arr.mean(axis=0)
        avg_curves[subkey] = avg

    print(agg_2)
    for subkey, list_2 in agg_2.items():

        arr_2 = np.array(list_2)   # shape (num_instances, T)
        avg_2 = arr_2.mean(axis=0)
        avg_curves_2[subkey] = avg_2

    print(avg_curves_2)

    # List of curves you want to plot
    selected = [
        (20, 100),
        (20, 200),
        (20, 300),
        # add more here
    ]

    # Base x-axis
    T = len(next(iter(avg_curves.values())))
    x_base = np.arange(T) * 5

    plt.rcParams.update({
        "font.size": 20,          # base font size
        "axes.titlesize": 22,     # title
        "axes.labelsize": 20,     # x/y labels
        "xtick.labelsize": 18,    # x tick labels
        "ytick.labelsize": 18,    # y tick labels
        "legend.fontsize": 18     # legend
    })

    plt.figure(figsize=(12, 6))

    for subkey in sorted(avg_curves.keys()):
        if subkey not in selected and subkey != (1, 1):
            continue

        avg_list = avg_curves[subkey]
        initial_time = avg_curves_2[subkey]   # scalar shift

        x_shifted = x_base + initial_time

        if subkey == (1, 1):
            plt.plot(
                x_shifted, avg_list,
                label="PyVRP",
                linewidth=3.0,
                color="black"
            )
        else:
            plt.plot(
                x_shifted, avg_list,
                label=str(subkey),
                linewidth=1.5
            )

    plt.xlabel("Runtime (seconds)")
    plt.ylabel("Average RVG %")
    plt.title("Average RVG % over runtime")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
