"""Sample generation for CVRP / CVRPTW instances + HGS solving + dataset splitting.

Consolidated entry point that builds Munich/CVRPLIB instances, solves them with
the HGS heuristic, post-processes solutions, and splits resulting datasets.
"""

import os
import gzip
import math
import random
import shutil
import pickle as pkl
from pathlib import Path
import re
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import numpy as np
import hydra
from omegaconf import DictConfig
from pyvrp import Model as HGS_Model
from pyvrp.stop import MaxRuntime as HGS_MaxRuntime

from core.utils.cvrp import CVRP_node
from core.utils.cvrp import CVRP
from core.data_processing.data_utils import dict_to_instance
from core.cvrp_solvers.heuristics import heu_solve_HGS_VRPTW
from core.cvrp_solvers.heuristics import heu_solve_HGS_VRP




def _import_cvrpTW_via_VRP_Easy():
    """Lazy import; only loaded when ``is_time_windows`` decoder is requested."""
    from core.cvrp_solvers.ip_grb import cvrpTW_via_VRP_Easy  # noqa: WPS433
    return cvrpTW_via_VRP_Easy

# ---------------------------------------------------------------------------
# Instance generators
# ---------------------------------------------------------------------------

def generate_cvrp_instance(num_nodes=20, vehicle_capacity=30, nb_vehicles=5):
    """Generate a random fully-connected CVRP instance."""
    # Create nodes (node 0 = depot, others = clients)
    nodes = [CVRP_node(0, demand=0, x=0.0, y=0.0)]
    for i in range(1, num_nodes):
        demand = np.random.randint(1, 10)
        x, y = np.random.rand(2) * 100
        nodes.append(CVRP_node(i, demand, x, y))

    # Fully connected undirected arcs (excluding self-loops)
    arc_index = np.array(
        [[i for i in range(num_nodes) for j in range(num_nodes) if i < j],
         [j for i in range(num_nodes) for j in range(num_nodes) if i < j]]
    )

    # Random arc costs
    arc_costs = np.random.randint(1, 20, size=arc_index.shape[1])

    return CVRP(nodes, arc_index, vehicle_capacity, arc_costs, nb_vehicles)

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
def parse_cvrp_literatur_instances(path_instance, undirected = True):
    """Parse a CVRP literature .vrp file (directed or undirected)."""
    # nb_vehicles = 100
    with open(path_instance, "r") as f :
        lines = f.readlines()

    coords = {}
    demands = {}
    vehicle_capacity = None
    mode = None

    for line in lines :
        line = line.strip()

        if line.startswith("CAPACITY"):
            vehicle_capacity = int(line.split(":")[1])

        if line.startswith("NODE_COORD_SECTION"):
            mode="coords"
            continue
        if line.startswith("DEMAND_SECTION"):
            mode="demands"
            continue
        if line.startswith("DEPOT_SECTION"):
            mode = "depot"
            continue
        if line.startswith("EOF"):
            break

        if mode=="coords" and line:
            parts = line.split()
            idx = int(parts[0])-1
            x = float(parts[1])
            y = float(parts[2])
            coords[idx] = (x, y)

        if mode=="demands" and line:
            parts = line.split()
            idx = int(parts[0])-1
            demands[idx] = int(parts[1])


    nodes = []

    for idx in sorted(coords.keys()):
        x, y = coords[idx]
        d = demands[idx]
        nodes.append(CVRP_node(idx, d, x, y))

    nb_vehicles = len(nodes)-1 #default value since here the number of vehicles is supposed to be unlimited
    arc_index_list = [[],[]]
    arc_cost_list = []

    node_ids = sorted(coords.keys())

    for i in node_ids:
        x1, y1 = coords[i]
        for j in node_ids:
            if undirected:
                if i < j:
                    x2, y2 = coords[j]

                    dist = round(math.sqrt((x1-x2)**2 + (y1-y2)**2))
                    arc_index_list[0].append(i)
                    arc_index_list[1].append(j)
                    arc_cost_list.append(dist)
            else:
                if i != j:
                    x2, y2 = coords[j]

                    dist = round(math.sqrt((x1-x2)**2 + (y1-y2)**2))
                    arc_index_list[0].append(i)
                    arc_index_list[1].append(j)
                    arc_cost_list.append(dist)

    arc_index = np.array(arc_index_list)
    arc_costs = np.array(arc_cost_list)
    return CVRP(nodes, arc_index, vehicle_capacity, arc_costs, nb_vehicles)

def parse_solution_file(path, depot=0):
    """Parse a CVRP .sol file into a {(u,v): 1} arc-usage dictionary."""
    arcs = {}   # dictionary of arcs (u, v) → 1

    with open(path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Stop at cost line
        if line.startswith("Cost"):
            break

        # Parse route lines
        if line.startswith("Route"):
            parts = line.split(":")[1].strip().split()
            route = [int(x) for x in parts]

            # Add depot → first
            arcs[(depot, route[0])] = 1

            # Add internal arcs
            for u, v in zip(route[:-1], route[1:]):
                arcs[(u, v)] = 1

            # Add last → depot
            arcs[(route[-1], depot)] = 1

    return arcs


def generate_cvrp_sample_from_litterature(instance_path, solution_path, save_folder_path, undirected = True, depot = 0):


    cvrp_instance = parse_cvrp_literatur_instances(instance_path, undirected)
    solution_cvrp_instance = parse_solution_file(solution_path, depot)


    arc_indicator = {} # All arcs in the instance
    sources = cvrp_instance.arc_index[0]
    targets = cvrp_instance.arc_index[1]

    for u, v in zip(sources, targets):
        if (u, v) in solution_cvrp_instance or (v,u) in solution_cvrp_instance:
            arc_indicator[(u, v)] = 1 
        else:
            arc_indicator[(u, v)] = 0


    sample = {
        "instance": cvrp_instance.to_dict(),
        "solution": arc_indicator,
        "runtime": 0,
        "opt_gap": None,
        "opt_status": "best_opt",
    }
    # save_filename = filename.replace(".vrp", "")
    save_filename = instance_path[:-6]
    with gzip.open(save_folder_path +save_filename +".pkl.gz", "wb") as f:
        pkl.dump(sample, f)


# ---------------------------------------------------------------------------
# HGS solvers
# ---------------------------------------------------------------------------

def solve_HGS_VRP(instance, path):
    """Solve a CVRP instance with HGS and pickle the sample to `path`."""

    index_node = 0

    m = HGS_Model()
    m.add_vehicle_type(capacity=instance.vehicle_capacity, num_available=instance.nb_vehicles)
    total_nodes_dict = {}

    depot_node = instance.depot
    depot_node_id = depot_node.node_id
    depot = m.add_depot(x=depot_node.x, y=depot_node.y, name="f{depot_node.node_id}")

    index_node += 1
    total_nodes_dict[depot_node.node_id] = depot
    for node in instance.clients:
        total_nodes_dict[node.node_id] = m.add_client(x=node.x, y=node.y, delivery=int(node.demand), name = "f{node.node_id}")

    for i , arc in enumerate(instance.arc_list):
        m.add_edge(total_nodes_dict[arc[0]], total_nodes_dict[arc[1]],
                    distance=instance.arc_costs[i])
    res = m.solve(stop=HGS_MaxRuntime(500))
    print(res)
    sol_dict = {(int(src), int(dst)): 0
             for (src, dst) in (zip(instance.arc_index[0], instance.arc_index[1]))}
    for route in res.best.routes():
        visits = route.visits()

        # now build sol_dict using original IDs
        sol_dict[(depot_node_id, visits[0])] = 1
        sol_dict[(visits[-1], depot_node_id)] = 1
        for j in range(1, len(visits)):
            sol_dict[(visits[j-1], visits[j])] = 1


    sample = {
        "instance": instance.to_dict(),
        "solution": sol_dict,
        "runtime": res.runtime,
        "opt_gap": None,
        "opt_status": "heuristic HGS",
    }
    print("path : ", path)
    with gzip.open(path, "wb") as f:
        pkl.dump(sample, f)


def solve_HGS_VRP_test_cluster(instance, path):
    """Solve a CVRP with HGS (short timeout) and store all-pairs cluster arcs."""

    index_node = 0

    m = HGS_Model()
    m.add_vehicle_type(capacity=instance.vehicle_capacity, num_available=instance.nb_vehicles)
    total_nodes_dict = {}

    depot_node = instance.depot
    depot_node_id = depot_node.node_id
    depot = m.add_depot(x=depot_node.x, y=depot_node.y, name="f{depot_node.node_id}")

    index_node += 1
    total_nodes_dict[depot_node.node_id] = depot
    for node in instance.clients:
        total_nodes_dict[node.node_id] = m.add_client(x=node.x, y=node.y, delivery=int(node.demand), name = "f{node.node_id}")

    for i , arc in enumerate(instance.arc_list):
        m.add_edge(total_nodes_dict[arc[0]], total_nodes_dict[arc[1]],
                    distance=instance.arc_costs[i])
    res = m.solve(stop=HGS_MaxRuntime(3))
    print(res)
    sol_dict = {(int(src), int(dst)): 0
             for (src, dst) in (zip(instance.arc_index[0], instance.arc_index[1]))}
    for route in res.best.routes():
        visits = route.visits()

        # now build sol_dict using original IDs

        for i in range(len(visits)-1):
            for j in range(i+1, len(visits)):
                sol_dict[(visits[i], visits[j])] = 1
                sol_dict[(visits[j], visits[i])] = 1

    sample = {
        "instance": instance.to_dict(),
        "solution": sol_dict,
        "runtime": res.runtime,
        "opt_gap": None,
        "opt_status": "heuristic HGS",
    }
    with gzip.open(path, "wb") as f:
        pkl.dump(sample, f)


def solve_HGS_VRP_undirected(instance, path, ts = 1000):
    """Persist an undirected CVRP instance with placeholder HGS solution metadata."""

    index_node = 0

    sol_dict = {}


    #     total_nodes_dict[node.node_id] = m.add_client(x=node.x, y=node.y, delivery=int(node.demand), name = "f{node.node_id}")

    #     cost = instance.arc_costs[i]

    #     m.add_edge(total_nodes_dict[u], total_nodes_dict[v], distance=cost)
    #     m.add_edge(total_nodes_dict[v], total_nodes_dict[u], distance=cost)


    #          for (src, dst) in (zip(instance.arc_index[0], instance.arc_index[1]))}
    #     visits = route.visits()

    #     sol_dict[(depot_node_id, visits[0])] = 1
    #     sol_dict[(depot_node_id, visits[-1])] = 1
    #             sol_dict[(visits[j-1], visits[j])] = 1
    #         else:
    #             sol_dict[(visits[j], visits[j-1])] = 1

    runtime = 0
    best_known_value=999999999
    sample = {
        "instance": instance.to_dict(),
        "solution": sol_dict,
        "runtime": runtime,
        "opt_gap": None,
        "opt_status": "heuristic HGS",
        "best_known_value": best_known_value
    }
    print("path : ", path)
    with gzip.open(path, "wb") as f:
        pkl.dump(sample, f)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def generate_double_arc_solution(instances_path, new_instance_path):
    """Symmetrize HGS solutions by adding the reverse arc for every used arc."""

    os.makedirs(new_instance_path, exist_ok=True)
    for root, filename, files in os.walk(instances_path):
        for instance_file in files:
            with gzip.open(instances_path +"/" + instance_file, "rb") as f:
                result_dict = pkl.load(f)

            solution_dict = result_dict["solution"]
            new_solution_dict = solution_dict

            for (u,v) in new_solution_dict:
                if (new_solution_dict[(u,v)]==1):
                    new_solution_dict[(v,u)]=1


            sample = {
                "instance": result_dict["instance"],
                "solution": new_solution_dict,
                "runtime": result_dict["runtime"],
                "opt_gap": None,
                "opt_status": "heuristic HGS",
            }
            new_path = f"{new_instance_path}/" + "double" + instance_file
            with gzip.open(new_path, "wb") as f:
                pkl.dump(sample, f)


def generate_undirected_solution(instances_path, new_instance_path, start=0, end=None):
    """Convert directed HGS solutions/instances to their undirected representation."""

    os.makedirs(new_instance_path, exist_ok=True)
    all_files = sorted([
        f for f in os.listdir(instances_path)
        if f.endswith(".pkl.gz") or f.endswith(".gz")
    ])

    # Default: process until the end
    if end is None:
        end = len(all_files)

    # Slice the range
    selected_files = all_files[start:end]

    print(f"Processing files {start} to {end-1} (total {len(selected_files)})")

    for instance_file in selected_files:
        with gzip.open(os.path.join(instances_path, instance_file), "rb") as f:
            result_dict = pkl.load(f)

            solution_dict = result_dict["solution"]
            new_solution_dict = solution_dict

            for (u,v) in new_solution_dict:
                if (new_solution_dict[(u,v)]==1):
                    new_solution_dict[(v,u)]=1

            undirected_solution_dict = {}

            for (u,v) in  new_solution_dict:
                if u < v :
                    undirected_solution_dict[(u,v)] = new_solution_dict[(u,v)]

            copy_instance, _ = dict_to_instance(result_dict)

            arc_index_list = [[], []]
            arc_costs_list = []

            for k, (u,v) in enumerate(copy_instance.arc_list):
                if u < v:
                    arc_index_list[0].append(u)
                    arc_index_list[1].append(v)
                    arc_costs_list.append(copy_instance.arc_costs[k])

            arc_index = np.array(arc_index_list)
            arc_costs = np.array(arc_costs_list)
            new_instance = CVRP(copy_instance.nodes, arc_index,
            copy_instance.vehicle_capacity, arc_costs, copy_instance.nb_vehicles)


            sample = {
                "instance": new_instance.to_dict(),
                "solution": undirected_solution_dict,
                "runtime": result_dict["runtime"],
                "opt_gap": None,
                "opt_status": "heuristic HGS",
            }
            new_path = f"{new_instance_path}/" + "undirected_" + instance_file
            with gzip.open(new_path, "wb") as f:
                pkl.dump(sample, f)


def generate_directed_solution_TW(instances_path, new_instance_path):
    """Duplicate every arc both directions so an undirected CVRPTW becomes directed."""

    os.makedirs(new_instance_path, exist_ok=True)
    for root, filename, files in os.walk(instances_path):
        for instance_file in files:
            with gzip.open(instances_path +"/" + instance_file, "rb") as f:
                result_dict = pkl.load(f)

            copy_instance, _ = dict_to_instance(result_dict)

            arc_index_list = [[], []]
            arc_costs_list = []

            for k, (u,v) in enumerate(copy_instance.arc_list):
                    arc_index_list[0].append(u)
                    arc_index_list[1].append(v)
                    arc_costs_list.append(copy_instance.arc_costs[k])
                    arc_index_list[0].append(v)
                    arc_index_list[1].append(u)
                    arc_costs_list.append(copy_instance.arc_costs[k])
                    if (v,u) not in result_dict["solution"].keys():
                        result_dict["solution"][(v,u)] = 0


            arc_index = np.array(arc_index_list)
            arc_costs = np.array(arc_costs_list)
            new_instance = CVRP(copy_instance.nodes, arc_index,
            copy_instance.vehicle_capacity, arc_costs, copy_instance.nb_vehicles)


            sample = {
                "instance": new_instance.to_dict(),
                "solution": result_dict["solution"],
                "runtime": result_dict["runtime"],
                "opt_gap": None,
                "opt_status": "heuristic HGS",
            }
            new_path = f"{new_instance_path}/" + instance_file
            with gzip.open(new_path, "wb") as f:
                pkl.dump(sample, f)


# ---------------------------------------------------------------------------
# CVRPTW restricted
# ---------------------------------------------------------------------------

def plot_study_CVRPTW_vs_CVRP_generation(nb_common_arcs_dict, relative_cost_gap_dict):
    # Consistent ordering
    keys = list(nb_common_arcs_dict.keys())
    x = range(len(keys))

    # Extract values
    nb_vals = [nb_common_arcs_dict[k][1] for k in keys]
    nb_types = [nb_common_arcs_dict[k][0] for k in keys]
    nb_status = [nb_common_arcs_dict[k][2] for k in keys]
    nb_new_routes = [nb_common_arcs_dict[k][3] for k in keys]

    cost_vals = [relative_cost_gap_dict[k][1] for k in keys]
    cost_types = [relative_cost_gap_dict[k][0] for k in keys]
    cost_status = [relative_cost_gap_dict[k][2] for k in keys]
    cost_new_routes = [relative_cost_gap_dict[k][3] for k in keys]

    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # ---- LEFT AXIS (nb_common_arcs) ----
    ax1.plot(x, nb_vals, color='blue', label='nb_common_arcs (curve)')
    for i in x:
        marker = 'o' if nb_types[i] else 'x'
        ax1.scatter(i, nb_vals[i], color='blue', marker=marker)
        ax1.text(i, nb_vals[i] + 0.003, str(cost_new_routes[i]), ha='center', va='top', fontsize=9)
    ax1.set_xlabel("Instance index")
    ax1.set_ylabel("nb_common_arcs", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # ---- RIGHT AXIS (relative_cost_gap) ----
    ax2 = ax1.twinx()
    ax2.plot(x, cost_vals, color='red', label='relative_cost_gap (curve)')
    for i in x:
        marker2 = 'o' if cost_types[i] else 'x'
        ax2.scatter(i, cost_vals[i], color='red', marker=marker2)
        ax2.text(i, cost_vals[i] - 0.0002, str(cost_status[i]), ha='center', va='top', fontsize=9)

    ax2.set_ylabel("relative_cost_gap", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # ---- LEGEND FOR CURVES + MARKERS ----
    curve_legend = ax1.legend(loc='upper left')

    large_tw_marker = mlines.Line2D([], [], color='black', marker='o', linestyle='None', label='Large TW')
    narrow_tw_marker = mlines.Line2D([], [], color='black', marker='x', linestyle='None', label='Narrow TW')

    ax1.legend(handles=[large_tw_marker, narrow_tw_marker], loc='upper right')

    plt.title("nb_common_arcs (left axis) vs relative_cost_gap (right axis)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def generate_cvrptw_restricted(instances_path, new_instance_path,
    width_min_prob_large=0.66, width_max_prob_large=0.9, width_min_prob_narrow=0.1,
    width_max_prob_narrow=0.3, speed = 13.9/1000, service_time=30, seed=0, HGS_runtime=1000,
    start=0, end=10, ready_time_init=0, solve_exact=False, exact_time_limit=100):
    """Augment any CVRP instance set with random time windows and solve via HGS.

    The source CVRP graph is undirected; CVRPTW requires directed arcs, so the
    arc index and arc costs are doubled (forward + reverse) before constructing
    the new instance.
    """


    os.makedirs(new_instance_path, exist_ok=True)

    all_files = sorted([
        f for f in os.listdir(instances_path)
        if f.endswith(".pkl.gz") or f.endswith(".gz")
    ])

    # Default: process until the end
    if end is None:
        end = len(all_files)

    # Slice the range
    selected_files = all_files[start:end]

    print(f"Processing files {start} to {end-1} (total {len(selected_files)})")
    nb_common_arc_dict = {}
    relative_cost_gap_dict = {}

    for instance_file in selected_files:
            
            print("instance_file : ", instance_file)
            
            
            with gzip.open(instances_path +"/" + instance_file, "rb") as f:
                result_dict = pkl.load(f)

            rng = np.random.default_rng(seed)

            nums = re.findall(r"\d+", instance_file)

            num1 = nums[-2]   
            num2 = nums[-1]   
            concat_name = num1 + num2

            rng_type_window = np.random.default_rng(int(concat_name))

            choose_type_window = rng_type_window.random() > 0.5
            if choose_type_window: #1 corresponds to large windows
                width_min_prob = width_min_prob_large
                width_max_prob = width_max_prob_large
            else:
                width_min_prob = width_min_prob_narrow
                width_max_prob = width_max_prob_narrow

            print("width_min_prob = ", width_min_prob, ", width_max_prob = ",
            width_max_prob, ", service_time = ", service_time, ", speed = ", speed, "seed = ", seed,
            "choose_type_window = ", choose_type_window)


            if width_min_prob <= 0 or width_max_prob < width_min_prob:
                raise ValueError("Invalid time window width range.")


            copy_instance, solution = dict_to_instance(result_dict)

            max_distance = 0

            min_distance = 999999

            depot_node_time = {}

            new_solution_dict = {}

            total_cost =  0
            nb_original_routes = 0
            for k, (u,v) in enumerate(zip(copy_instance.arc_index[0], copy_instance.arc_index[1])):
                new_solution_dict[(u,v)] = solution[(u,v)]
                new_solution_dict[(v,u)] = solution[(u,v)]
                if(u==0):
                    depot_node_time[v]=math.ceil(copy_instance.arc_costs[k]/speed)
                    nb_original_routes += solution[(u,v)]
                    if copy_instance.arc_costs[k]> max_distance:
                        max_distance = copy_instance.arc_costs[k]
                    if copy_instance.arc_costs[k] < min_distance:
                        min_distance = copy_instance.arc_costs[k]

            nb_original_routes = nb_original_routes/2
            mean_travelling_time = np.mean(copy_instance.arc_costs)/speed

            print("mean_travelling_time : ", mean_travelling_time)
            min_duration = math.ceil(min_distance/speed)
            max_duration = math.ceil(max_distance/speed)

            print("max_duration = ", max_duration)
            print("min_duration = ", min_duration)


            # horizon = int(2*max_duration
            # + sum([node.demand for node in copy_instance.nodes])/copy_instance.vehicle_capacity * (service_time + mean_travelling_time))
            demands = [node.demand for node in copy_instance.nodes]
            prod_after_first = np.prod(demands[1:])

            if prod_after_first == 1:
                aver_nb_customer = copy_instance.vehicle_capacity
            else:
                aver_nb_customer = math.ceil(copy_instance.vehicle_capacity/sum([node.demand for node in copy_instance.nodes])*(len(copy_instance.nodes)-1))

            horizon = math.ceil(2*max_duration + 
                          aver_nb_customer
                          *(mean_travelling_time + service_time)) + ready_time_init # by default ready_time_init=0
            

            print("average nb customers : ", aver_nb_customer)
            print("horizon : ", horizon)

            width_min = width_min_prob*horizon
            width_max = width_max_prob*horizon

            width_mean = (width_max+width_min)/2
            width_sd = (width_max-width_min)/np.sqrt(12)

            new_cvrptw_nodes = []
            i=0
            for node in copy_instance.nodes:
                x, y = node.x, node.y
                d = node.demand
                idx = node.node_id
                if node.demand == 0:
                    ready_time = ready_time_init
                    due_time = int(horizon)
                    new_cvrptw_nodes.append(CVRP_node(idx, d, x, y, ready_time, due_time, 0))

                else:
                    center= rng.uniform(depot_node_time[node.node_id] + ready_time_init,
                                             horizon- depot_node_time[node.node_id]-service_time) #symmetric arcs
                    w = max(rng.normal(width_mean, width_sd), service_time/2)

                    ready_time = max(int(center - w), ready_time_init)
                    due_time = min(int(center + w), horizon)

                    # print("node : ", i,", center : ", center, ", width: ", w, ", ready_time : ", ready_time, ", due_time : ", due_time)
                    new_cvrptw_nodes.append(CVRP_node(idx, d, x, y, ready_time, due_time, service_time))

                i +=1


            # CVRPTW is directed: duplicate every arc with its reverse so that
            # same order (symmetric travel times under the constant `speed`).
            _arc_idx = np.asarray(copy_instance.arc_index)   # shape (2, m)

            # Extract i and j rows
            i = _arc_idx[0]
            j = _arc_idx[1]

            # Build directed arcs
            directed_i = np.concatenate([i, j])   # forward then reverse
            directed_j = np.concatenate([j, i])

            directed_arc_index = np.vstack([directed_i, directed_j])

            # Duplicate costs
            directed_arc_costs = np.concatenate([copy_instance.arc_costs,
                                                copy_instance.arc_costs])


            new_instance = CVRP(new_cvrptw_nodes, directed_arc_index,
            copy_instance.vehicle_capacity, np.ceil(directed_arc_costs/speed), copy_instance.nb_vehicles)

            relevant_connections = [True for k in range(len(new_instance.arc_costs))]
            cvrptw_solution, best_cost, _, feasibility_test, _, _ = heu_solve_HGS_VRPTW(new_instance.nodes, new_instance.arc_index,
                            new_instance.arc_costs, new_instance.nb_vehicles, new_instance.vehicle_capacity,
                            relevant_connections, heu_time = HGS_runtime)

            if feasibility_test :
                runtime  = 0
                status = "HGS_time_limit"
                if solve_exact:

                    cvrpTW_via_VRP_Easy = _import_cvrpTW_via_VRP_Easy()
                    cvrptw_solution_temp, runtime, solver_value, lower_bound, status, build_solver_runtime = cvrpTW_via_VRP_Easy(
                        new_instance.nodes,
                        new_instance.arc_index,
                        new_instance.arc_costs,
                        new_instance.nb_vehicles,
                        new_instance.vehicle_capacity,
                        relevant_connections,
                        False,
                        time_limit=exact_time_limit,
                        upper_bound=best_cost
                    )

                    if solver_value>0:
                        cvrptw_solution = cvrptw_solution_temp
                        best_cost = solver_value



                nb_common_arc=0
                nb_original_activated_arc = 0
                nb_new_routes = 0

                for k, arc in enumerate(cvrptw_solution.keys()):
                    nb_original_activated_arc += new_solution_dict[arc]
                    total_cost += new_instance.arc_costs[k]*new_solution_dict[arc]
                    if cvrptw_solution[arc]==new_solution_dict[arc] and cvrptw_solution[arc]==1:
                        nb_common_arc +=1
                    if arc[0]==0 and cvrptw_solution[arc]==1:
                        nb_new_routes +=1

                total_cost = total_cost/2
                nb_original_activated_arc = nb_original_activated_arc/2
                relative_original_cost_gap = (best_cost-total_cost)/total_cost
                relative_nb_new_route = (nb_new_routes-nb_original_routes)/nb_original_routes

                print("nb_new_routes = ", nb_new_routes)
                print("nb_original_routes = ", nb_original_routes)

                print("original_total_cost = ", total_cost)
                print("relative original cost gap = ", relative_original_cost_gap)
                print("relative_nb_common_arc = ", nb_common_arc/nb_original_activated_arc)
                print("relative_nb_new_route = ", relative_nb_new_route)



                nb_common_arc_dict[instance_file] = (choose_type_window,nb_common_arc/nb_original_activated_arc,status, relative_nb_new_route )
                relative_cost_gap_dict[instance_file] = (choose_type_window, relative_original_cost_gap, status, relative_nb_new_route)

                sample = {
                    "instance": new_instance.to_dict(),
                    "solution": cvrptw_solution,
                    "runtime": HGS_runtime + runtime,
                    "large_window": choose_type_window,
                    "opt_gap": None,
                    "opt_status": status,
                    "relative_nb_common_arcs": nb_common_arc/nb_original_activated_arc,
                    "relative_original_cost_gap": relative_original_cost_gap
                }
                new_path = f"{new_instance_path}/" + "cvrptw_" + instance_file
                with gzip.open(new_path, "wb") as f:
                    pkl.dump(sample, f)

    # print("nb_common_arcs_dict : ", nb_common_arc_dict)
    # print("relative_cost_gap_dict : ", relative_cost_gap_dict)

    # plot_study_CVRPTW_vs_CVRP_generation(nb_common_arc_dict, relative_cost_gap_dict)

    return nb_common_arc_dict, relative_cost_gap_dict

def plot_CVRPTW_comparison(comparison_dict):


    # Prepare x-axis
    keys = list(comparison_dict.keys())
    x = np.arange(len(keys))

    # Storage for TRUE curves
    nb_true_means = []
    nb_true_stds = []

    routes_true_means = []
    routes_true_stds = []

    cost_true_means = []
    cost_true_stds = []

    # Storage for FALSE curves
    nb_false_means = []
    nb_false_stds = []

    routes_false_means = []
    routes_false_stds = []

    cost_false_means = []
    cost_false_stds = []

    # Extract values
    for key in keys:
        true_tuple, false_tuple = comparison_dict[key]

        # TRUE
        nb_true_mean, nb_true_std = true_tuple[0]
        routes_true_mean, routes_true_std = true_tuple[1]
        cost_true_mean, cost_true_std = true_tuple[2]

        nb_true_means.append(nb_true_mean)
        nb_true_stds.append(nb_true_std)

        routes_true_means.append(routes_true_mean)
        routes_true_stds.append(routes_true_std)

        cost_true_means.append(cost_true_mean)
        cost_true_stds.append(cost_true_std)

        # FALSE
        nb_false_mean, nb_false_std = false_tuple[0]
        routes_false_mean, routes_false_std = false_tuple[1]
        cost_false_mean, cost_false_std = false_tuple[2]

        nb_false_means.append(nb_false_mean)
        nb_false_stds.append(nb_false_std)

        routes_false_means.append(routes_false_mean)
        routes_false_stds.append(routes_false_std)

        cost_false_means.append(cost_false_mean)
        cost_false_stds.append(cost_false_std)

    # Create figure with two subplots
    fig, (ax_true, ax_false) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # -------------------- TRUE PLOT --------------------
    ax_true.errorbar(x, nb_true_means, yerr=nb_true_stds, fmt='-o', label='Common arcs (TRUE)')
    ax_true.errorbar(x, routes_true_means, yerr=routes_true_stds, fmt='-s', label='Relative routes (TRUE)')
    ax_true.errorbar(x, cost_true_means, yerr=cost_true_stds, fmt='-^', label='Cost relative gap (TRUE)')

    ax_true.set_title("TRUE category")
    ax_true.set_ylabel("Values")
    ax_true.grid(True)
    ax_true.legend()

    # -------------------- FALSE PLOT --------------------
    ax_false.errorbar(x, nb_false_means, yerr=nb_false_stds, fmt='-o', label='Common arcs (FALSE)')
    ax_false.errorbar(x, routes_false_means, yerr=routes_false_stds, fmt='-s', label='Relative routes (FALSE)')
    ax_false.errorbar(x, cost_false_means, yerr=cost_false_stds, fmt='-^', label='Cost relative gap (FALSE)')

    ax_false.set_title("FALSE category")
    ax_false.set_ylabel("Values")
    ax_false.grid(True)
    ax_false.legend()

    # X-axis labels = keys
    plt.xticks(x, [str(k) for k in keys], rotation=45, ha='right')

    plt.xlabel("Category (key)")
    plt.tight_layout()
    plt.show()


                
def comparison_CVRPTW_generation(instances_path, new_instance_path,
    width_min_prob_large_list=[0.66], width_max_prob_large_list=[0.9],
    width_min_prob_narrow_list=[0.1], width_max_prob_narrow_list=[0.3],
    speed = 13.9/1000, service_time=30, seed=0, HGS_runtime=1000,
    start=0, end=10, ready_time_init=0, solve_exact=False, exact_time_limit=100):

    comparison_dict = {}

    for i in range(len(width_max_prob_large_list)):
        width_max_prob_large = width_max_prob_large_list[i]
        width_min_prob_large = width_min_prob_large_list[i]
        width_max_prob_narrow = width_max_prob_narrow_list[i]
        width_min_prob_narrow = width_min_prob_narrow_list[i]


        nb_common_arcs_dict, relative_cost_gap_dict = generate_cvrptw_restricted(instances_path, new_instance_path,
        width_min_prob_large=width_min_prob_large, width_max_prob_large=width_max_prob_large,
        width_min_prob_narrow=width_min_prob_narrow, width_max_prob_narrow=width_max_prob_narrow,
        speed = speed, service_time=service_time, seed=seed, HGS_runtime=HGS_runtime,
        start=start, end=end, ready_time_init=ready_time_init, solve_exact=solve_exact,
        exact_time_limit=exact_time_limit)

        keys = list(nb_common_arcs_dict.keys())

        # Extract values
        nb_types = [nb_common_arcs_dict[k][0] for k in keys]
        nb_vals = [nb_common_arcs_dict[k][1] for k in keys]
        nb_new_routes = [nb_common_arcs_dict[k][3] for k in keys]
        cost_vals = [relative_cost_gap_dict[k][1] for k in keys]


        # Indices for each type
        idx_true  = [i for i in range(len(nb_types)) if nb_types[i]]
        idx_false = [i for i in range(len(nb_types)) if not nb_types[i]]

        # TRUE group
        nb_true_vals        = np.array([nb_vals[i] for i in idx_true])
        nb_true_new_routes  = np.array([nb_new_routes[i] for i in idx_true])
        cost_true_vals      = np.array([cost_vals[i] for i in idx_true])

        summary_true = (
            (nb_true_vals.mean(),       nb_true_vals.std()),
            (nb_true_new_routes.mean(), nb_true_new_routes.std()),
            (cost_true_vals.mean(),     cost_true_vals.std())
        )

        # FALSE group
        nb_false_vals        = np.array([nb_vals[i] for i in idx_false])
        nb_false_new_routes  = np.array([nb_new_routes[i] for i in idx_false])
        cost_false_vals      = np.array([cost_vals[i] for i in idx_false])

        summary_false = (
            (nb_false_vals.mean(),       nb_false_vals.std()),
            (nb_false_new_routes.mean(), nb_false_new_routes.std()),
            (cost_false_vals.mean(),     cost_false_vals.std())
        )

        # Final structure
        result = [summary_true, summary_false]

        comparison_dict[(width_min_prob_large, width_max_prob_large,
                        width_min_prob_narrow, width_max_prob_narrow)] = result
        
    print(comparison_dict)
    return comparison_dict






# ---------------------------------------------------------------------------
# CVRPLIB-style instance generator
# ---------------------------------------------------------------------------

def generate_CVRP_LIB_instances(path, nb_clients, seed, nb_instances = 100) -> None:
    """Generate CVRPLIB XML100-style .vrp instance files into `path`."""

    np.random.seed(seed)
    Depos_pos_list = []
    Customer_pos_list = []
    Demand_distrib_list = []
    Average_road_list = []

    for i in range(nb_instances//3 + 1):
        Depos_pos_list.extend(np.random.permutation([1, 2, 3]))
        Customer_pos_list.extend(np.random.permutation([1, 2, 3]))
    for i in range(nb_instances//7 + 1):
        Demand_distrib_list.extend(np.random.permutation([1, 2, 3, 4, 5, 6, 7]))

    values = np.random.triangular(left=3, mode=6, right=25, size=100)

    sorted_vals = np.sort(values)
    quintiles = np.array_split(sorted_vals, 5)

    indices = [0, 0, 0, 0, 0]
    for k in range(1, nb_instances + 1):
        q = ((k - 1) % 5)
        Average_road_list.append(quintiles[q][indices[q]])
        indices[q] += 1

    for i in range(nb_instances):
        def distance(x,y):
            """Euclidean distance between two 2D points."""
            return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

        # constants
        maxCoord = 1000
        decay = 40

        # read input argmuments
        n = int(nb_clients)
        rootPos = int(Depos_pos_list[i])
        custPos = int(Customer_pos_list[i])
        demandType = int(Demand_distrib_list[i])
        instanceID = int(i)
        randSeed = int(i) # random seed for reproducibility
        r = Average_road_list[i]
        if demandType > 7:
            print("Demant type out of range!")
            exit(0)

        random.seed(randSeed)

        nSeeds = random.randint(2,6)

        #     print("Average route size out of range!")
        #     exit(0)

        # change '02d' if you need more than two digits (e.g. with '03d' you can index from 001 to 999)
        instanceName = 'sample_'+str(n)+'_'+str(rootPos)+str(custPos)+str(demandType)+'_'+ format(instanceID, '02d')

        pathToWrite = path + "/" +instanceName+'.vrp'

        depot = (-1,-1) # depot position
        S = set() # set of coordinates for the customers

        x_,y_ = (-1,-1)
        #Root positioning
        if rootPos == 1:
            x_ = random.randint(0,maxCoord)
            y_ = random.randint(0,maxCoord)
        elif rootPos == 2:
            x_ = y_ = int(maxCoord/2.0)
        elif rootPos == 3:
            x_ = y_ = 0
        else:
            print("Depot Positioning out of range!")
            exit(0)
        depot = (x_,y_)

        #Customer positioning
        nRandCust = -1
        if custPos == 3:
            nRandCust = int(n/2.0)
        elif custPos == 2:
            nRandCust = 0
        elif custPos == 1:
            nRandCust = n
            nSeeds = 0
        else:
            print("Costumer Positioning out of range!")
            exit(0)

        nClustCust = n - nRandCust

        #Generating random customers
        for i in range(1, nRandCust+1):
            x_ = random.randint(0,maxCoord)
            y_ = random.randint(0,maxCoord)
            while (x_,y_) in S or (x_,y_) == depot:
                x_ = random.randint(0,maxCoord)
                y_ = random.randint(0,maxCoord)
            S.add((x_,y_))

        nS = nRandCust

        seeds = []
        # Generation of the clustered customers
        if nClustCust > 0:
            if nClustCust < nSeeds:
                print("Too many seeds!")
                exit(0)

            #Generate the seeds
            for i in range(nSeeds):
                x_ = random.randint(0,maxCoord)
                y_ = random.randint(0,maxCoord)
                while (x_,y_) in S or (x_,y_) == depot:
                    x_ = random.randint(0,maxCoord)
                    y_ = random.randint(0,maxCoord)
                S.add((x_,y_))
                seeds.append((x_,y_))
            nS = nS + nSeeds

            # Determine the seed with maximum sum of weights (w.r.t. all seeds)
            maxWeight = 0.0
            for i,j in seeds:
                w_ij = 0.0
                for i_,j_ in seeds:
                    w_ij += 2**(-distance((i,j), (i_,j_)) / decay)
                if w_ij > maxWeight:
                    maxWeight = w_ij

            norm_factor = 1.0/maxWeight

            # Generate the remaining customers using Accept-reject method
            while nS < n:
                x_ = random.randint(0,maxCoord)
                y_ = random.randint(0,maxCoord)
                while (x_,y_) in S or (x_,y_) == depot:
                    x_ = random.randint(0,maxCoord)
                    y_ = random.randint(0,maxCoord)

                weight = 0.0
                for i_,j_ in seeds:
                    weight += 2**(-distance((x_,y_), (i_,j_)) / decay)
                weight *= norm_factor
                rand = random.uniform(0,1)

                if rand <= weight: # Will we accept the customer?
                    S.add((x_,y_))
                    nS = nS + 1

        V = [depot] + list(S) # set of vertices (from now on, the ids are defined)

        # Demands
        demandMinValues = [1,1,5,1,50,1,51,50,1]
        demandMaxValues = [1,10,10,100,100,50,100,100,10]
        demandMin = demandMinValues[demandType-1]
        demandMax = demandMaxValues[demandType-1]
        demandMinEvenQuadrant = 51
        demandMaxEvenQuadrant = 100
        demandMinLarge = 50
        demandMaxLarge = 100
        largePerRoute = 1.5
        demandMinSmall = 1
        demandMaxSmall = 10

        D = [] # demands
        sumDemands = 0
        maxDemand = 0

        for i in range(2,n + 2):
            j = int((demandMax - demandMin + 1) * random.uniform(0,1) + demandMin)
            if demandType == 6:
                if (V[i - 1][0] < maxCoord/2.0 and V[i - 1][1] < maxCoord/2.0) or (V[i - 1][0] >= maxCoord/2.0 and V[i - 1][1] >= maxCoord/2.0):
                    j = int((demandMaxEvenQuadrant - demandMinEvenQuadrant + 1) * random.uniform(0,1) + demandMinEvenQuadrant)
            if demandType == 7:
                if i < (n / r) * largePerRoute:
                    j = int((demandMaxLarge - demandMinLarge + 1) * random.uniform(0,1) + demandMinLarge)
                else:
                    j = int((demandMaxSmall - demandMinSmall + 1) * random.uniform(0,1) + demandMinSmall)
            D.append(j)
            if j > maxDemand:
                maxDemand = j
            sumDemands = sumDemands + j

        # Generate capacity
        capacity = -1
        if sumDemands == n:
            capacity = math.floor(r)
        else:
            capacity = max(maxDemand, math.ceil(r * sumDemands / n))

        k = math.ceil(sumDemands/float(capacity))

        f = open(pathToWrite, 'w')
        f.write('NAME : ' + instanceName + '\n')
        f.write('COMMENT : Generated as the XML100 dataset from the CVRPLIB\n')
        f.write('TYPE : CVRP\n')
        f.write('DIMENSION : ' + str(n+1) + '\n')
        f.write('EDGE_WEIGHT_TYPE : EUC_2D\n')
        f.write('CAPACITY : ' + str(int(capacity)) + '\n')
        f.write('NODE_COORD_SECTION\n')

        for i,v in enumerate(V):
            f.write('{:<4}'.format(i+1)+' '+'{:<4}'.format(v[0])+' '+'{:<4}'.format(v[1])+'\n')

        f.write('DEMAND_SECTION\n')
        if demandType != 6:
            random.shuffle(D)
        D = [0] + D
        for i,d in enumerate(V):
            f.write('{:<4}'.format(i+1)+' '+'{:<4}'.format(D[i])+'\n')

        f.write('DEPOT_SECTION\n1\n-1\nEOF\n')

        f.close()
    return pathToWrite


"""New generation function using the instance generator from literature to create sample: Instance and Solution"""

def samples_generation_CVRP_literature(instance_path, save_path, HGS_time_limit=100):
    cvrp_instance = parse_cvrp_literatur_instances(instance_path)
    # solution_cvrp_instance = parse_solution_file(solution_path)


    solution, cost, runtime = heu_solve_HGS_VRP(
        cvrp_instance.demands,
        cvrp_instance.arc_index,
        cvrp_instance.arc_costs,
        cvrp_instance.nb_vehicles,
        cvrp_instance.vehicle_capacity,
        None,
        heu_time=HGS_time_limit,
        undirected=True,
        arc_likelihood=None,
        instance_log_HGS_dict=None,
        threshold=None,
        nodes=None,
        relevant_connections_1=None,
        pyvrp_version=None,
    )



    sample = {
        "instance": cvrp_instance.to_dict(),
        "solution": solution,
        "runtime": runtime,
        "opt_status": f"HGS_runtime{HGS_time_limit}"
    }
    save_filename = instance_path[:-9].replace(".vrp", "")
    with gzip.open(save_path +save_filename +".pkl.gz", "wb") as f:
        pkl.dump(sample, f)


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def split_dataset(
    data_folder,
    new_data_folder,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
):
    """Shuffle a folder of .pkl.gz instances into train/validation/test splits."""
    random.seed(seed)

    instances_dir = Path(data_folder)

    # Collect all instance files
    instance_files = sorted([f for f in instances_dir.glob("*.pkl.gz")])

    # Extract basenames (without extension)
    basenames = [f.stem for f in instance_files]

    # Shuffle
    random.shuffle(basenames)

    # Compute split indices
    n = len(basenames)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_set = basenames[:n_train]
    val_set = basenames[n_train:n_train + n_val]
    test_set = basenames[n_train + n_val:]

    # Prepare output directories
    for split in ["train", "validation", "test"]:
        path = Path(new_data_folder) / split
        path.mkdir(parents=True, exist_ok=True)

    # Helper to copy files
    def copy_pair(base, split):
        """Copy the instance file ``{base}.gz`` into the ``split`` subdirectory."""
        shutil.copy(instances_dir/f"{base}.gz",
                    Path(new_data_folder) / split / f"{base}.gz")

    # Copy files
    for base in train_set:
        copy_pair(base, "train")

    for base in val_set:
        copy_pair(base, "validation")

    for base in test_set:
        copy_pair(base, "test")

    print(f"Total instances: {n}")
    print(f"Train: {len(train_set)}, Validation: {len(val_set)}, Test: {len(test_set)}")
    print(f"Data written to: {new_data_folder}")




# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(
    version_base=None,
    config_path="configs/data_generation",
    config_name="config",
)


def main(data_generation_config: DictConfig) -> None:
    """Hydra entry point: dispatches to the configured sample-generation routine."""


    instance_type = data_generation_config.instance_type

    seed = data_generation_config.seed
    HGS_runtime = data_generation_config.HGS_runtime

    if instance_type == "CVRP":
        print("here CVRP")
        data_path = data_generation_config.data_path
        nb_clients = data_generation_config.nb_clients
        nb_instances = data_generation_config.nb_instances
        save_path = data_generation_config.save_path
        instance_path = generate_CVRP_LIB_instances(data_path, nb_clients, seed, nb_instances)
        samples_generation_CVRP_literature(instance_path, save_path, HGS_time_limit=HGS_runtime)
   
    elif instance_type == "CVRPTW":
        print("here CVRPTW")
        instances_path = data_generation_config.instances_path
        new_instance_path = data_generation_config.new_instance_path
        width_min_prob_large = data_generation_config.width_min_prob_large
        width_max_prob_large =  data_generation_config.width_max_prob_large
        width_min_prob_narrow = data_generation_config.width_min_prob_narrow
        width_max_prob_narrow = data_generation_config.width_max_prob_narrow
        speed = data_generation_config.speed
        service_time = data_generation_config.service_time
        start = data_generation_config.start
        end = data_generation_config.end
        ready_time_init = data_generation_config.ready_time_init
        solve_exact = data_generation_config.solve_exact
        exact_time_limit = data_generation_config.exact_time_limit

        # generate_cvrptw_restricted(instances_path, new_instance_path,
        # width_min_prob_large=width_min_prob_large, width_max_prob_large=width_max_prob_large,
        # width_min_prob_narrow=width_min_prob_narrow, width_max_prob_narrow=width_max_prob_narrow,
        # speed = speed, service_time=service_time, seed=seed, HGS_runtime=HGS_runtime,
        # start=start, end=end, ready_time_init=ready_time_init, solve_exact=solve_exact,
        # exact_time_limit=exact_time_limit)

        # width_min_prob_large_list = [0.07, 0.1, 0.2, 0.3]
        # width_max_prob_large_list = [0.09, 0.2, 0.3, 0.4]
        # width_min_prob_narrow_list = [0.01, 0.02, 0.03, 0.05]
        # width_max_prob_narrow_list = [0.02, 0.03, 0.05, 0.07]
        
        width_min_prob_large_list = [width_min_prob_large]
        width_max_prob_large_list = [width_max_prob_large]
        width_min_prob_narrow_list = [width_min_prob_narrow]
        width_max_prob_narrow_list = [width_max_prob_narrow]



        comparison_dict = comparison_CVRPTW_generation(instances_path, new_instance_path,
        width_min_prob_large_list=width_min_prob_large_list,
        width_max_prob_large_list=width_max_prob_large_list,
        width_min_prob_narrow_list=width_min_prob_narrow_list,
        width_max_prob_narrow_list=width_max_prob_narrow_list,
        speed = speed, service_time=service_time, seed=seed, HGS_runtime=HGS_runtime,
        start=start, end=end, ready_time_init=ready_time_init, solve_exact=solve_exact,
        exact_time_limit=exact_time_limit)

        # plot_CVRPTW_comparison(comparison_dict=comparison_dict)
    
    else:
        raise ValueError



# --- Main: generate a few samples ---
if __name__ == "__main__":
    main()
