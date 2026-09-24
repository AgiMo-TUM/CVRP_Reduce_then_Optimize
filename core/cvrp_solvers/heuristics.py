"""CVRP heuristics."""

import contextlib
import io
import logging
import os
import re
from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache
from time import time

import numpy as np
import yaml

from pyvrp import Model as HGS_Model
from pyvrp.stop import MaxRuntime as HGS_MaxRuntime
# `PyVRP` (capital) is the "new" backend; loaded lazily inside `_*_new`
# functions so the module imports even when only the old `pyvrp` is installed.




##Changes

def Clark_Wright_heuristic(demands, arc_index, arc_costs, nb_vehicles, vehicle_capacity):
    """Clarke & Wright savings heuristic — returns a feasible CVRP route assignment."""
    n_nodes = len(demands) # We need to take care of the depot
    client_routing_list = [None] * n_nodes #It will contain both integers and list of integers
    client_routing_left_capacity = [vehicle_capacity- demand for demand in demands]
    nb_available_vehicles = nb_vehicles
    depot_index = 0

    for i in range(len(demands)):
        if(demands[i]==0):
            depot_index = i
            break

    arc_list = [(int(src), int(dst)) for src, dst in zip(arc_index[0], arc_index[1])]

    cost_dict = {(int(src), int(dst)): arc_costs[k]
             for k, (src, dst) in enumerate(zip(arc_index[0], arc_index[1]))}
    
    savings_dict = {}
    for arc in arc_list:
        if(arc[0]!=depot_index and arc[1]!= depot_index):
            saving = (cost_dict[(depot_index, arc[0])] +
                  cost_dict[(depot_index, arc[1])] -
                  cost_dict[(arc[0], arc[1])])
            savings_dict[arc] = saving


    sorted_savings = sorted(savings_dict.items(), key=lambda x: x[1], reverse=True)

    for arc, saving in sorted_savings:

        if(client_routing_list[arc[0]] == None and client_routing_list[arc[1]] == None
          and nb_available_vehicles > 0 
          and demands[arc[0]] + demands[arc[1]] <= client_routing_left_capacity[arc[0]]):
            
            
            client_routing_list[arc[0]] = [arc[0], arc[1]]
            nb_available_vehicles -= 1
            client_routing_left_capacity[arc[0]] -= demands[arc[1]]

            client_routing_list[arc[1]] = arc[0]

        elif(isinstance(client_routing_list[arc[0]], int)
            and client_routing_list[arc[1]] == None):

                
            if([client_routing_list[client_routing_list[arc[0]]]][-1] == arc[0]
                and client_routing_left_capacity[client_routing_list[arc[0]]] >= demands[arc[1]]):
                client_routing_list[client_routing_list[arc[0]]].append(arc[1])
                client_routing_left_capacity[client_routing_list[arc[0]]] -= demands[arc[1]]

                client_routing_list[arc[1]] = arc[0]

        elif(isinstance(client_routing_list[arc[1]], list) 
             and client_routing_list[arc[0]] == None
             and client_routing_left_capacity[arc[1]] >= demands[arc[0]]):

            client_routing_list[arc[0]] = [arc[0], arc[1]]

            for i in range (1, len(client_routing_list[arc[1]])):
                client_routing_list[arc[0]].append(client_routing_list[arc[1]][i])

                client_routing_list[client_routing_list[arc[1]][i]] = arc[0]

            client_routing_left_capacity[arc[0]] = client_routing_left_capacity[arc[1]] - demands[arc[0]]
            client_routing_left_capacity[arc[1]] = vehicle_capacity - demands[arc[1]]

            client_routing_list[arc[1]]  = arc[0]

        elif(isinstance(client_routing_list[arc[0]], int)):

            if(client_routing_list[arc[0]] != arc[1]):

                if(client_routing_list[client_routing_list[arc[0]]][-1] == arc[0]
                and isinstance(client_routing_list[arc[1]], list)
                and client_routing_left_capacity[client_routing_list[arc[0]]] >=
                vehicle_capacity - client_routing_left_capacity[arc[1]]):
                    
                    client_routing_list[client_routing_list[arc[0]]].append(arc[1])
                    
                    for i in range(1, len(client_routing_list[arc[1]])):
                        client_routing_list[client_routing_list[arc[0]]].append(
                            client_routing_list[arc[1]][i])

                        client_routing_list[client_routing_list[arc[1]][i]] = client_routing_list[arc[0]]

                    
                    client_routing_left_capacity[client_routing_list[arc[0]]] -= (vehicle_capacity - client_routing_left_capacity[arc[1]])
                    client_routing_left_capacity[arc[1]] = vehicle_capacity - demands[arc[1]]

                    client_routing_list[arc[1]]  = client_routing_list[arc[0]]
                    nb_available_vehicles += 1

    
    for i in range(len(demands)):
        if client_routing_list[i] == None and i != depot_index:
            if(nb_available_vehicles > 0):
                client_routing_list[i] = [i]
                nb_available_vehicles -= 1
            else:
                raise RuntimeError("Not enough vehicles to serve all clients")

    sol_dict = {(int(src), int(dst)): 0
             for k, (src, dst) in enumerate(zip(arc_index[0], arc_index[1]))}
    for i in range(len(demands)):
        if(isinstance(client_routing_list[i], list)):
            sol_dict[(depot_index, client_routing_list[i][0])] = 1
            sol_dict[(client_routing_list[i][-1], depot_index)] = 1
            if len(client_routing_list[i]) >= 2:
                for j in range(1, len(client_routing_list[i])):
                    sol_dict[(client_routing_list[i][j-1], client_routing_list[i][j])] = 1
    
    return sol_dict

# ---------------------------------------------------------------------------
# HGS solvers (PyVRP)
#
# Two PyVRP versions are supported and live side-by-side in this module:
#   - "old": the legacy `pyvrp` package (positional `add_depot(x=, y=)` API).
#   - "new": the `PyVRP.pyvrp` package (location-based API).
#
# The version is chosen via `pyvrp_version` in `configs/training/config.yaml`.
# Public dispatchers (`heu_solve_HGS_VRP`, `heu_solve_HGS_VRPTW`) read that
# value and forward to the matching private implementation. The dispatcher
# signatures intentionally match the OLD-style positional calls used across
# the codebase; new-only inputs (e.g. `nodes`) are accepted as kwargs.
# ---------------------------------------------------------------------------


_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs",
    "training",
    "config.yaml",
)


@lru_cache(maxsize=1)
def _get_pyvrp_version():
    """Read `pyvrp_version` ("old"/"new") from the training config."""
    try:
        with open(_CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("pyvrp_version", "old")
    except Exception:
        return "old"


def _extract_improvements_old(log_text):
    """Parse OLD-PyVRP progress log lines into ``(iter, time_str, best_cost)`` tuples."""
    improvements = []
    for line in log_text.splitlines():
        line = line.strip()
        if not line.startswith("H"):
            continue
        iter_match = re.search(r"H\s+(\d+)", line)
        iteration = int(iter_match.group(1)) if iter_match else None
        time_match = re.search(r"\b(\d+s)\b", line)
        time_str = time_match.group(1) if time_match else None
        feas_match = re.search(r"\|\s*\d+\s+\d+\s+(\d+)\s*\|", line)
        best_cost = int(feas_match.group(1)) if feas_match else None
        improvements.append((iteration, time_str, best_cost))
    return improvements


def _extract_improvements_new(log_text):
    """Parse NEW-PyVRP progress log lines into ``(iter, time_str, best_cost)`` tuples."""
    improvements = []
    for line in log_text.splitlines():
        line = line.strip()
        if not line.startswith("H"):
            continue
        iter_match = re.search(r"H\s+(\d+)", line)
        if not iter_match:
            continue
        iteration = int(iter_match.group(1))
        time_match = re.search(r"\b(\d+s)\b", line)
        time_str = time_match.group(1) if time_match else None
        cost_matches = re.findall(r"(\d+)\s+Y", line)
        if not cost_matches:
            continue
        best_cost = int(cost_matches[-1])
        improvements.append((iteration, time_str, best_cost))
    return improvements


@contextmanager
def _capture_pyvrp_logs():
    """Context manager that redirects PyVRP's progress logger into a StringIO buffer."""
    buffer = io.StringIO()
    logger = logging.getLogger("PyVRP.pyvrp.ProgressPrinter")
    old_handlers = logger.handlers[:]
    old_propagate = logger.propagate
    handler = logging.StreamHandler(buffer)
    handler.setLevel(logging.INFO)
    logger.handlers = [handler]
    logger.propagate = False
    try:
        yield buffer
    finally:
        logger.handlers = old_handlers
        logger.propagate = old_propagate


# ---------------------------------------------------------------------------
# OLD (pyvrp): CVRP
# ---------------------------------------------------------------------------
def _heu_solve_HGS_VRP_old(
    demands,
    arc_index,
    arc_costs,
    nb_vehicles,
    vehicle_capacity,
    relevant_connections=None,
    heu_time=100,
    undirected=True,
    arc_likelihood=None,
    instance_log_HGS_dict=None,
    threshold=None,
):
    """Solve CVRP with the OLD PyVRP HGS backend on the reduced arc set."""
    start_time = time()
    for i in range(len(demands)):
        if demands[i] == 0:
            depot_index = i
            break

    arc_list = [(int(src), int(dst)) for src, dst in zip(arc_index[0], arc_index[1])]
    if relevant_connections is None:
        relevant_connections = [True for arc in arc_list]

    m = HGS_Model()
    m.add_vehicle_type(capacity=vehicle_capacity, num_available=nb_vehicles)
    total_nodes_dict = {}
    depot = m.add_depot(x=0, y=0)
    total_nodes_dict[depot_index] = depot
    for i in range(len(demands)):
        if demands[i] != 0:
            total_nodes_dict[i] = m.add_client(x=0, y=0, delivery=int(demands[i]))
    if arc_likelihood is None:
        for i, arc in enumerate(arc_list):
            m.add_edge(
                total_nodes_dict[arc[0]],
                total_nodes_dict[arc[1]],
                distance=arc_costs[i],
                duration=(1 - relevant_connections[i]) * 4e8,
            )
            if undirected:
                m.add_edge(
                    total_nodes_dict[arc[1]],
                    total_nodes_dict[arc[0]],
                    distance=arc_costs[i],
                    duration=(1 - relevant_connections[i]) * 4e8,
                )
    else:
        for i, arc in enumerate(arc_list):
            m.add_edge(
                total_nodes_dict[arc[0]],
                total_nodes_dict[arc[1]],
                distance=arc_costs[i],
                duration=1 / arc_likelihood[i] * 100,
            )
            if undirected:
                m.add_edge(
                    total_nodes_dict[arc[1]],
                    total_nodes_dict[arc[0]],
                    distance=arc_costs[i],
                    duration=1 / arc_likelihood[i] * 100,
                )

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        res = m.solve(stop=HGS_MaxRuntime(heu_time), display=True)

    build_solver_runtime = time() -start_time
    output = buffer.getvalue()

    if threshold is not None:
        improvements = _extract_improvements_old(output)
        instance_log_HGS_dict[threshold] = improvements
        print(output)
        print(improvements)

    sol_dict = {
        (int(src), int(dst)): 0
        for (src, dst) in zip(arc_index[0], arc_index[1])
    }
    for i in range(len(res.best.routes())):
        if undirected:
            sol_dict[(depot_index, res.best.routes()[i].visits()[0])] = 1
            sol_dict[(depot_index, res.best.routes()[i].visits()[-1])] = 1
            if len(res.best.routes()[i].visits()) >= 2:
                for j in range(1, len(res.best.routes()[i].visits())):
                    if (
                        res.best.routes()[i].visits()[j - 1]
                        < res.best.routes()[i].visits()[j]
                    ):
                        sol_dict[
                            (
                                res.best.routes()[i].visits()[j - 1],
                                res.best.routes()[i].visits()[j],
                            )
                        ] = 1
                    else:
                        sol_dict[
                            (
                                res.best.routes()[i].visits()[j],
                                res.best.routes()[i].visits()[j - 1],
                            )
                        ] = 1
        else:
            sol_dict[(depot_index, res.best.routes()[i].visits()[0])] = 1
            sol_dict[(res.best.routes()[i].visits()[-1], depot_index)] = 1
            if len(res.best.routes()[i].visits()) >= 2:
                for j in range(1, len(res.best.routes()[i].visits())):
                    sol_dict[
                        (
                            res.best.routes()[i].visits()[j - 1],
                            res.best.routes()[i].visits()[j],
                        )
                    ] = 1
    print("objective value with HGS : ", res.cost())
    return sol_dict, res.cost(), res.runtime, build_solver_runtime, res.best.is_feasible()


# ---------------------------------------------------------------------------
# OLD (pyvrp): CVRPTW
# ---------------------------------------------------------------------------
def _heu_solve_HGS_VRPTW_old(
    cvrptw_nodes,
    arc_index,
    arc_costs,
    nb_vehicles,
    vehicle_capacity,
    relevant_connections,
    heu_time=2,
    undirected=False,
    true_time_cost=None,
):
    """Solve CVRPTW with the OLD PyVRP HGS backend on the reduced arc set."""
    start_time = time()
    demands = [node.demand for node in cvrptw_nodes]
    depot_index = next(i for i, d in enumerate(demands) if d == 0)

    arc_list = [(int(src), int(dst)) for src, dst in zip(arc_index[0], arc_index[1])]

    m = HGS_Model()
    print("cap : ", vehicle_capacity)
    print("nb vehicles : ", nb_vehicles)
    m.add_vehicle_type(capacity=vehicle_capacity, num_available=nb_vehicles)

    total_nodes = {}

    depot_node = cvrptw_nodes[depot_index]
    depot = m.add_depot(
        x=0,
        y=0,
        tw_early=int(depot_node.ready_time),
        tw_late=int(depot_node.due_time),
    )
    total_nodes[depot_index] = depot

    for i, node in enumerate(cvrptw_nodes):
        if i == depot_index:
            continue
        total_nodes[i] = m.add_client(
            x=0,
            y=0,
            delivery=int(node.demand),
            tw_early=int(node.ready_time),
            tw_late=int(node.due_time),
            service_duration=int(node.service_time),
        )

    for i, arc in enumerate(arc_list):
        if relevant_connections[i]:
            if true_time_cost is None:
                m.add_edge(
                    total_nodes[arc[0]],
                    total_nodes[arc[1]],
                    distance=arc_costs[i],
                    duration=arc_costs[i],
                )
            else:
                m.add_edge(
                    total_nodes[arc[0]],
                    total_nodes[arc[1]],
                    distance=arc_costs[i],
                    duration=true_time_cost[i],
                )
        if undirected:
            m.add_edge(
                total_nodes[arc[1]],
                total_nodes[arc[0]],
                distance=arc_costs[i],
                duration=arc_costs[i],
            )

    res = m.solve(stop=HGS_MaxRuntime(heu_time), display=False)
    build_solver_runtime = time() - start_time

    sol = res.best
    print("Feasible:", sol.is_feasible())

    sol_dict = {
        (int(src), int(dst)): 0
        for (src, dst) in zip(arc_index[0], arc_index[1])
    }
    for (u, v) in zip(arc_index[0], arc_index[1]):
        sol_dict[(v, u)] = 0

    for route in res.best.routes():
        visits = route.visits()
        sol_dict[(depot_index, visits[0])] = 1
        sol_dict[(visits[-1], depot_index)] = 1
        for u, v in zip(visits[:-1], visits[1:]):
            sol_dict[(u, v)] = 1

    print("objective value with HGS:", res.cost())
    return sol_dict, res.cost(), res.runtime, sol.is_feasible(), build_solver_runtime


# ---------------------------------------------------------------------------
# NEW (PyVRP.pyvrp): CVRP
# ---------------------------------------------------------------------------
def _heu_solve_HGS_VRP_new(
    nodes,
    demands,
    arc_index,
    arc_costs,
    nb_vehicles,
    vehicle_capacity,
    relevant_connections,
    heu_time=2,
    undirected=True,
    relevant_connections_1=None,
    arc_likelihood=None,
    instance_log_HGS_dict=None,
    threshold=None,
):
    """Solve CVRP with the NEW PyVRP HGS backend on the reduced arc set."""
    from PyVRP.pyvrp import Model as new_HGS_Model
    from PyVRP.pyvrp.stop import MaxRuntime as new_HGS_MaxRuntime

    start_time = time()

    for i in range(len(demands)):
        if demands[i] == 0:
            depot_index = i
            break

    arc_list = [(int(src), int(dst)) for src, dst in zip(arc_index[0], arc_index[1])]

    m = new_HGS_Model()
    m.add_vehicle_type(capacity=vehicle_capacity, num_available=nb_vehicles)
    total_nodes_dict = {}
    depot_location = m.add_location(nodes[0].x, nodes[0].y)
    m.add_depot(location=depot_location, name="Depot")
    total_nodes_dict[depot_index] = depot_location
    for i in range(len(demands)):
        if demands[i] != 0:
            client_location = m.add_location(nodes[i].x, nodes[i].y)
            total_nodes_dict[i] = client_location
            m.add_client(
                location=client_location,
                delivery=int(demands[i]),
                name=f"Client {i + 1}",
            )

    for i, arc in enumerate(arc_list):
        if relevant_connections_1 is None:
            m.add_edge(
                total_nodes_dict[arc[0]],
                total_nodes_dict[arc[1]],
                distance=arc_costs[i],
                duration=(1 - relevant_connections[i]) * 4e13,
            )
            if undirected:
                m.add_edge(
                    total_nodes_dict[arc[1]],
                    total_nodes_dict[arc[0]],
                    distance=arc_costs[i],
                    duration=(1 - relevant_connections[i]) * 4e13,
                )
        else:
            if relevant_connections_1[i]:
                m.add_edge(
                    total_nodes_dict[arc[0]],
                    total_nodes_dict[arc[1]],
                    distance=arc_costs[i],
                    duration=(1 - relevant_connections[i]) * 4e13,
                )
                if undirected:
                    m.add_edge(
                        total_nodes_dict[arc[1]],
                        total_nodes_dict[arc[0]],
                        distance=arc_costs[i],
                        duration=(1 - relevant_connections[i]) * 4e13,
                    )

    with _capture_pyvrp_logs() as buf:
        res = m.solve(stop=new_HGS_MaxRuntime(heu_time), display=True)

    build_solver_runtime = time()-start_time
    output = buf.getvalue()

    if threshold is not None:
        improvements = _extract_improvements_new(output)
        instance_log_HGS_dict[threshold] = improvements
        print(output)
        print(improvements)

    sol_dict = {
        (int(src), int(dst)): 0
        for (src, dst) in zip(arc_index[0], arc_index[1])
    }
    for i in range(len(res.best.routes())):
        visits = [act.idx for act in res.best.routes()[i] if act.idx != 0]
        if undirected and len(visits) > 0:
            sol_dict[(depot_index, visits[0])] = 1
            sol_dict[(depot_index, visits[-1])] = 1
            if len(visits) >= 2:
                for j in range(1, len(visits)):
                    if visits[j - 1] < visits[j]:
                        sol_dict[(visits[j - 1], visits[j])] = 1
                    else:
                        sol_dict[(visits[j], visits[j - 1])] = 1
        else:
            if len(visits) > 0:
                sol_dict[(depot_index, visits[0])] = 1
                sol_dict[(visits[-1], depot_index)] = 1
                if len(visits) >= 2:
                    for j in range(1, len(visits)):
                        sol_dict[(visits[j - 1], visits[j])] = 1
    print("objective value with HGS : ", res.cost())
    return sol_dict, res.cost(), res.runtime, res.best.is_feasible(), build_solver_runtime


# ---------------------------------------------------------------------------
# NEW (PyVRP.pyvrp): CVRPTW
# ---------------------------------------------------------------------------
def _heu_solve_HGS_VRPTW_new(
    cvrptw_nodes,
    arc_index,
    arc_costs,
    vehicle_capacity,
    nb_vehicles,
    relevant_connections,
    heu_time=2,
    arc_likelihood=None,
    instance_log_HGS_dict=None,
    threshold=None,
    true_time_costs=None,
):
    """Solve CVRPTW with the NEW PyVRP HGS backend on the reduced arc set."""
    from PyVRP.pyvrp import Model as new_HGS_Model
    from PyVRP.pyvrp.stop import MaxRuntime as new_HGS_MaxRuntime

    start_time = time()

    demands = [node.demand for node in cvrptw_nodes]
    depot_index = next(i for i, d in enumerate(demands) if d == 0)

    arc_list = [(int(src), int(dst)) for src, dst in zip(arc_index[0], arc_index[1])]

    m = new_HGS_Model()
    print("cap : ", vehicle_capacity)
    print("nb vehicles : ", nb_vehicles)
    m.add_vehicle_type(capacity=vehicle_capacity, num_available=nb_vehicles)

    total_nodes_dict = {}
    depot_location = m.add_location(cvrptw_nodes[0].x, cvrptw_nodes[0].y)
    m.add_depot(
        location=depot_location,
        name="Depot",
        tw_early=cvrptw_nodes[0].ready_time,
        tw_late=cvrptw_nodes[0].due_time,
    )
    total_nodes_dict[depot_index] = depot_location

    for i in range(len(demands)):
        if demands[i] != 0:
            client_location = m.add_location(cvrptw_nodes[i].x, cvrptw_nodes[i].y)
            total_nodes_dict[i] = client_location
            m.add_client(
                location=client_location,
                delivery=int(demands[i]),
                name=f"Client {i + 1}",
                tw_early=int(cvrptw_nodes[i].ready_time),
                tw_late=int(cvrptw_nodes[i].due_time),
                service_duration=int(cvrptw_nodes[i].service_time),
            )

    for i, arc in enumerate(arc_list):
        if true_time_costs is None:
            m.add_edge(
                total_nodes_dict[arc[0]],
                total_nodes_dict[arc[1]],
                distance=arc_costs[i],
                duration=arc_costs[i] + (1 - relevant_connections[i]) * 1e9,
            )
        else:
            m.add_edge(
                total_nodes_dict[arc[0]],
                total_nodes_dict[arc[1]],
                distance=arc_costs[i],
                duration=true_time_costs[i],
            )

    with _capture_pyvrp_logs() as buf:
        res = m.solve(stop=new_HGS_MaxRuntime(heu_time), display=True)

    build_solver_runtime = time()-start_time
    output = buf.getvalue()

    if threshold is not None:
        improvements = _extract_improvements_new(output)
        instance_log_HGS_dict[threshold] = improvements
        print(output)
        print(improvements)

    sol = res.best
    print("Feasible:", sol.is_feasible())

    sol_dict = {
        (int(src), int(dst)): 0
        for (src, dst) in zip(arc_index[0], arc_index[1])
    }

    for i in range(len(res.best.routes())):
        visits = [act.idx for act in res.best.routes()[i] if act.idx != 0]
        sol_dict[(depot_index, visits[0])] = 1
        sol_dict[(visits[-1], depot_index)] = 1
        if len(visits) >= 2:
            for j in range(1, len(visits)):
                sol_dict[(visits[j - 1], visits[j])] = 1
    print("objective value with HGS : ", res.cost())
    return sol_dict, res.cost(), res.runtime, sol.is_feasible(), build_solver_runtime


# ---------------------------------------------------------------------------
# Public dispatchers — version selected via config.yaml `pyvrp_version`
# ---------------------------------------------------------------------------
def heu_solve_HGS_VRP(
    demands,
    arc_index,
    arc_costs,
    nb_vehicles,
    vehicle_capacity,
    relevant_connections,
    heu_time=100,
    undirected=True,
    arc_likelihood=None,
    instance_log_HGS_dict=None,
    threshold=None,
    nodes=None,
    relevant_connections_1=None,
    pyvrp_version=None,
):
    """Dispatch CVRP HGS solve to the configured PyVRP version.

    The signature follows the OLD-style positional API used across the
    codebase. New-only inputs (`nodes`, `relevant_connections_1`) are kwargs;
    `nodes` is required when `pyvrp_version="new"`. If ``pyvrp_version`` is
    None, the value is read from the benchmarking/training config.
    """
    version = pyvrp_version if pyvrp_version is not None else _get_pyvrp_version()
    if version == "new":
        if nodes is None:
            raise ValueError(
                "heu_solve_HGS_VRP: `nodes` is required for pyvrp_version='new'."
            )
        return _heu_solve_HGS_VRP_new(
            nodes=nodes,
            demands=demands,
            arc_index=arc_index,
            arc_costs=arc_costs,
            nb_vehicles=nb_vehicles,
            vehicle_capacity=vehicle_capacity,
            relevant_connections=relevant_connections,
            heu_time=heu_time,
            undirected=undirected,
            relevant_connections_1=relevant_connections_1,
            arc_likelihood=arc_likelihood,
            instance_log_HGS_dict=instance_log_HGS_dict,
            threshold=threshold,
        )
    return _heu_solve_HGS_VRP_old(
        demands=demands,
        arc_index=arc_index,
        arc_costs=arc_costs,
        nb_vehicles=nb_vehicles,
        vehicle_capacity=vehicle_capacity,
        relevant_connections=relevant_connections,
        heu_time=heu_time,
        undirected=undirected,
        arc_likelihood=arc_likelihood,
        instance_log_HGS_dict=instance_log_HGS_dict,
        threshold=threshold,
    )


def heu_solve_HGS_VRPTW(
    cvrptw_nodes,
    arc_index,
    arc_costs,
    nb_vehicles,
    vehicle_capacity,
    relevant_connections,
    heu_time=2,
    undirected=False,
    true_time_cost=None,
    arc_likelihood=None,
    instance_log_HGS_dict=None,
    threshold=None,
    pyvrp_version=None,
):
    """Dispatch CVRPTW HGS solve to the configured PyVRP version.

    The dispatcher signature uses the OLD positional convention
    (`nb_vehicles, vehicle_capacity`); for the NEW backend the order is
    swapped internally to match its API. If ``pyvrp_version`` is None, the
    value is read from the benchmarking/training config.
    """
    version = pyvrp_version if pyvrp_version is not None else _get_pyvrp_version()
    if version == "new":
        return _heu_solve_HGS_VRPTW_new(
            cvrptw_nodes=cvrptw_nodes,
            arc_index=arc_index,
            arc_costs=arc_costs,
            vehicle_capacity=vehicle_capacity,
            nb_vehicles=nb_vehicles,
            relevant_connections=relevant_connections,
            heu_time=heu_time,
            arc_likelihood=arc_likelihood,
            instance_log_HGS_dict=instance_log_HGS_dict,
            threshold=threshold,
            true_time_costs=true_time_cost,
        )
    return _heu_solve_HGS_VRPTW_old(
        cvrptw_nodes=cvrptw_nodes,
        arc_index=arc_index,
        arc_costs=arc_costs,
        nb_vehicles=nb_vehicles,
        vehicle_capacity=vehicle_capacity,
        relevant_connections=relevant_connections,
        heu_time=heu_time,
        undirected=undirected,
        true_time_cost=true_time_cost,
    )

