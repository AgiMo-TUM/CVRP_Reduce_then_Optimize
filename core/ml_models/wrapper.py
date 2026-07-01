"""Wrapper and helper functions for ML-based CVRP / CVRPTW reduce-then-optimize."""

import logging
from time import time

import numpy as np
import torch
from torch_geometric.data import Data

from core.cvrp_solvers.heuristics import heu_solve_HGS_VRP
from core.cvrp_solvers.heuristics import heu_solve_HGS_VRPTW
from core.cvrp_solvers.ip_grb import cvrp_via_VRP_Easy
from core.utils.standard_reduction import standard_pruning


def _import_cvrpTW_via_VRP_Easy():
    """Lazy import; only loaded when ``is_time_windows`` decoder is requested."""
    from core.cvrp_solvers.ip_grb import cvrpTW_via_VRP_Easy  # noqa: WPS433
    return cvrpTW_via_VRP_Easy


logger = logging.getLogger()


def get_max_likelihood_sol(
    instance,
    relevant_connections,
    prediction,
    completion_heu_time,
    is_time_windows=False,
    pyvrp_version=None,
):
    """Get a feasible solution that greedily maximises total arc likelihood.

    Costs are crafted from the predicted arc likelihoods so that arcs in
    ``relevant_connections`` are cheap and the rest are heavily penalised.
    HGS is then run on the full arc set with a small time budget to recover
    a feasible tour.

    Parameters
    ----------
    instance : CVRP instance.
    relevant_connections : 1D iterable of bool, length = num_arcs.
    prediction : 1D iterable of float, length = num_arcs.
    completion_heu_time : int. HGS budget in seconds. Caller must ensure > 0.
    is_time_windows : bool. If True, dispatch to the CVRPTW HGS variant.

    Returns
    -------
    sol : dict. Arc-indexed solution from HGS.
    completion_runtime : float. HGS runtime.
    """
    handmade_costs = [0.0] * len(relevant_connections)
    for i, kept in enumerate(relevant_connections):
        base = 1.0 / (prediction[i] + 0.001)
        handmade_costs[i] = base if kept else 100.0 * base

    all_connections = [True] * len(handmade_costs)

    if is_time_windows:
        sol, _, completion_runtime = heu_solve_HGS_VRPTW(
            instance.nodes,
            instance.arc_index,
            handmade_costs,
            instance.nb_vehicles,
            instance.vehicle_capacity,
            all_connections,
            heu_time=completion_heu_time,
            undirected=False,
            true_time_cost=instance.arc_costs,
            pyvrp_version=pyvrp_version,
        )
    else:
        sol, _, completion_runtime = heu_solve_HGS_VRP(
            instance.demands,
            instance.arc_index,
            handmade_costs,
            instance.nb_vehicles,
            instance.vehicle_capacity,
            all_connections,
            heu_time=completion_heu_time,
            undirected=True,
            pyvrp_version=pyvrp_version,
        )
    return sol, completion_runtime


def sol_arc_predictor_wrapper(instance, predictor_model, cached_features=None):
    """Run a trained arc predictor on a single instance and return likelihoods.

    Parameters
    ----------
    instance : CVRP instance.
    predictor_model : tuple (model, feature_fun).
    cached_features : optional precomputed (node_feat, edge_attr, edge_index).

    Returns
    -------
    preds : np.ndarray of arc likelihoods aligned with ``arc_index``.
    arc_index : 2 x E np.ndarray of source/destination indices.
    """
    model, feature_fun = predictor_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cached_features is None:
        node_feat, edge_attr, edge_index, _, _ = feature_fun(instance)
    else:
        node_feat, edge_attr, edge_index = cached_features

    node_feat = torch.as_tensor(node_feat, device=device)
    edge_attr = torch.as_tensor(edge_attr, device=device)
    edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=device)

    data = Data(x=node_feat, edge_index=edge_index, edge_attr=edge_attr)

    with torch.no_grad():
        preds = model.predict_arcs(data, train=False)
    if isinstance(preds, tuple):
        _, preds = preds

    preds = preds.cpu().numpy()
    arc_index = edge_index.cpu().numpy()
    return preds, arc_index


def _select_relevant_arcs(instance, arc_likelihood, threshold_type, threshold):
    """Select the arc subset implied by ``threshold_type`` and ``threshold``.

    - ``"top_k"``: keep, for each node, its top-k highest-likelihood arcs
      (symmetrised). ``threshold`` is interpreted as the integer K.
    - ``"size"``: keep the top ``threshold`` fraction of arcs by likelihood.
    - ``"probability"``: keep arcs whose normalised likelihood exceeds
      ``threshold``.
    """
    arc_likelihood_flat = np.asarray(arc_likelihood).reshape(-1)

    if threshold_type == "top_k":
        num_nodes = len(instance.nodes)
        adj = np.full((num_nodes, num_nodes), -np.inf)
        for idx, (u, v) in enumerate(instance.arc_list):
            adj[u, v] = arc_likelihood_flat[idx]
            adj[v, u] = arc_likelihood_flat[idx]

        K = int(threshold)
        K = min(K, num_nodes - 1)
        topk_indices = np.argpartition(adj, -K, axis=1)[:, -K:]
        mask = np.zeros_like(adj, dtype=bool)
        for u in range(num_nodes):
            row_idx = topk_indices[u]
            mask[u, row_idx] = True
            mask[row_idx, u] = True
        return np.array([mask[u, v] for (u, v) in instance.arc_list])

    if threshold_type == "size":
        cutoff = np.quantile(arc_likelihood_flat, 1.0 - float(threshold))
        return arc_likelihood_flat >= cutoff

    if threshold_type == "probability":
        denom = np.max(arc_likelihood_flat) - np.min(arc_likelihood_flat) + 1e-8
        norm = (arc_likelihood_flat - np.min(arc_likelihood_flat)) / denom
        return norm >= float(threshold)

    raise ValueError(f"Unknown threshold_type: {threshold_type}")


def get_reduced_problem(
    instance,
    predictor_model,
    threshold_type="top_k",
    threshold=20,
    cached_features=None,
    completion_heu_time=0,
    is_time_windows=False,
    pyvrp_version=None,
):
    """Run the predictor and return the reduced arc set.

    If ``completion_heu_time > 0``, augment the reduced set with arcs from a
    feasibility heuristic (``get_max_likelihood_sol``).
    """
    start = time()
    arc_likelihood, _ = sol_arc_predictor_wrapper(
        instance, predictor_model, cached_features=cached_features
    )
    inference_runtime = time() - start

    relevant_connections = _select_relevant_arcs(
        instance, arc_likelihood, threshold_type, threshold
    )

    completion_runtime = 0.0
    if completion_heu_time > 0:
        arc_index_map = {
            (int(instance.arc_index[0, idx]), int(instance.arc_index[1, idx])): idx
            for idx in range(instance.arc_index.shape[1])
        }
        greedy_sol, completion_runtime = get_max_likelihood_sol(
            instance,
            relevant_connections,
            np.asarray(arc_likelihood).reshape(-1),
            completion_heu_time=completion_heu_time,
            is_time_windows=is_time_windows,
            pyvrp_version=pyvrp_version,
        )
        for arc, val in greedy_sol.items():
            if val > 0 and arc in arc_index_map:
                relevant_connections[arc_index_map[arc]] = True
    
    else:
        for k, (u,v) in enumerate(zip(instance.arc_index[0], instance.arc_index[1])): #fast feasibility step
            if u==0 or v==0:                                                           #with unlimited number of vehicles
                relevant_connections[k] = True

    num_arcs_pred = int(np.sum(relevant_connections))
    num_arcs_enriched = num_arcs_pred
    total_inference_time = time() - start
    logger.info("Inference_runtime = %s", total_inference_time)

    return (
        relevant_connections,
        (num_arcs_pred, num_arcs_enriched),
        completion_runtime,
        inference_runtime,
    )


def solve_reduced_problem(
    instance,
    relevant_connections,
    decoder="hgs",
    decoder_cfg=None,
    heu_time=100,
    time_limit=5000,
    arc_likelihood=None,
    threshold=0,
    instance_log_HGS_dict=None,
    is_time_windows=False,
    pyvrp_version="old",
):
    """Solve the reduced problem with either the exact (VRP-Easy) decoder or HGS.

    Parameters
    ----------
    decoder : "exact" -> branch-and-cut via VRP-Easy; "hgs" -> PyVRP HGS.
    is_time_windows : selects the CVRPTW variant of the decoder.
    pyvrp_version : passed through to the HGS dispatcher (consumed via config).
    heu_time : HGS time budget in seconds (only used for ``decoder == "hgs"``).
    time_limit : exact-solver time limit in seconds.
    """
    if decoder_cfg is None:
        decoder_cfg = {}

    status = None
    lower_bound = None

    if decoder == "exact":
        if is_time_windows:
            cvrpTW_via_VRP_Easy = _import_cvrpTW_via_VRP_Easy()
            solution, runtime, solver_value, lower_bound, status, build_solver_runtime = cvrpTW_via_VRP_Easy(
                instance.nodes,
                instance.arc_index,
                instance.arc_costs,
                instance.nb_vehicles,
                instance.vehicle_capacity,
                relevant_connections,
                False,
                time_limit=time_limit
            )
        else:
            solution, runtime, solver_value, lower_bound, status, build_solver_runtime = cvrp_via_VRP_Easy(
                instance.demands,
                instance.arc_index,
                instance.arc_costs,
                instance.nb_vehicles,
                instance.vehicle_capacity,
                relevant_connections,
                time_limit=time_limit
            )
    elif decoder == "hgs":
        if is_time_windows:
            solution, solver_value, runtime, status, build_solver_runtime = heu_solve_HGS_VRPTW(
                instance.nodes,
                instance.arc_index,
                instance.arc_costs,
                instance.nb_vehicles,
                instance.vehicle_capacity,
                relevant_connections,
                heu_time=heu_time,
                undirected=False,
                true_time_cost=instance.arc_costs,
                pyvrp_version=pyvrp_version,
            )
        else:
            solution, solver_value, runtime, status, build_solver_runtime = heu_solve_HGS_VRP(
                instance.demands,
                instance.arc_index,
                instance.arc_costs,
                instance.nb_vehicles,
                instance.vehicle_capacity,
                relevant_connections,
                heu_time=heu_time,
                undirected=True,
                arc_likelihood=arc_likelihood,
                instance_log_HGS_dict=instance_log_HGS_dict,
                threshold=threshold,
                nodes=instance.nodes,
                pyvrp_version=pyvrp_version,
            )
    else:
        raise ValueError(f"Unknown decoder: {decoder}")

    return solution, runtime, solver_value, lower_bound, status, build_solver_runtime


def ml_based_cvrp_reduction(
    instance,
    predictor_model,
    threshold_type="top_k",
    threshold=20,
    decoder="hgs",
    decoder_cfg=None,
    heu_time=100,
    time_limit=5000,
    cached_features=None,
    completion_heu_time=0,
    instance_log_HGS_dict=None,
    is_time_windows=False,
    pyvrp_version="old",
):
    """End-to-end ML-based reduce-then-optimize pipeline.

    Step 1: predict arc likelihoods and reduce the arc set.
    Step 2: solve the reduced problem with the configured decoder.
    """
    if threshold!=1:
        (
            relevant_connections,
            (num_arcs_pred, num_arcs_enriched),
            completion_runtime,
            inference_runtime
        ) = get_reduced_problem(
            instance,
            predictor_model,
            threshold_type=threshold_type,
            threshold=threshold,
            cached_features=cached_features,
            completion_heu_time=completion_heu_time,
            is_time_windows=is_time_windows,
            pyvrp_version=pyvrp_version,
        )
    else:
        (relevant_connections,  (num_arcs_pred, num_arcs_enriched), 
         completion_runtime, inference_runtime) =  ([True]*len(instance.arc_costs),
                                                     (len(instance.arc_costs),len(instance.arc_costs)), 0, 0)

    solution, solver_runtime, solver_value, lower_bound, status, build_solver_runtime = solve_reduced_problem(
        instance,
        relevant_connections,
        decoder=decoder,
        decoder_cfg=decoder_cfg,
        heu_time=heu_time,
        time_limit=time_limit,
        threshold=threshold,
        instance_log_HGS_dict=instance_log_HGS_dict,
        is_time_windows=is_time_windows,
        pyvrp_version=pyvrp_version,
    )

    return (
        solution,
        num_arcs_pred,
        num_arcs_enriched,
        solver_value,
        status,
        solver_runtime,
        inference_runtime,
        lower_bound,
        completion_runtime,
        build_solver_runtime
    )
