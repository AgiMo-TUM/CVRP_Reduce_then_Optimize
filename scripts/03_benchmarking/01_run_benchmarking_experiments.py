""" Benchmark FCTP algorithms. """

from datetime import datetime
import gzip
import logging
import os
import pickle as pkl
import random
from time import time

import hydra
import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf
import torch
import matplotlib.pyplot as plt
import gc


from core.data_processing.data_utils import load_instance
from core.utils.utils import flatten_dict
from core.evaluation.benchmarking_utils import get_performance_table

from core.cvrp_solvers.ip_grb import sol_vals
from core.cvrp_solvers.ip_grb import cvrp
from core.ml_models.wrapper import ml_based_cvrp_reduction
from core.utils.ml_utils import load_arc_predictor_model
from core.cvrp_solvers.heuristics import heu_solve_HGS_VRP


def get_k_vals(size_thresholds, m, n):
    """Translate relative size thresholds into absolute size thresholds.

    Parameters
    ----------
    size_thresholds: list
        List of relative size threshold.
    m: int
        Number of supply nodes.
    n: int
        Number of demand nodes.

    Returns
    -------
    k_vals: list
        List of absolute size thresholds.

    """
    num_edges = m * n
    k_vals = [round(num_edges * p) for p in size_thresholds]
    return k_vals


def save_results(path, result_dict):
    """Helper function to save benchmarking results.

    Parameters
    ----------
    path: str
        Path to save results.
    result_dict: dict
        Dictionary with benchmarking information and results.

    Returns
    -------
    None

    """
    with gzip.open(path, "wb") as file:
        pkl.dump(
            result_dict,
            file,
        )


@hydra.main(
    version_base=None,
    config_path="configs/benchmarking",
    config_name="config",
)
#     """Run benchmarking experiments."""

#     seed = cfg.seed
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)

#     os.makedirs(cfg.solution_dir, exist_ok=True)

#     method = cfg.method.name

#     os.makedirs(cfg.log_dir, exist_ok=True)
#     logger = logging.getLogger()
#     logger.addHandler(
#         logging.FileHandler(
#             os.path.join(
#                 cfg.log_dir,
#                 f"{datetime.now().strftime('%Y%m%d_%H:%M:%S')}_benchmarking.log",
#             ),
#             mode="w",
#         )
#     )

#     logger.info(f"Experiment parameters: {cfg}")

#     os.environ["JULIA_NUM_THREADS"] = str(cfg.num_threads)
#     os.environ["OPENBLAS_NUM_THREADS"] = str(cfg.num_threads)
#         logger.info("Initializing Julia environment for TS...")
#         ts_env = TabuSearchJuliaEnv()
#         logger.info("Initializing Julia environment for EA...")
#         ea_env = EvolutionaryAlgorithmJuliaEnv()

#         edge_predictor_model = (
#             load_edge_predictor_model(cfg.method.model_path, get_feature_fun=True),
#             cfg.method.model_name,
#         )

#         decoder = cfg.decoder.name
#         decoder_cfg = OmegaConf.to_container(cfg.decoder)
#         decoder_env = None
#         del decoder_cfg["name"]
#         decoder_param_spec = "_".join(
#             [f"{k}_{v}" for k, v in flatten_dict(decoder_cfg).items()]
#         )
#             decoder_cfg["tabu_in_range"] = tuple(decoder_cfg["tabu_in_range"].values())
#             decoder_cfg["tabu_out_range"] = tuple(
#                 decoder_cfg["tabu_out_range"].values()
#             )
#             decoder_cfg["seed"] = seed
#             logger.info("Initializing Julia environment for TS...")
#             decoder_env = TabuSearchJuliaEnv()
#             decoder_cfg["seed"] = seed
#             logger.info("Initializing Julia environment for EA...")
#             decoder_env = EvolutionaryAlgorithmJuliaEnv()
#             decoder_cfg["grb_threads"] = cfg.num_threads
#             decoder_cfg["grb_verbosity"] = cfg.verbose

#     instance_paths = [
#         os.path.join(cfg.instance_dir, filename)
#         for filename in os.listdir(cfg.instance_dir)
#     ]
#     logger.info(f"{len(instance_paths)} benchmark instances")


#         instance_id = instance_path.split("/")[-1].split(".")[0].split("_")[-1]
#         solution_filename = f"sol_instance_{instance_id}.pkl.gz"

#         result_dict = {
#             "instance_path": instance_path,
#             "method": method,
#             "experiment_config": cfg,
#         }

#         logger.info(
#             f"Processing instance {instance_id} ({counter+1}/{len(instance_paths)})..."
#         )

#         instance = load_instance(instance_path)

#             grb_cfg = OmegaConf.to_container(cfg.method)
#             del grb_cfg["name"]
#             param_spec = "_".join([f"{k}_{v}" for k, v in grb_cfg.items()])
#             start = time()
#                 model, x, _ = capacitated_fctp(
#                     instance.supply,
#                     instance.demand,
#                     instance.var_costs,
#                     instance.fix_costs,
#                     instance.edge_capacities,
#                 )
#                 model, x, _ = fixed_step_fctp(
#                     instance.supply,
#                     instance.demand,
#                     instance.var_costs,
#                     instance.fix_costs,
#                     instance.vehicle_capacities,
#                 )
#             else:
#                 model, x, _ = fctp(
#                     instance.supply,
#                     instance.demand,
#                     instance.var_costs,
#                     instance.fix_costs,
#                 )
#             model.setParam("OutputFlag", 0)
#             model.setParam("TimeLimit", cfg.method.grb_timeout)
#             if cfg.num_threads is not None:
#                 model.setParam("Threads", cfg.num_threads)
#             model.setParam("Seed", seed)
#             model.optimize()
#             sol = sol_vals(x)
#             runtime = time() - start
#             result_dict.update(
#                 {
#                     "solution": sol,
#                     "objective_value": instance.eval_sol_dict(sol),
#                     "runtime": runtime,
#                     "solver_runtime": model.Runtime,
#                     "solver_status": model.Status,
#                     "mip_gap": model.MIPGap,
#                 }
#             )
#             solution_path = os.path.join(cfg.solution_dir, method, param_spec)
#             os.makedirs(solution_path, exist_ok=True)
#             save_results(os.path.join(solution_path, solution_filename), result_dict)

#             if isinstance(instance, CapacitatedFCTP) or isinstance(
#                 instance, FixedStepFCTP
#                 raise NotImplementedError
#             ea_cfg = OmegaConf.to_container(cfg.method)
#             del ea_cfg["name"]
#             param_spec = "_".join([f"{k}_{v}" for k, v in ea_cfg.items()])
#             ea_cfg["seed"] = seed
#             sol, sol_val, runtime = ea_env.run(instance, ea_cfg)
#                 sol
#             ), "Inconsistent objective function values"
#             result_dict.update(
#                 {
#                     "solution": sol,
#                     "objective_value": sol_val,
#                     "runtime": runtime,
#                 }
#             )
#             solution_path = os.path.join(cfg.solution_dir, method, param_spec)
#             os.makedirs(solution_path, exist_ok=True)
#             save_results(os.path.join(solution_path, solution_filename), result_dict)

#             if isinstance(instance, CapacitatedFCTP) or isinstance(
#                 instance, FixedStepFCTP
#                 raise NotImplementedError
#             ts_cfg = OmegaConf.to_container(cfg.method)
#             del ts_cfg["name"]
#             param_spec = "_".join([f"{k}_{v}" for k, v in flatten_dict(ts_cfg).items()])
#             ts_cfg["tabu_in_range"] = tuple(ts_cfg["tabu_in_range"].values())
#             ts_cfg["tabu_out_range"] = tuple(ts_cfg["tabu_out_range"].values())
#             ts_cfg["seed"] = seed
#             bfs = get_fctp_bfs(instance)
#             sol, sol_val, runtime = ts_env.run(instance, bfs, ts_cfg)
#                 sol
#             ), "Inconsistent objective function values"
#             result_dict.update(
#                 {
#                     "solution": sol,
#                     "objective_value": sol_val,
#                     "runtime": runtime,
#                 }
#             )
#             solution_path = os.path.join(cfg.solution_dir, method, param_spec)
#             os.makedirs(solution_path, exist_ok=True)
#             save_results(os.path.join(solution_path, solution_filename), result_dict)

#             start = time()
#                 model, x, _ = capacitated_fctp(
#                     instance.supply,
#                     instance.demand,
#                     instance.var_costs,
#                     instance.fix_costs,
#                     instance.edge_capacities,
#                     relax=True,
#                 )
#                 model, x, _ = fixed_step_fctp(
#                     instance.supply,
#                     instance.demand,
#                     instance.var_costs,
#                     instance.fix_costs,
#                     instance.vehicle_capacities,
#                     relax=True,
#                 )
#             else:
#                 model, x, _ = fctp(
#                     instance.supply,
#                     instance.demand,
#                     instance.var_costs,
#                     instance.fix_costs,
#                     relax=True,
#                 )
#             model.setParam("OutputFlag", 0)
#             model.setParam("TimeLimit", cfg.method.grb_timeout)
#             if cfg.num_threads is not None:
#                 model.setParam("Threads", cfg.num_threads)
#             model.setParam("Seed", seed)
#             model.optimize()
#             sol = sol_vals(x)
#             runtime = time() - start
#             result_dict.update(
#                 {
#                     "solution": sol,
#                     "objective_value": instance.eval_sol_dict(sol),
#                     "runtime": runtime,
#                     "solver_runtime": model.Runtime,
#                     "solver_status": model.Status,
#                 }
#             )
#             solution_path = os.path.join(cfg.solution_dir, method)
#             os.makedirs(solution_path, exist_ok=True)
#             save_results(os.path.join(solution_path, solution_filename), result_dict)

#             if isinstance(instance, CapacitatedFCTP) or isinstance(
#                 instance, FixedStepFCTP
#                 raise NotImplementedError
#             start = time()
#             model, x = tp(instance.supply, instance.demand, costs=instance.var_costs)
#             model.setParam("OutputFlag", 0)
#             model.setParam("TimeLimit", cfg.method.grb_timeout)
#             if cfg.num_threads is not None:
#                 model.setParam("Threads", cfg.num_threads)
#             model.setParam("Seed", seed)
#             model.optimize()
#             sol = sol_vals(x)
#             runtime = time() - start
#             result_dict.update(
#                 {
#                     "solution": sol,
#                     "objective_value": instance.eval_sol_dict(sol),
#                     "runtime": runtime,
#                     "solver_runtime": model.Runtime,
#                     "solver_status": model.Status,
#                 }
#             )
#             solution_path = os.path.join(cfg.solution_dir, method)
#             os.makedirs(solution_path, exist_ok=True)
#             save_results(os.path.join(solution_path, solution_filename), result_dict)

#             if isinstance(instance, CapacitatedFCTP) or isinstance(
#                 instance, FixedStepFCTP
#                 raise NotImplementedError
#             k_vals = get_k_vals(cfg.method.size_threshold, instance.m, instance.n)
#                 thrsh = cfg.method.size_threshold[i]
#                 start = time()
#                 relevant_connections = random_edges_predictor(
#                     (instance.m, instance.n),
#                     k=k_val,
#                 )
#                 num_edges_pred = np.sum(relevant_connections)
#                 relevant_connections = add_feasible_sol_connections(
#                     relevant_connections,
#                     instance,
#                     method="nwc",
#                 )
#                 num_edges_enriched = np.sum(relevant_connections)
#                 sol, solver_runtime, solver_status, mip_gap = solve_reduced_problem(
#                     instance,
#                     relevant_connections,
#                     decoder,
#                     decoder_cfg,
#                     decoder_env,
#                     seed,
#                 )
#                 runtime = time() - start
#                 result_dict_k = result_dict.copy()
#                 result_dict_k.update(
#                     {
#                         "solution": sol,
#                         "objective_value": instance.eval_sol_dict(sol),
#                         "runtime": runtime,
#                         "solver_runtime": solver_runtime,
#                         "num_edges_pred": num_edges_pred,
#                         "num_edges_enriched": num_edges_enriched,
#                         "method_param": k_val,
#                     }
#                 )
#                 if solver_status is not None:
#                     result_dict_k["solver_status"] = solver_status
#                 if mip_gap is not None:
#                     result_dict_k["mip_gap"] = mip_gap
#                 solution_path = os.path.join(
#                     cfg.solution_dir,
#                     method,
#                     f"{decoder}-{decoder_param_spec}",
#                     str(thrsh),
#                 )
#                 os.makedirs(solution_path, exist_ok=True)
#                 save_results(
#                     os.path.join(solution_path, solution_filename), result_dict_k
#                 )

#             if isinstance(instance, CapacitatedFCTP) or isinstance(
#                 instance, FixedStepFCTP
#                 raise NotImplementedError
#             k_vals = get_k_vals(cfg.method.size_threshold, instance.m, instance.n)
#                 thrsh = cfg.method.size_threshold[i]
#                 start = time()
#                 relevant_connections = k_shortest_edges_predictor(
#                     instance,
#                     k=k_val,
#                 )
#                 num_edges_pred = np.sum(relevant_connections)
#                 relevant_connections = add_feasible_sol_connections(
#                     relevant_connections,
#                     instance,
#                     method="lcm",
#                 )
#                 num_edges_enriched = np.sum(relevant_connections)
#                 sol, solver_runtime, solver_status, mip_gap = solve_reduced_problem(
#                     instance,
#                     relevant_connections,
#                     decoder,
#                     decoder_cfg,
#                     decoder_env,
#                     seed,
#                 )
#                 runtime = time() - start
#                 result_dict_k = result_dict.copy()
#                 result_dict_k.update(
#                     {
#                         "solution": sol,
#                         "objective_value": instance.eval_sol_dict(sol),
#                         "runtime": runtime,
#                         "solver_runtime": solver_runtime,
#                         "num_edges_pred": num_edges_pred,
#                         "num_edges_enriched": num_edges_enriched,
#                         "method_param": k_val,
#                     }
#                 )
#                 if solver_status is not None:
#                     result_dict_k["solver_status"] = solver_status
#                 if mip_gap is not None:
#                     result_dict_k["mip_gap"] = mip_gap
#                 solution_path = os.path.join(
#                     cfg.solution_dir,
#                     method,
#                     f"{decoder}-{decoder_param_spec}",
#                     str(thrsh),
#                 )
#                 os.makedirs(solution_path, exist_ok=True)
#                 save_results(
#                     os.path.join(solution_path, solution_filename), result_dict_k
#                 )

#             threshold_type = cfg.method.threshold_type
#                 thresholds = cfg.method.size_threshold
#                 thresholds = cfg.method.prob_threshold
#             else:
#                 raise ValueError

#             for thrsh in thresholds:
#                 start = time()
#                 (
#                     sol,
#                     num_edges_pred,
#                     num_edges_enriched,
#                     solver_runtime,
#                     solver_status,
#                     mip_gap,
#                 ) = ml_based_fctp_reduction(
#                     instance,
#                     predictor_model=edge_predictor_model[0],
#                     threshold_type=threshold_type,
#                     threshold=thrsh,
#                     decoder=decoder,
#                     decoder_cfg=decoder_cfg,
#                     decoder_env=decoder_env,
#                     seed=seed,
#                 )
#                 runtime = time() - start
#                 result_dict_k = result_dict.copy()
#                 result_dict_k.update(
#                     {
#                         "solution": sol,
#                         "objective_value": instance.eval_sol_dict(sol),
#                         "runtime": runtime,
#                         "solver_runtime": solver_runtime,
#                         "num_edges_pred": num_edges_pred,
#                         "num_edges_enriched": num_edges_enriched,
#                         "method_param": thrsh,
#                         "model": edge_predictor_model[1],
#                     }
#                 )
#                 if solver_status is not None:
#                     result_dict_k["solver_status"] = solver_status
#                 if mip_gap is not None:
#                     result_dict_k["mip_gap"] = mip_gap
#                 solution_path = os.path.join(
#                     cfg.solution_dir,
#                     method,
#                     threshold_type,
#                     f"{decoder}-{decoder_param_spec}",
#                     edge_predictor_model[1],
#                     str(thrsh),
#                 )
#                 os.makedirs(solution_path, exist_ok=True)
#                 save_results(
#                     os.path.join(solution_path, solution_filename), result_dict_k
#                 )

#     logger.info("**************** Finished benchmarking ****************")

#     if cfg.summarize:
#         logger.info("Preparing summary table...")
#         df = get_performance_table(cfg.solution_dir)
#         os.makedirs(cfg.result_dir, exist_ok=True)
#         df.to_csv(os.path.join(cfg.result_dir, "benchmarking_summary.csv"))


#     main()

def main(cfg: DictConfig) -> None:
    """Run benchmarking experiments."""

    # Set all random seeds
    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set up solution directory
    os.makedirs(cfg.solution_dir, exist_ok=True)

    # Set up method
    method = cfg.method.name

    # Set up logger
    os.makedirs(cfg.log_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(
            os.path.join(
                cfg.log_dir,
                f"{datetime.now().strftime('%Y%m%d_%H-%M-%S')}_benchmarking.log",
            ),
            mode="w",
        )
    )

    logger.info(f"Experiment parameters: {cfg}")

    # Load prediction model
    if cfg.method.get("model_path") is not None:
        arc_predictor_model = (
            load_arc_predictor_model(cfg.method.model_path, get_feature_fun=True),
            cfg.method.model_name,
        )

    # Set up decoder for problem reduction approaches
    if cfg.get("decoder") is not None:
        decoder = cfg.decoder.name
        decoder_cfg = OmegaConf.to_container(cfg.decoder)
        decoder_env = None
        del decoder_cfg["name"]
        decoder_param_spec = "_".join(
            [f"{k}_{v}" for k, v in flatten_dict(decoder_cfg).items()]
        )

        if decoder in ["exact", "lp"]:
            decoder_cfg["grb_threads"] = cfg.num_threads
            decoder_cfg["grb_verbosity"] = cfg.verbose

    # Process all instances
    instance_paths = [
        os.path.join(cfg.instance_dir, filename)
        for filename in os.listdir(cfg.instance_dir)
    ]
    logger.info(f"{len(instance_paths)} benchmark instances")

    start_index = cfg.start
    end_index = cfg.end
    print("start_index : ", start_index, ", end_index : ", end_index)
    threshold_recall_dict = {}
    recall_standard_method_dict = {}
    test_performance_dict={}
    for counter, instance_path in enumerate(instance_paths):

        if(counter < start_index or counter > end_index):
            continue

        
        filename = instance_path.split("/")[-1].split(".")[0]
        parts = filename.split("_")
        instance_id = "_".join(parts[-3:])


        #     print("first_number = ", first_number)
        #     continue

        solution_filename = f"sol_instance_{instance_id}.pkl.gz"

        # Initialize result dict
        result_dict = {
            "instance_path": instance_path,
            "method": method,
            "experiment_config": cfg,
        }

        # Print progress
        logger.info(
            f"Processing instance {instance_id} ({counter+1}/{len(instance_paths)})..."
        )

        # Load instance
        instance, instance_solution = load_instance(instance_path)


        # Solve exactly (Gurobi)
        if method == "exact":
            grb_cfg = OmegaConf.to_container(cfg.method)
            del grb_cfg["name"]
            param_spec = "_".join([f"{k}_{v}" for k, v in grb_cfg.items()])
            start = time()
            
            demands = np.array([node.demand for node in instance.nodes])
            model, x, _ = cvrp(demands,
                   instance.arc_index,
                   instance.arc_costs,
                   instance.nb_vehicles,
                   instance.vehicle_capacity)
            model.setParam("OutputFlag",0)
            model.setParam("TimeLimit", cfg.method.grb_timeout)
            if cfg.num_threads is not None:
                model.setParam("Threads", cfg.num_threads)
            model.setParam("Seed", seed)
            model.optimize()
            sol = sol_vals(x)
            runtime = time() - start
            result_dict.update(
                {
                    "solution": sol,
                    "objective_value": instance.eval_sol_dict(sol),
                    "runtime": runtime,
                    "solver_runtime": model.Runtime,
                    "solver_status": model.Status,
                    "mip_gap": model.MIPGap,
                }
            )
            solution_path = os.path.join(cfg.solution_dir, method, param_spec)
            os.makedirs(solution_path, exist_ok=True)
            save_results(os.path.join(solution_path, solution_filename), result_dict)

        elif method == "ml-reduction":
            # Threshold type and values
            threshold_type = cfg.method.threshold_type
            if threshold_type == "size":
                thresholds = cfg.method.size_threshold
            elif threshold_type == "prob":
                thresholds = cfg.method.prob_threshold
            else:
                raise ValueError
            
            # _, exact_objective_value = heu_solve_HGS_VRP(instance.demands, instance.arc_index, 
            #           instance.arc_costs, instance.nb_vehicles, instance.vehicle_capacity, 
            #           all_connections)

            exact_objective_value = sum(instance_solution[(u,v)] * instance.arc_costs[k] for k, (u,v) in 
                        enumerate(zip(instance.arc_index[0], instance.arc_index[1])))

            print("exact_objective_value = ", exact_objective_value)

            start = time()
            node_feat, edge_attr, edge_index, _, _ = arc_predictor_model[0][1](instance)
            features_computation_time = time() - start

            print("features_computation_time : ", features_computation_time)
    
            cached_features = (node_feat, edge_attr, edge_index)
            for thrsh in thresholds:

                instance_log_HGS_dict = {}
                start = time()
                (
                    sol,
                    num_arcs_pred,
                    num_arcs_enriched,
                    solver_value,
                    solver_status,
                    solver_runtime,
                    num_missing_arcs,
                    lower_bound,
                    completion_runtime
                ) = ml_based_cvrp_reduction(
                    instance,
                    predictor_model=arc_predictor_model[0],
                    threshold_type=threshold_type,
                    threshold=thrsh,
                    decoder=decoder,
                    decoder_cfg=decoder_cfg,
                    seed=seed,
                    heu_time=cfg.HGS_runtime,
                    time_limit=cfg.exact_time_limit,
                    cached_features=cached_features,
                    completion_heu_time=cfg.completion_heu_time,
                    instance_log_HGS_dict=instance_log_HGS_dict,
                    is_time_windows=False,
                    pyvrp_version=cfg.pyvrp_version,
                )
                print("completion_runtime : ", completion_runtime)
                runtime = time() - start + features_computation_time
                result_dict_k = result_dict.copy()
                result_dict_k.update(
                    {
                        "solution": sol,
                        "exact_objective_value": exact_objective_value,
                        "runtime": runtime,
                        "solver_runtime": solver_runtime,
                        "num_arcs_pred": num_arcs_pred,
                        "num_arcs_enriched": num_arcs_enriched,
                        "method_param": thrsh,
                        "model": arc_predictor_model[1],
                        "solver_value": solver_value,
                        "num_missing_arcs": num_missing_arcs,
                        "lower_bound": lower_bound,
                        "features_computation_time": features_computation_time,
                        "completion_runtime" : completion_runtime,
                        "instance_log_HGS_dict" : instance_log_HGS_dict
                    }
                )
                print(instance_log_HGS_dict)
                if solver_status is not None:
                    result_dict_k["solver_status"] = solver_status
                #     result_dict_k["mip_gap"] = mip_gap
                solution_path = os.path.join(
                    cfg.solution_dir,
                    method,
                    threshold_type,
                    f"{decoder}-{decoder_param_spec}",
                    arc_predictor_model[1],
                    str(thrsh),
                )
                os.makedirs(solution_path, exist_ok=True)
                save_results(
                    os.path.join(solution_path, solution_filename), result_dict_k
                )
                logger.info(
                    f"done_{solver_runtime}_{solver_value}"
                )


            


    #     arr = np.array(recall_standard_method_dict[k])  # shape (n, 2)
    #     kept_mean.append(arr[:, 0].mean())
    #     kept_std.append(arr[:, 0].std())
    #     recall_mean.append(arr[:, 1].mean())
    #     recall_std.append(arr[:, 1].std())


    #                 np.array(kept_mean) - np.array(kept_std),
    #                 np.array(kept_mean) + np.array(kept_std),
    #                 color='blue', alpha=0.2)

    #                 np.array(recall_mean) - np.array(recall_std),
    #                 np.array(recall_mean) + np.array(recall_std),
    #                 color='orange', alpha=0.2)


    for threshold in test_performance_dict.keys():
        print("Threshold : ", threshold)
        for key in test_performance_dict[threshold].keys():
            print(key, " : ", np.mean(test_performance_dict[threshold][key]))

    logger.info("**************** Finished benchmarking ****************")

    # logger.info summary table
    if cfg.summarize:
        logger.info("Preparing summary table...")
        df = get_performance_table(cfg.solution_dir)
        os.makedirs(cfg.result_dir, exist_ok=True)
        df.to_csv(os.path.join(cfg.result_dir, "benchmarking_summary_test.csv"))


if __name__ == "__main__":
    main()