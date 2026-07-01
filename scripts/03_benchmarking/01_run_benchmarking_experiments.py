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
            pre_cached_features = cfg.method.pre_cached_features
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
            features_computation_time = 0

            start = time()
            if pre_cached_features:
                node_feat, edge_attr, edge_index, _, _ = arc_predictor_model[0][1](instance)
                features_computation_time = time() - start

                print("features_computation_time : ", features_computation_time)
        
                cached_features = (node_feat, edge_attr, edge_index)
            else:
                cached_features= None
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
                    inference_runtime,
                    lower_bound,
                    completion_runtime,
                    build_solver_runtime
                ) = ml_based_cvrp_reduction(
                    instance,
                    predictor_model=arc_predictor_model[0],
                    threshold_type=threshold_type,
                    threshold=thrsh,
                    decoder=decoder,
                    decoder_cfg=decoder_cfg,
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
                        "lower_bound": lower_bound,
                        "features_computation_time": features_computation_time,
                        "completion_runtime" : completion_runtime,
                        "instance_log_HGS_dict" : instance_log_HGS_dict,
                        "build_solver_runtime":build_solver_runtime,
                        "inference_runtime": inference_runtime
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