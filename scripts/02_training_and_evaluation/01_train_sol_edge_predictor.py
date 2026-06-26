""" Training script for solution edge prediction model (supervised with binary labels)."""


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

from datetime import datetime
from functools import partial
import logging

import hydra
import numpy as np
from omegaconf import DictConfig
from omegaconf import open_dict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import torch
import glob


from core.ml_models.cvrp_sol_predictor import GCNNSolArcPredictor
from core.utils.ml_utils import CVRPData

from core.utils.ml_utils import StandardNormalizer
from core.utils.ml_utils import MultiInputNormalizer
from core.utils.ml_utils import save_dataset_to_pt
from core.utils.ml_utils import load_labels_from_files

from core.utils.ml_utils import get_graph_raw_features
from core.utils.ml_utils import get_arc_features
from core.utils.ml_utils import get_graph_additional_1_features
from core.utils.ml_utils import get_graph_additional_2_features
from core.utils.ml_utils import get_graph_additional_3_features

from core.utils.ml_utils import get_raw_targets
from core.utils.ml_utils import get_sample_paths
from core.utils.ml_utils import get_class_weights
from core.utils.ml_utils import training_wrapper
from core.utils.ml_utils import binary_classification_eval_display_func
from core.utils.ml_utils import multiclass_classification_eval_display_func
from core.evaluation.ml_model_evaluation import select_best_model_across_candidates


@hydra.main(
    version_base=None,
    config_path="configs/training",
    config_name="config",
)

def main(training_config: DictConfig) -> None:

    # Set random seeds
    seed = training_config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Extract model config
    model_config = training_config.model
    model_spec = "_".join([f"{k}_{v}" for k, v in model_config.items()])

    # Prepare output directory
    out_dir = os.path.join(training_config.out_dir, model_spec)
    if training_config.cross_validate:
        out_dir = os.path.join(out_dir, "cross_val")
    else:
        out_dir = os.path.join(out_dir, "application")
    os.makedirs(out_dir, exist_ok=True)

    # Set up logger
    os.makedirs(training_config.log_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(
            os.path.join(
                training_config.log_dir,
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_train.log",
            ),
            mode="w",
        )
    )

    logger.info(f"Training parameters: {training_config}")

    # Get list of (subset of) sample file paths
    sample_paths = get_sample_paths(
        training_config.data_path, training_config.num_samples
    )
    validation_sample_paths = get_sample_paths(
        training_config.validation_data_path, training_config.validation_num_samples
    )

    # Extract features

    if model_config.features == "graph_raw":
        if not glob.glob("train_pt/graph_raw/*.pt"):
            save_dataset_to_pt(sample_paths, "train_pt/graph_raw", get_graph_raw_features)
        if not glob.glob("val_pt/graph_raw/*.pt"):
            save_dataset_to_pt(validation_sample_paths, "val_pt/graph_raw", get_graph_raw_features)

        with open_dict(model_config):
            model_config.node_dim = 1
            model_config.arc_dim = 1


    elif model_config.features =="graph_additional_1":
        if not glob.glob("train_pt/graph_additional_1/*.pt"):
            save_dataset_to_pt(sample_paths, "train_pt/graph_additional_1", get_graph_additional_1_features)
        if not glob.glob("val_pt/graph_additional_1/*.pt"):
            save_dataset_to_pt(validation_sample_paths, "val_pt/graph_additional_1", get_graph_additional_1_features)

        with open_dict(model_config):
            model_config.node_dim = 2
            model_config.arc_dim = 1
            print("node_dim = ", model_config.node_dim)
            print("arc_dim = ", model_config.arc_dim)
    elif model_config.features =="graph_additional_2":
        if not glob.glob("train_pt/graph_additional_2/*.pt"):
            save_dataset_to_pt(sample_paths, "train_pt/graph_additional_2", get_graph_additional_2_features)
        if not glob.glob("val_pt/graph_additional_2/*.pt"):
            save_dataset_to_pt(validation_sample_paths, "val_pt/graph_additional_2", get_graph_additional_2_features)

        with open_dict(model_config):
            model_config.node_dim = 1
            model_config.arc_dim = 3
            print("node_dim = ", model_config.node_dim)
            print("arc_dim = ", model_config.arc_dim)
    elif model_config.features =="graph_additional_3":
        if not glob.glob(training_config.data_path + "/graph_additional_3/*.pt"):
            save_dataset_to_pt(sample_paths, training_config.data_path + "/graph_additional_3", get_graph_additional_3_features)
        if not glob.glob(training_config.validation_data_path + "/graph_additional_3/*.pt"):
            save_dataset_to_pt(validation_sample_paths, training_config.validation_data_path + "/graph_additional_3", get_graph_additional_3_features)

        with open_dict(model_config):
            model_config.node_dim = 2
            model_config.arc_dim = 4
            print("node_dim = ", model_config.node_dim)
            print("arc_dim = ", model_config.arc_dim)


    elif model_config.features =="graph_additional_TW_1":
        if not glob.glob(training_config.data_path + "/graph_additional_TW_1/*.pt"):
            save_dataset_to_pt(sample_paths, training_config.data_path + "/graph_additional_TW_1", get_graph_additional_3_features)
        if not glob.glob(training_config.validation_data_path + "/graph_additional_TW_1/*.pt"):
            save_dataset_to_pt(validation_sample_paths, training_config.validation_data_path + "/graph_additional_TW_1", get_graph_additional_3_features)

        with open_dict(model_config):
            model_config.node_dim = 5
            model_config.arc_dim = 1
            print("node_dim = ", model_config.node_dim)
            print("arc_dim = ", model_config.arc_dim)


    else:
        raise ValueError

    # Extract labels

    if model_config.prediction_task == "binary_classification":
        with open_dict(model_config):
            model_config.arc_output_dim = 1
    else:
        raise ValueError

    # Define normalization

    input_transformer = None
    if model_config.normalization == "standard":
        normalizers = tuple([StandardNormalizer().to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) for _ in range(2)]) ##careful
        input_transformer = MultiInputNormalizer(normalizers).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Define policy

    if model_config.model == "gcnn":
        policy_fun = GCNNSolArcPredictor
    else:
        raise ValueError

    # Define class weighting

    class_weight_fun = lambda x: None
    if model_config.prediction_task == "binary_classification":
        class_weight_fun = partial(get_class_weights, binary=True)

    # Define optimizer

    adam_params = {
        "lr": training_config.learning_rate,
        "betas": (training_config.momentum, 0.999),
        "weight_decay": training_config.weight_decay,
    }
    if training_config.lr_decay:
        lr_schedule = {
            "factor": training_config.lr_decay_factor,
            "patience": training_config.patience,
            "threshold": training_config.opt_threshold,
        }
    else:
        lr_schedule = None

    # Configure training performance logging

    initial_eval_display_func = None
    running_eval_display_func = None
    if model_config.prediction_task == "binary_classification":
        initial_eval_display_func = partial(
            binary_classification_eval_display_func, include_train=True
        )
        running_eval_display_func = partial(
            binary_classification_eval_display_func, include_train=False
        )
    elif model_config.prediction_task in ["threeclass_classification"]:
        initial_eval_display_func = partial(
            multiclass_classification_eval_display_func, include_train=True
        )
        running_eval_display_func = partial(
            multiclass_classification_eval_display_func, include_train=False
        )

    # Training

    if not training_config.cross_validate:

        if model_config.features =="graph_raw":
            train_files = sorted(glob.glob("train_pt/graph_raw/*.pt"))
            val_files   = sorted(glob.glob("val_pt/graph_raw/*.pt"))
        elif model_config.features =="graph_additional_1":
            train_files = sorted(glob.glob(training_config.data_path + "/graph_additional_1/*.pt"))
            val_files   = sorted(glob.glob(training_config.validation_data_path + "/graph_additional_1/*.pt"))
        elif model_config.features =="graph_additional_2":
            train_files = sorted(glob.glob("train_pt/graph_additional_2/*.pt"))
            val_files   = sorted(glob.glob("val_pt/graph_additional_2/*.pt"))
        elif model_config.features =="graph_additional_3":
            train_files = sorted(glob.glob(training_config.data_path+"/graph_additional_3/*.pt"))
            val_files   = sorted(glob.glob(training_config.validation_data_path + "/graph_additional_3/*.pt"))
        elif model_config.features =="graph_additional_TW_1":
            train_files = sorted(glob.glob(training_config.data_path+"/graph_additional_TW_1/*.pt"))
            val_files   = sorted(glob.glob(training_config.validation_data_path + "/graph_additional_TW_1/*.pt"))


        if training_config.num_samples is not None:
            train_files = train_files[:training_config.num_samples]

        if training_config.validation_num_samples is not None:
            val_files = val_files[:training_config.validation_num_samples]


        train_dataset = CVRPData(train_files)
        val_dataset   = CVRPData(val_files)

        if input_transformer is not None:
            all_nodes = []
            all_edges = []

            for f in train_files:
                sample = torch.load(f)
                all_nodes.append(torch.tensor(sample["x_nodes"], dtype=torch.float))
                all_edges.append(torch.tensor(sample["x_connections"], dtype=torch.float))

            all_nodes = torch.cat(all_nodes, dim=0)
            all_edges = torch.cat(all_edges, dim=0)

            input_transformer.fit([all_nodes, all_edges])

        train_labels = load_labels_from_files(train_files)

        input_transformer = input_transformer.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        policy = policy_fun(
            model_config=model_config,
            adam_params=adam_params,
            lr_schedule=lr_schedule,
            input_transformer=input_transformer,
            class_weight=class_weight_fun(train_labels),
        )

        training_wrapper(
            training_config,
            model_config,
            train_dataset,
            val_dataset,
            policy,
            out_dir,
            logger,
            eval_train_data=False,
            initial_eval_display_func=initial_eval_display_func,
            running_eval_display_func=running_eval_display_func,
        )

    else:
        if model_config.features == "graph_raw":
            all_files = sorted(glob.glob("train_pt/graph_raw/*.pt"))
        elif model_config.features == "graph_additional_1":
            all_files = sorted(glob.glob(training_config.data_path + "/graph_additional_1/*.pt"))
        elif model_config.features == "graph_additional_2":
            all_files = sorted(glob.glob("train_pt/graph_additional_2/*.pt"))
        elif model_config.features == "graph_additional_3":
            all_files = sorted(glob.glob(training_config.data_path + "/graph_additional_3/*.pt"))
        elif model_config.features == "graph_additional_TW_1":
            all_files = sorted(glob.glob(training_config.data_path + "/graph_additional_TW_1/*.pt"))

        if training_config.num_samples is not None:
            all_files = all_files[:training_config.num_samples]

        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        for i, (train_idx, val_idx) in enumerate(kf.split(all_files)):
            logger.info(f"Training on fold {i}")
            fold_out_dir = os.path.join(out_dir, f"fold_{i}")
            os.makedirs(fold_out_dir, exist_ok=True)

            train_files = [all_files[j] for j in train_idx]
            val_files   = [all_files[j] for j in val_idx]

            train_dataset = CVRPData(train_files)
            val_dataset   = CVRPData(val_files)

            if input_transformer is not None:
                all_nodes = []
                all_edges = []
                for f in train_files:
                    sample = torch.load(f, weights_only=False)
                    all_nodes.append(torch.tensor(sample["x_nodes"], dtype=torch.float))
                    all_edges.append(torch.tensor(sample["x_connections"], dtype=torch.float))
                all_nodes = torch.cat(all_nodes, dim=0)
                all_edges = torch.cat(all_edges, dim=0)
                input_transformer.fit([all_nodes, all_edges])

            train_labels = load_labels_from_files(train_files)

            input_transformer = input_transformer.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            policy = policy_fun(
                model_config=model_config,
                directed=training_config.directed,
                adam_params=adam_params,
                lr_schedule=lr_schedule,
                input_transformer=input_transformer,
                class_weight=class_weight_fun(train_labels),
            )

            training_wrapper(
                training_config,
                model_config,
                train_dataset,
                val_dataset,
                policy,
                fold_out_dir,
                logger,
                eval_train_data=False,
                initial_eval_display_func=initial_eval_display_func,
                running_eval_display_func=running_eval_display_func,
            )
        logger.info("Select best model across folds.")
        select_best_model_across_candidates(out_dir)


if __name__ == "__main__":
    main()