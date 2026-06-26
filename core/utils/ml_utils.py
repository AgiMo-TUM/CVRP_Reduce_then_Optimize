""" Utility functions for training and evaluating ML models. """

from collections import defaultdict
import datetime
from functools import partial
import os
import random
import time

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch
# from torch.utils.data import DataLoader

##Changes
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix
import wandb
##End Changes


class _NoOpWandbRun:
    """Drop-in replacement for ``wandb.run`` when ``wandb_use`` is False."""

    def log(self, *args, **kwargs):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

from core.data_processing.data_utils import load_sample

# from core.utils.fctp import CapacitatedFCTP
# from core.utils.fctp import FixedStepFCTP
# from core.fctp_solvers.heuristics import north_west_corner
# from core.ml_models.fctp_sol_predictor import GCNNSolEdgePredictor
# from core.ml_models.fctp_sol_predictor import EdgeLogRegSolEdgePredictor
# from core.ml_models.fctp_sol_predictor import EdgeMLPSolEdgePredictor

##Changes
from core.ml_models.cvrp_sol_predictor import GCNNSolArcPredictor
##End changes

from core.utils.kpi import evaluate_solution_costs_dict
# from core.utils.postprocessing import sol_to_matrix

##Changes 
from core.utils.postprocessing import sol_to_list
from core.utils.additional_features import MST_feature
from core.utils.additional_features import compute_voronoi_adjacency
from core.utils.additional_features import compute_Clark_Wright_savings
##End changes

from core.utils.utils import default_to_regular_dict
import logging
##End changes



logger1 = logging.getLogger()

###################################
# Data processing
###################################


def get_sample_paths(sample_dir, num_samples=None):
    """Get list of sample paths.

    Parameters
    ----------
    sample_dir: str
        Directory containing sample files.
    num_samples: int, optional
        Maximum number of samples.

    Returns
    -------
    sample_paths: list
        List of sample paths.

    """
    sample_paths = [
        os.path.join(sample_dir, filename) for filename in os.listdir(sample_dir)
    ]
    if num_samples is not None:
        np.random.shuffle(sample_paths)
        sample_paths = sample_paths[:num_samples]
    return sample_paths


###################################
# Feature generation
###################################


# def get_bipartite_raw_features_for_instance(instance, batch_dim=True):
#     """Translate instance into feature arrays for bipartite GNN.

#     Features:
#     - supply nodes: supply
#     - demand nodes: demand
#     - edges: variable costs, fixed costs, (capacities)

#     Parameters
#     ----------
#     instance: tuple
#         Instance tuple.
#     batch_dim: bool, optional
#         Indicate whether batch dimension should be added.

#     Returns
#     -------
#     tuple
#         3-element tuple containing supply node features, demand node features,
#         and edge features.

#     """
#     x_s = np.array(instance.supply)[:, None]
#     x_d = np.array(instance.demand)[:, None]
#     if isinstance(instance, CapacitatedFCTP):
#         x_e = np.dstack(
#             (instance.var_costs, instance.fix_costs, instance.edge_capacities)
#         )
#     elif isinstance(instance, FixedStepFCTP):
#         x_e = np.dstack(
#             (instance.var_costs, instance.fix_costs, instance.vehicle_capacities)
#         )
#     else:
#         x_e = np.dstack((instance.var_costs, instance.fix_costs))

#     if batch_dim:
#         x_s = x_s[None, :]
#         x_d = x_d[None, :]
#         x_e = x_e[None, :, :, :]

#     return (x_s, x_d, x_e)


# def get_bipartite_raw_features(
#     sample_paths,
# ):
#     """Translate a set of N sample instances into feature arrays for bipartite GNN.

#     Features:
#     - supply nodes: supply (N x num_suppliers x 1)
#     - demand nodes: demand (N x num_customers x 1)
#     - edges: variable costs, fixed costs, (capacities) (N x num_suppliers x num_customers x 2 or 3)

#     Parameters
#     ----------
#     sample_paths: list
#         List of sample paths.

#     Returns
#     -------
#     tuple
#         3-element tuple containing supply node features, demand node features,
#         and edge features.

#     """

#     x_supply = []
#     x_demand = []
#     x_connection = []

#     # Read samples from pickle files
#     for sample_path in sample_paths:
#         sample = load_sample(sample_path)
#         # Prepare features
#         x_s, x_d, x_e = get_bipartite_raw_features_for_instance(
#             sample["instance"], batch_dim=True
#         )
#         x_supply.append(x_s)
#         x_demand.append(x_d)
#         x_connection.append(x_e)

#     x_supply = np.concatenate(x_supply)
#     x_demand = np.concatenate(x_demand)
#     x_connection = np.concatenate(x_connection)

#     return x_supply, x_demand, x_connection


# def get_bipartite_advanced_edge_features_for_instance(
#     instance,
#     include_statistical_features=False,
#     batch_dim=True,
# ):
#     """Translate instance into feature arrays for bipartite GNN, including advanced edge features.

#     Features:
#     - supply nodes: supply
#     - demand nodes: demand
#     - edges: advanced edge features

#     Parameters
#     ----------
#     instance: tuple
#         Instance tuple.
#     include_statistical_features: bool, optional
#         Indicate whether statistical features should be included. Default is
#         False.
#     batch_dim: bool, optional
#         Indicate whether batch dimension should be added.

#     Returns
#     -------
#     tuple
#         3-element tuple containing supply node features, demand node features,
#         and edge features.

#     """
#     supply = instance.supply
#     demand = instance.demand

#     edge_features = get_advanced_edge_features_for_instance(
#         instance,
#         include_node_features=False,
#         include_statistical_features=include_statistical_features,
#         batch_dim=batch_dim,
#     )

#     if batch_dim:
#         supply_features = np.array(supply)[None, :, None]
#         demand_features = np.array(demand)[None, :, None]
#     else:
#         supply_features = np.array(supply)[:, None]
#         demand_features = np.array(demand)[:, None]
#     return supply_features, demand_features, edge_features


# def get_bipartite_advanced_edge_features(
#     sample_paths,
#     features="bipartite_advanced_edge_features",
# ):
#     """Translate a set of N sample instances into feature arrays for bipartite GNN, including
#     advanced edge features.

#     Features:
#     - supply nodes: supply (N x num_suppliers x 1)
#     - demand nodes: demand (N x num_customers x 1)
#     - edges: advanced edge features (N x num_suppliers x num_customers x num_features)

#     Parameters
#     ----------
#     sample_paths: list
#         List of sample paths.
#     features: str, optional
#         Type of features. Can be 'bipartite_advanced_edge_features' or
#         'bipartite_advanced_edge_features_plus_stat_features'.

#     Returns
#     -------
#     tuple
#         3-element tuple containing supply node features, demand node features,
#         and edge features.

#     """
#     if features == "bipartite_advanced_edge_features":
#         feature_fun = get_bipartite_advanced_edge_features_for_instance
#     elif features == "bipartite_advanced_edge_features_plus_stat_features":
#         feature_fun = partial(
#             get_bipartite_advanced_edge_features_for_instance,
#             include_statistical_features=True,
#         )

#     x_connection = []
#     x_supply = []
#     x_demand = []
#     # Read samples from pickle files
#     for sample_path in sample_paths:
#         sample = load_sample(sample_path)
#         # Prepare features
#         x_s, x_d, x_e = feature_fun(sample["instance"], batch_dim=True)
#         x_supply.append(x_s)
#         x_demand.append(x_d)
#         x_connection.append(x_e)

#     x_connection = np.concatenate(x_connection)
#     x_supply = np.concatenate(x_supply)
#     x_demand = np.concatenate(x_demand)

#     return x_supply, x_demand, x_connection


# def get_combined_raw_edge_features_for_instance(instance, batch_dim=True):
#     """Translate instance into combined edge features for baseline models.

#     Features:
#     - edges: supply, demand, variable costs, fixed costs, (capacities)

#     Parameters
#     ----------
#     instance: tuple
#         Instance tuple.
#     batch_dim: bool, optional
#         Indicate whether batch dimension should be added.

#     Returns
#     -------
#     np.array
#         Edge features.

#     """
#     supply_exp = np.repeat(
#         np.expand_dims(instance.supply, axis=1), repeats=instance.n, axis=1
#     )
#     demand_exp = np.repeat(
#         np.expand_dims(instance.demand, axis=0), repeats=instance.m, axis=0
#     )
#     if isinstance(instance, CapacitatedFCTP):
#         edge_features = np.dstack(
#             (
#                 supply_exp,
#                 demand_exp,
#                 instance.var_costs,
#                 instance.fix_costs,
#                 instance.edge_capacities,
#             )
#         )
#     elif isinstance(instance, FixedStepFCTP):
#         edge_features = np.dstack(
#             (
#                 supply_exp,
#                 demand_exp,
#                 instance.var_costs,
#                 instance.fix_costs,
#                 instance.vehicle_capacities,
#             )
#         )
#     else:
#         edge_features = np.dstack(
#             (supply_exp, demand_exp, instance.var_costs, instance.fix_costs)
#         )

#     if batch_dim:
#         return edge_features[None, :, :, :]
#     else:
#         return edge_features


# def get_num_random_sols(m, n, exp_occ=100):
#     """Calculate how many random solutions are needed to observe edges with a certain frequency.

#     Parameters
#     ----------
#     m: int
#         Number of supply nodes.
#     n: int
#         Number of demand nodes.
#     exp_occ: int
#         Target frequency.

#     Returns
#     -------
#     int
#         Number of random solutions that should be generated.

#     """
#     return int(exp_occ * ((m * n) / (m + n - 1)))


# def get_random_sol(supply, demand):
#     """Get random solution.

#     Shuffle instance and generate solution by applying North-West-Corner Method.

#     Parameters
#     ----------
#     supply: 1D np.array or list
#         A list of supplier capacities/supplies.
#     demand: 1D np.array or list
#         A list of customer demands.

#     Returns
#     -------
#     sol: dict
#         Solution dictionary.

#     """
#     # Randomly permute instance
#     n_supply = len(supply)
#     n_demand = len(demand)

#     supply_mapping = list(range(n_supply))
#     random.shuffle(supply_mapping)

#     demand_mapping = list(range(n_demand))
#     random.shuffle(demand_mapping)

#     supply_shuffled = np.array(
#         [supply[supply_mapping[i]] for i in range(n_supply)], dtype=int
#     )
#     demand_shuffled = np.array(
#         [demand[demand_mapping[i]] for i in range(n_demand)], dtype=int
#     )

#     sol_shuffled = north_west_corner(supply_shuffled, demand_shuffled)
#     sol = {
#         (supply_mapping[i], demand_mapping[j]): v for (i, j), v in sol_shuffled.items()
#     }

#     return sol


# def get_statistical_edge_features(instance):
#     """Get statistical edge features.

#     Features:
#     - edges: ranking-based score, correlation-based score

#     Parameters
#     ----------
#     instance: tuple
#         Instance tuple.

#     Returns
#     -------
#     edge_ranks: np.array
#         Matrix containing rank-based score for each edge.
#     edge_correlations: np.array
#         Matrix containing correlation-based score for each edge.

#     """
#     supply, demand, var_costs, fix_costs = (
#         instance.supply,
#         instance.demand,
#         instance.var_costs,
#         instance.fix_costs,
#     )
#     # Generate random solutions and evaluate their solution quality
#     num_random_sols = get_num_random_sols(len(supply), len(demand))
#     sols = []
#     for _ in range(num_random_sols):
#         sol = get_random_sol(supply, demand)
#         sol_val = evaluate_solution_costs_dict(sol, var_costs, fix_costs)
#         sols.append((sol_val, sol))

#     # Get average solution value
#     sol_vals = np.array([sol[0] for sol in sols])
#     avg_sol_val = np.mean(sol_vals)
#     sol_val_diff = (sol_vals - avg_sol_val).sum()
#     sol_val_var = np.var(sol_vals)

#     # Sort solutions to obtain ranking
#     sols = sorted(sols, key=lambda x: x[0])

#     edges = [(i, j) for i in range(len(supply)) for j in range(len(demand))]
#     # For each edge, collect the ranks of the solutions it is part of (with positive flow)
#     avg_edge_ranks = {e: 0 for e in edges}
#     x_bar = {e: 0 for e in edges}
#     s1 = {e: 0 for e in edges}
#     for i, (sol_val, sol) in enumerate(sols):
#         rank = i + 1
#         for edge, val in sol.items():
#             if val > 0:
#                 avg_edge_ranks[edge] += 1 / rank
#                 x_bar[edge] += 1 / num_random_sols
#                 s1[edge] += sol_val - avg_sol_val

#     edge_correlations = {}
#     for edge in edges:
#         sigma_c = (1 - x_bar[edge]) * s1[edge] - x_bar[edge] * (sol_val_diff - s1[edge])
#         sigma_x = x_bar[edge] * (1 - x_bar[edge]) * num_random_sols
#         edge_correlations[edge] = sigma_c / np.sqrt(sigma_x * sol_val_var)

#     # Convert into numpy matrix
#     edge_ranks = sol_to_matrix(avg_edge_ranks, shape=var_costs.shape)
#     edge_correlations = sol_to_matrix(edge_correlations, shape=var_costs.shape)

#     # Normalize by maximum rank
#     edge_ranks = edge_ranks / edge_ranks.max()
#     edge_correlations = edge_correlations / edge_correlations.min()

#     return edge_ranks, edge_correlations


# def get_advanced_edge_features_for_instance(
#     instance,
#     include_statistical_features=False,
#     include_node_features=True,
#     batch_dim=True,
# ):
#     """Get advanced edge features.

#     Features:
#     - supply_demand_ratio
#     - relative capacity
#     - variable costs
#     - fixed costs
#     - variable costs (relative)
#     - fixed costs (relative)
#     - variable-fixed-cost ratio
#     - supply source node
#     - demand target node
#     - rank-based score
#     - correlation-based score

#     Parameters
#     ----------
#     instance: tuple
#         Instance tuple.
#     include_statistical_features: bool, optional
#         Indicate whether statistical features should be included. Default is
#         False.
#     include_node_features: bool, optional
#         Indicate whether features of source and target should be included.
#         Default is True.
#     batch_dim: bool, optional
#         Indicate whether batch dimension should be added.

#     Returns
#     -------
#     edge_features: np.array
#         Edge features.

#     """
#     supply, demand, var_costs, fix_costs = (
#         instance.supply,
#         instance.demand,
#         instance.var_costs,
#         instance.fix_costs,
#     )
#     supply_exp = np.repeat(np.expand_dims(supply, axis=1), repeats=len(demand), axis=1)
#     demand_exp = np.repeat(np.expand_dims(demand, axis=0), repeats=len(supply), axis=0)

#     capacity = np.minimum(supply_exp, demand_exp)
#     rel_capacity = capacity / capacity.mean()
#     supply_demand_ratio = supply_exp / demand_exp
#     rel_var_costs = var_costs / np.mean(var_costs)
#     rel_fix_costs = fix_costs / np.mean(fix_costs)
#     c_f_ratio = (var_costs * capacity) / fix_costs
#     edge_features = [
#         supply_demand_ratio,
#         rel_capacity,
#         var_costs,
#         fix_costs,
#         rel_var_costs,
#         rel_fix_costs,
#         c_f_ratio,
#     ]
#     if include_node_features:
#         edge_features = [supply_exp, demand_exp] + edge_features

#     if include_statistical_features:
#         edge_ranks, edge_correlations = get_statistical_edge_features(instance)
#         edge_features.append(edge_ranks)
#         edge_features.append(edge_correlations)

#     edge_features = np.dstack(edge_features)

#     if batch_dim:
#         return edge_features[None, :, :, :]
#     else:
#         return edge_features


# def get_edge_features(sample_paths, features="combined_raw_edge_features"):
#     """Translate a set of N sample instances into an edge feature array.

#     Parameters
#     ----------
#     sample_paths: list
#         List of sample paths.
#     features: str, optional
#         Type of features. Can be 'combined_raw_edge_features', 'bipartite_advanced_edge_features' or
#         'bipartite_advanced_edge_features_plus_stat_features'.

#     Returns
#     -------
#     np.array
#         Edge features.

#     """

#     if features == "combined_raw_edge_features":
#         feature_fun = get_combined_raw_edge_features_for_instance
#     elif features == "advanced_edge_features":
#         feature_fun = get_advanced_edge_features_for_instance
#     elif features == "advanced_edge_features_plus_stat_features":
#         feature_fun = partial(
#             get_advanced_edge_features_for_instance, include_statistical_features=True
#         )

#     x_connection = []
#     # Read samples from pickle files
#     for sample_path in sample_paths:
#         sample = load_sample(sample_path)
#         # Prepare features
#         x_connection.append(feature_fun(sample["instance"]))

#     x_connection = np.concatenate(x_connection)

#     return x_connection


###################################
# Label generation
###################################


def get_raw_target_for_instance(
    sol, instance=None, binary_target=True, output_dim=True, batch_dim=True
):
    """Translate solution into target for learning.

    Parameters
    ----------
    sol: np.array
        Solution matrix containing flows.
    instance: tuple
        Instance.
    binary_target: bool, optional
        Indicate whether target should be binary (1 if flow >0, 0 otherwise).
    output_dim: bool, optional
        Indicate whether output dimension should be added.
    batch_dim: bool, optional
        Indicate whether batch dimension should be added.

    Returns
    -------
    np.array
        Target array.

    """
    if output_dim:
        sol = sol[:, :, None]
    if binary_target:
        sol = np.where(sol > 0, 1, 0)
    if batch_dim:
        sol = np.expand_dims(sol, axis=0)
    return sol


# def get_raw_targets(
#     sample_paths,
#     binary_target=True,
#     output_dim=True,
# ):
#     """Translate a set of N sample solutions into labels for learning.

#     Parameters
#     ----------
#     sample_paths: list
#         List of sample paths.
#     binary_target: bool, optional
#         Indicate whether target should be binary (1 if flow >0, 0 otherwise).
#     batch_dim: bool, optional
#         Indicate whether batch dimension should be added.

#     Returns
#     -------
#     np.array
#         Target array.

#     """
#     y = []

#     # Read samples from pickle files
#     for sample_path in sample_paths:
#         sample = load_sample(sample_path)
#         sol = sol_to_matrix(sample["solution"])
#         # Prepare target
#         if output_dim:
#             sol = sol[:, :, None]
#         if binary_target:
#             sol = np.where(sol > 0, 1, 0)
#         y.append(sol)

#     y = np.array(y)

#     return y


def get_class_weights(y, binary=False):
    """Calculate class weights for unbalanced data set.

    Parameters
    ----------
    y: np.array
        Labels.
    binary: bool, optional
        Indicate whether there are two or more classes.

    Returns
    -------
    float or np.array
        Class weight for positive class (binary classification) or class weights
        for all classes.

    """
    if isinstance(y, list):
        y = np.concatenate([i.flatten() for i in y])
    y = y.flatten()

    cw = compute_class_weight("balanced", classes=np.unique(y), y=y)

    if binary:
        ratio = cw[1] / cw[0]
        return ratio
    else:
        return cw


###################################
# Data set classes
###################################


# class FCTPData(Dataset):
#     """Dataset wrapper for FCTP training data.

#     Parameters
#     ----------
#     x: list of np.array
#         List of feature arrays.
#     y: np.array
#         Labels.

#     """

#     def __init__(self, x, y):
#         super(FCTPData, self).__init__()

#         self.x = x
#         self.y = y

#         self.data_shapes = tuple([x_i[0].shape[-1] for x_i in self.x])

#     def __getitem__(self, index):
#         return (
#             tuple([x_i[index] for x_i in self.x]),
#             self.y[index],
#         )

#     def __len__(self):
#         return len(self.y)


###################################
# Normalization and transformation functions
###################################


class StandardNormalizer:
    """Normalizer to normalize to zero mean and unit variance.

    Parameters
    ----------
    mean: np.array, optional
        Array of feature means. If not provided, it will be calculated from data in the fit
        function.
    std: np.array, optional
        Array of feature standard deviations. If not provided, it will be calculated from data in
        the fit function.

    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std


    def to(self, device):
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.std is not None:
            self.std = self.std.to(device)
        return self


    def fit(self, data):
        """Fit normalizer.

        Parameters
        ----------
        data: torch.Tensor
            Data (num_samples x num_features).

        Returns
        -------
        None

        """
        if isinstance(data, list):
            raise NotImplementedError

        # compute mean and std for each feature (i.e., exclude feature dimension)
        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0)
        self.std[self.std < 1e-5] = 1.0  # prevent divison by zero

    def transform(self, data):
        """Apply normalization.

        Parameters
        ----------
        data: torch.Tensor
            Data (num_samples x num_features).

        Returns
        -------
        torch.Tensor
            Normalized data.

        """

        mean = self.mean.to(data.device)
        std = self.std.to(data.device)
        return (data - mean) / std



    def get_state(self):
        """Retrieve normalizer attributes.

        Returns
        -------
        tuple
            2-element tuple, containing mean and standard deviation arrays.

        """
        return (self.mean, self.std)


class MultiInputNormalizer:
    """Wrapper around multiple normalizers.

    Parameters
    ----------
    normalizers: list of StandardNormalizer
        Normalizers.

    """

    def __init__(self, normalizers):
        self._normalizers = normalizers

    def to(self, device):
        for n in self._normalizers:
            if n is not None:
                n.to(device)
        return self

    def fit(self, data):
        """Fit all normalizers.

        Parameters
        ----------
        data: list of torch.Tensor
            Each tensor is [num_samples, num_features] or concatenated 
            features across graphs. Needs to have the same length as self._normalizers.

        Returns
        -------
        None

        """
        assert len(data) == len(
            self._normalizers
        ), "Number of inputs and normalizers does not match"

        for i, normalizer in enumerate(self._normalizers):
            if normalizer is not None:
                normalizer.fit(data[i])

    def transform(self, data_orig):
        """Apply normalizers.

        Parameters
        ----------
        data_orig: list of torch.Tensor
            Each tensor is [num_samples, num_features] to be normalized.
            Needs to have the same length as self._normalizers.

        Returns
        -------
        data: list of np.array
            Normalized data.

        """
        assert len(data_orig) == len(
            self._normalizers
        ), "Number of inputs and normalizers does not match"

        data = []
        for i, normalizer in enumerate(self._normalizers):
            if normalizer is not None:
                data.append(normalizer.transform(data_orig[i]))
            else:
                data.append(data_orig[i])

        return tuple(data)

    def fit_transform(self, data):
        """Fit and transform.

        Parameters
        ----------
        data: list of np.array
            List of data arrays (num_samples x num_features). Needs to have the same length as
            self._normalizers.

        Returns
        -------
        list of np.array
            Normalized data.

        """
        self.fit(data)
        return self.transform(data)

    def get_state(self):
        """Retrieve normalizer attributes.

        Returns
        -------
        tuple
            Mean and standard deviation arrays for all normalizers.

        """
        state = []
        for normalizer in self._normalizers:
            if normalizer is not None:
                state.append(normalizer.get_state())
        return tuple(state)


############################
# Training wrapper and helper functions
############################


def binary_classification_eval_display_func(p, include_train=True):
    """Print model performance of binary classification model in pretty format.

    Parameters
    ----------
    p: dict
        Model performance dictionary.
    include_train: bool, optional
        Indicate whether train data performance should be printed in addition to
        validation data performance. Default is True.

    Returns
    -------
    str
        Performance summary as string.

    """
    if include_train:
        return (
            f"Train loss: {np.round(p['train_loss'], 4)} "
            f"Train acc: {np.round(100 * p['train_accuracy'],2)}%. "
            f"Train rec: {np.round(100 * p['train_recall'],2)}%. "
            f"Validation loss: {np.round(p['validation_loss'], 4)} "
            f"Validation acc: {np.round(100 * p['validation_accuracy'],2)}%. "
            f"Validation rec: {np.round(100 * p['validation_recall'],2)}%. "
        )

    else:
        return (
            f"Validation loss: {np.round(p['validation_loss'], 4)} "
            f"Validation acc: {np.round(100 * p['validation_accuracy'],2)}%. "
            f"Validation rec: {np.round(100 * p['validation_recall'],2)}%. "
        )


def multiclass_classification_eval_display_func(p, include_train=True):
    """Print model performance of multiclass classification model in pretty format.

    Parameters
    ----------
    p: dict
        Model performance dictionary.
    include_train: bool, optional
        Indicate whether train data performance should be printed in addition to
        validation data performance. Default is True.

    Returns
    -------
    str
        Performance summary as string.

    """
    if include_train:
        return (
            f"Train loss: {np.round(p['train_loss'], 4)} "
            f"Train acc: {np.round(100 * p['train_accuracy'],2)}%. "
            f"Validation loss: {np.round(p['validation_loss'], 4)} "
            f"Validation acc: {np.round(100 * p['validation_accuracy'],2)}%. "
        )

    else:
        return (
            f"Validation loss: {np.round(p['validation_loss'], 4)} "
            f"Validation acc: {np.round(100 * p['validation_accuracy'],2)}%. "
        )


# def save_checkpoint(
#     policy, exp_dict, epoch, model_config, training_config, chkpnt_path
# ):
#     """Save pytorch model parameters, experiment dict and configs.

#     Parameters
#     ----------
#     policy: BaseSolEdgePredictor
#         FCTP solution edge predictor.
#     exp_dict: dict or defaultdict
#         Experiment dictionary.
#     epoch: int
#         Training epoch.
#     model_config: dict
#         Model config dict.
#     training_config: dict
#         Training config dict.
#     chkpnt_path: str
#         Path to save checkpoint.

#     Returns
#     -------
#     None

#     """
#     if policy.input_transformer is not None:
#         normalizer_state = policy.input_transformer.get_state()
#     else:
#         normalizer_state = None
#     torch.save(
#         {
#             "epoch": epoch,
#             "model_config": model_config,
#             "state_dict": policy.model.state_dict(),
#             "normalizers": normalizer_state,
#             "training_config": training_config,
#             "exp_dict": default_to_regular_dict(exp_dict),
#         },
#         chkpnt_path,
#     )


# def training_wrapper(
#     training_config,
#     model_config,
#     train_dataset,
#     val_dataset,
#     policy,
#     out_dir,
#     logger,
#     eval_train_data=False,
#     initial_eval_display_func=None,
#     running_eval_display_func=None,
# ):
#     """Model training wrapper function.

#     Parameters
#     ----------
#     training_config: dict
#         Training config dict.
#     model_config: dict
#         Model config dict.
#     train_dataset: FCTPData
#         Training data.
#     val_dataset: FCTPData
#         Validation data.
#     policy: BaseSolEdgePredictor
#         FCTP solution edge predictor to be trained.
#     out_dir: str
#         Path to output directory to save checkpoints and logging file.
#     logger: logging.Logger
#         Logger object.
#     eval_train_data: bool, optional
#         Indicate whether training data performance should be evaluated after every epoch. Default
#         is False.
#     initial_eval_display_func: Function
#         Function to display initial model performance as pretty string.
#     running_eval_display_func: Function
#         Function to display model performance after every epoch as pretty string.

#     Returns
#     -------
#     exp_dict: dict or defaultdict
#         Experiment dictionary.
#     best_period: int
#         Epoch of best model performance.
#     best_kpi_val: float
#         Value of monitoring KPI (loss, accuracy, etc.) in best epoch.

#     """
#     if initial_eval_display_func is None:
#         initial_eval_display_func = lambda x: x

#     if running_eval_display_func is None:
#         running_eval_display_func = lambda x: x

#     # Training
#     train_loader = DataLoader(
#         dataset=train_dataset,
#         batch_size=training_config.train_batch_size,
#         shuffle=True,
#     )

#     # Validation
#     val_loader = DataLoader(
#         dataset=val_dataset,
#         batch_size=training_config.val_batch_size,
#         shuffle=False,
#     )

#     logger.info(f"Number of training samples: {len(train_dataset)}")
#     logger.info(f"Number of validation samples: {len(val_dataset)}")

#     #######################################################
#     # Training Loop
#     #######################################################

#     # Pre-configure saving function
#     save_chkpnt = partial(
#         save_checkpoint,
#         model_config=model_config,
#         training_config=dict(training_config),
#     )

#     exp_dict = defaultdict(lambda: [])

#     exp_dict["device"] = "gpu" if training_config.use_gpu else "cpu"
#     exp_dict["num_train_samples"] = len(train_dataset)
#     exp_dict["max_num_epochs"] = training_config.max_num_epochs
#     exp_dict["initial_performance"] = defaultdict(lambda: [])
#     exp_dict["performance"] = defaultdict(lambda: [])

#     # Evaluate policy before first update
#     p = policy.evaluate({"train": train_loader, "validation": val_loader})
#     for key, value in p.items():
#         exp_dict["initial_performance"][key].append(value)
#     logger.info(("[Initial performance] " f"{initial_eval_display_func(p)}"))

#     # Training loop
#     best_period = 0
#     best_kpi_val = np.inf
#     total_start_time = time.time()
#     logger.info("Starting training loop...")
#     for i in range(training_config.max_num_epochs):

#         policy.model.train()

#         start_time = time.time()
#         start_time_process = time.process_time()

#         running_loss = 0
#         for batch in train_loader:
#             # Update model parameters
#             loss, _ = policy.train_step(batch)
#             running_loss += loss.cpu().item()
#         running_loss = running_loss / len(train_loader)

#         train_time = time.time() - start_time
#         train_time_process = time.process_time() - start_time_process
#         exp_dict["train_time"].append(train_time)
#         exp_dict["train_time_process"].append(train_time_process)
#         exp_dict["performance"]["train_loss"].append(running_loss)

#         # Evaluate model performance
#         if eval_train_data:
#             p = policy.evaluate({"train": train_loader, "validation": val_loader})
#         else:
#             p = policy.evaluate({"validation": val_loader})
#         for key, value in p.items():
#             exp_dict["performance"][key].append(value)

#         # Adapt learning rate based on performance stagnation
#         policy.lr_scheduler_step(p[training_config.monitoring_kpi])

#         # Log current epoch performance and save checkpoint
#         logger.info(
#             (
#                 f"[Epoch {i+1}/{training_config.max_num_epochs}] "
#                 f"Running train loss: {np.round(running_loss, 4)} "
#                 f"{running_eval_display_func(p)} "
#                 f"({datetime.timedelta(seconds=train_time)})"
#             )
#         )
#         save_chkpnt(
#             policy,
#             exp_dict,
#             i + 1,
#             chkpnt_path=os.path.join(out_dir, "checkpoint.pth.tar"),
#         )

#         # Check if model performance has improved
#         if p[training_config.monitoring_kpi] < best_kpi_val * (
#             1 - training_config.opt_threshold
#         ):
#             logger.info("Better model found!")
#             best_kpi_val = p[training_config.monitoring_kpi]
#             best_period = i
#             save_chkpnt(
#                 policy,
#                 exp_dict,
#                 i + 1,
#                 chkpnt_path=os.path.join(out_dir, "best_checkpoint.pth.tar"),
#             )

#         # Early stopping if model performance has not improved over several epochs
#         if i - best_period >= training_config.early_stopping:
#             logger.info(f"No improvement since period {best_period + 1} -> Stopping")
#             break

#     total_train_time = time.time() - total_start_time
#     logger.info(
#         f"Training finished after {datetime.timedelta(seconds=total_train_time)}."
#     )
#     return exp_dict, best_period, best_kpi_val


# ############################
# # Model pre- and postprocessing
# ############################


# def load_edge_predictor_model(model_path, get_feature_fun=True, get_label_fun=False):
#     """Helper function to load edge predictor model.

#     Parameters
#     ----------
#     model_path: str
#         Path to model checkpoint, containing model weights, normalizer states, and configs.
#     get_feature_fun: bool, optional
#         Indicate whether feature extraction function should be returned in addition to edge
#         predictor model. Default is True.
#     get_label_fun: bool, optional
#         Indicate whether label extraction function should be returned in addition to edge predictor
#         model. Default is True.

#     Returns
#     -------
#     tuple
#         1-element, 2-element, or 3-element tuple, comprising edge predictor model and potentially
#         feature extraction function and label extraction function.

#     """
#     return_vals = []

#     chkpnt = torch.load(model_path, map_location="cpu")
#     model_config = chkpnt["model_config"]
#     model_state = chkpnt["state_dict"]
#     normalizer_state = chkpnt["normalizers"]

#     # Configure policy
#     if model_config.model == "gcnn":
#         policy_fun = GCNNSolEdgePredictor
#     elif model_config.model == "linear_logreg":
#         policy_fun = EdgeLogRegSolEdgePredictor
#     elif model_config.model == "edge_mlp":
#         policy_fun = EdgeMLPSolEdgePredictor
#     else:
#         raise ValueError

#     # Configure normalizer
#     input_transformer = None
#     if model_config.normalization == "standard":
#         assert (
#             normalizer_state is not None
#         ), "Normalization parameters expected but are None"
#         normalizers = tuple([StandardNormalizer(*vals) for vals in normalizer_state])
#         input_transformer = MultiInputNormalizer(normalizers)

#     # Load model and parameters
#     policy = policy_fun(
#         model_config=model_config,
#         input_transformer=input_transformer,
#     )
#     policy.model.load_state_dict(model_state)
#     policy.model.eval()

#     if not get_feature_fun and not get_label_fun:
#         return policy

#     return_vals = [policy]

#     if get_feature_fun:
#         # Configure feature extraction function
#         if model_config.features == "bipartite_raw":
#             feature_fun = get_bipartite_raw_features_for_instance
#         elif model_config.features == "combined_raw_edge_features":
#             feature_fun = get_combined_raw_edge_features_for_instance
#         elif model_config.features == "advanced_edge_features":
#             feature_fun = get_advanced_edge_features_for_instance
#         elif model_config.features == "advanced_edge_features_plus_stat_features":
#             feature_fun = partial(
#                 get_advanced_edge_features_for_instance,
#                 include_statistical_features=True,
#             )
#         elif model_config.features == "bipartite_advanced_edge_features":
#             feature_fun = get_bipartite_advanced_edge_features_for_instance
#         elif (
#             model_config.features
#             == "bipartite_advanced_edge_features_plus_stat_features"
#         ):
#             feature_fun = partial(
#                 get_bipartite_advanced_edge_features_for_instance,
#                 include_statistical_features=True,
#             )
#         else:
#             raise ValueError
#         return_vals.append(feature_fun)

#     if get_label_fun:
#         # Configure label extraction function
#         if model_config.prediction_task == "binary_classification":
#             label_fun = partial(get_raw_target_for_instance, binary_target=True)
#         else:
#             raise ValueError
#         return_vals.append(label_fun)

#     return tuple(return_vals)



## Changes

def get_graph_raw_features_for_instance(instance, batch_dim=True):
    """Translate instance into feature arrays for GNN.

    Features:
    - nodes : demands
    - arcs: costs
    - arc_index : 2D np.array 

    Parameters
    ----------
    instance: object
        Instance object with .nodes, .arc_costs and .arc_index attributes.
    batch_dim: bool, optional
        Indicate whether batch dimension should be added.

    Returns
    -------
    tuple
        3-element tuple containing node features, arc features and arc_index.

    """

    # print("instance : ", instance)
    # x_nodes = np.array([node.demand for node in instance.nodes])[:, None] 

    # Demand is always present
    demands = np.array([float(n.demand) for n in instance.nodes], dtype=np.float32)
    features = [demands]

    # Detect CVRPTW vs CVRP
    has_tw = any(getattr(n, "ready_time", None) is not None for n in instance.nodes)

    if has_tw:
        ready = np.array([float(n.ready_time) for n in instance.nodes], dtype=np.float32)
        due   = np.array([float(n.due_time)   for n in instance.nodes], dtype=np.float32)
        serv  = np.array([float(n.service_time) for n in instance.nodes], dtype=np.float32)
        features += [ready, due, serv]

    # (N, F)
    x_nodes = np.stack(features, axis=1)[:, :, None]


    x_a = np.array(instance.arc_costs)[:, None]
    arc_index = np.array(instance.arc_index)
    nb_vehicles = instance.nb_vehicles
    vehicle_capacity = instance.vehicle_capacity

    # if batch_dim:
    #     x_nodes = x_nodes[None, :, :] #1xnx1 shape
    #     x_a = x_a[None, :, :] #1xmx1 shape
    #     # arc_index = arc_index[None, :, :]  # shape (1, 2, m)

    return (x_nodes, x_a, arc_index, nb_vehicles, vehicle_capacity)

def get_graph_raw_features(
    sample_paths,
):
    """Translate a set of N sample instances into feature arrays for GNN.

    Features:
    - nodes: node (N x (num_clients + 1) x 1)
    - arcs: arc costs (N x m x 1)
    - arc_index : index of arcs with source and destination (N x 2 x m x 1)

    Parameters
    ----------
    sample_paths: list
        List of sample paths.

    Returns
    -------
    tuple
        "-element tuple containing node features, arc features and arc_index.

    """

    x_nodes = []
    x_connection = []
    x_arc_index = []
    nb_vehicles_list = []
    capacity_vehicle_list = []

    # Read samples from pickle files
    for sample_path in sample_paths:
        sample = load_sample(sample_path)
        # Prepare features (now returns 3 elements)
        x_no, x_a, arc_index, nb_vehicles, capacity_vehicle = get_graph_raw_features_for_instance(
            sample["instance"], batch_dim=True
        )
        x_nodes.append(x_no)
        x_connection.append(x_a)
        x_arc_index.append(arc_index)
        nb_vehicles_list.append(nb_vehicles)
        capacity_vehicle_list.append(capacity_vehicle)

    # Concatenate along batch dimension

    return x_nodes, x_connection, x_arc_index, nb_vehicles_list, capacity_vehicle_list

def get_graph_additional_1_features_for_instance(instance, batch_dim=True):
    """Extract node + arc features (variant 1) for a single CVRP instance."""


    x_nodes = np.array([[node.demand, instance.vehicle_capacity] for node in instance.nodes])[:,:, None] 
    x_a = np.array([[cost] 
          for k, cost in enumerate(instance.arc_costs)])[:,:, None]
    arc_index = np.array(instance.arc_index)
    nb_vehicles = instance.nb_vehicles
    vehicle_capacity = instance.vehicle_capacity


    return (x_nodes, x_a, arc_index, nb_vehicles, vehicle_capacity)

def get_graph_additional_1_features(
    sample_paths,
):
    """Translate a set of N sample instances into feature arrays for GNN.

    Features:
    - nodes: node (N x (num_clients + 1) x 1)
    - arcs: arc costs (N x m x 1)
    - arc_index : index of arcs with source and destination (N x 2 x m x 1)

    Parameters
    ----------
    sample_paths: list
        List of sample paths.

    Returns
    -------
    tuple
        "-element tuple containing node features, arc features and arc_index.

    """

    x_nodes = []
    x_connection = []
    x_arc_index = []
    nb_vehicles_list = []
    capacity_vehicle_list = []

    # Read samples from pickle files
    for sample_path in sample_paths:
        sample = load_sample(sample_path)
        # Prepare features (now returns 3 elements)
        x_no, x_a, arc_index, nb_vehicles, capacity_vehicle = get_graph_additional_1_features_for_instance(
            sample["instance"], batch_dim=True
        )
        x_nodes.append(x_no)
        x_connection.append(x_a)
        x_arc_index.append(arc_index)
        nb_vehicles_list.append(nb_vehicles)
        capacity_vehicle_list.append(capacity_vehicle)

    # Concatenate along batch dimension

    return x_nodes, x_connection, x_arc_index, nb_vehicles_list, capacity_vehicle_list

def get_graph_additional_2_features_for_instance(instance, batch_dim=True):
    """Extract node + arc features (variant 2) for a single CVRP instance."""

    MST_bool =  MST_feature(instance)
    # Voro_adj_bool = compute_voronoi_adjacency(instance)
    Clark_Wright_savings = compute_Clark_Wright_savings(instance)

    x_nodes = np.array([[node.demand] for node in instance.nodes])[:,:, None] 
    x_a = np.array([[cost, MST_bool[k], Clark_Wright_savings[k]] 
          for k, cost in enumerate(instance.arc_costs)])[:,:, None]
    arc_index = np.array(instance.arc_index)
    nb_vehicles = instance.nb_vehicles
    vehicle_capacity = instance.vehicle_capacity


    return (x_nodes, x_a, arc_index, nb_vehicles, vehicle_capacity)

def get_graph_additional_2_features(
    sample_paths,
):
    """Translate a set of N sample instances into feature arrays for GNN.

    Features:
    - nodes: node (N x (num_clients + 1) x 1)
    - arcs: arc costs (N x m x 1)
    - arc_index : index of arcs with source and destination (N x 2 x m x 1)

    Parameters
    ----------
    sample_paths: list
        List of sample paths.

    Returns
    -------
    tuple
        "-element tuple containing node features, arc features and arc_index.

    """

    x_nodes = []
    x_connection = []
    x_arc_index = []
    nb_vehicles_list = []
    capacity_vehicle_list = []

    # Read samples from pickle files
    for sample_path in sample_paths:
        sample = load_sample(sample_path)
        # Prepare features (now returns 3 elements)
        x_no, x_a, arc_index, nb_vehicles, capacity_vehicle = get_graph_additional_2_features_for_instance(
            sample["instance"], batch_dim=True
        )
        x_nodes.append(x_no)
        x_connection.append(x_a)
        x_arc_index.append(arc_index)
        nb_vehicles_list.append(nb_vehicles)
        capacity_vehicle_list.append(capacity_vehicle)

    # Concatenate along batch dimension

    return x_nodes, x_connection, x_arc_index, nb_vehicles_list, capacity_vehicle_list

def get_graph_additional_3_features_for_instance(instance, batch_dim=True):
    """Extract node + arc features (variant 3) for a single CVRP instance."""
    logger1.info("Start MST")
    MST_bool =  MST_feature(instance)
    logger1.info("Start Voro")
    Voro_adj_bool = compute_voronoi_adjacency(instance)
    logger1.info("Start Clark")
    Clark_Wright_savings = compute_Clark_Wright_savings(instance)
    logger1.info("done")
    x_nodes = np.array([[node.demand, instance.vehicle_capacity] for node in instance.nodes])[:,:, None] 
    x_a = np.array([[cost, MST_bool[k], Voro_adj_bool[k], Clark_Wright_savings[k]] 
          for k, cost in enumerate(instance.arc_costs)])[:,:, None]
    arc_index = np.array(instance.arc_index)
    nb_vehicles = instance.nb_vehicles
    vehicle_capacity = instance.vehicle_capacity


    return (x_nodes, x_a, arc_index, nb_vehicles, vehicle_capacity)

def get_graph_additional_3_features(
    sample_paths,
):
    """Translate a set of N sample instances into feature arrays for GNN.

    Features:
    - nodes: node (N x (num_clients + 1) x 1)
    - arcs: arc costs (N x m x 1)
    - arc_index : index of arcs with source and destination (N x 2 x m x 1)

    Parameters
    ----------
    sample_paths: list
        List of sample paths.

    Returns
    -------
    tuple
        "-element tuple containing node features, arc features and arc_index.

    """

    x_nodes = []
    x_connection = []
    x_arc_index = []
    nb_vehicles_list = []
    capacity_vehicle_list = []

    # Read samples from pickle files
    for sample_path in sample_paths:
        sample = load_sample(sample_path)
        # Prepare features (now returns 3 elements)
        x_no, x_a, arc_index, nb_vehicles, capacity_vehicle = get_graph_additional_3_features_for_instance(
            sample["instance"], batch_dim=True
        )
        x_nodes.append(x_no)
        x_connection.append(x_a)
        x_arc_index.append(arc_index)
        nb_vehicles_list.append(nb_vehicles)
        capacity_vehicle_list.append(capacity_vehicle)

    # Concatenate along batch dimension

    return x_nodes, x_connection, x_arc_index, nb_vehicles_list, capacity_vehicle_list

def get_graph_additional_TW_1_features_for_instance(instance, batch_dim=True):
    """Extract node + arc features (CVRPTW variant 1) for a single instance."""

    # MST_bool =  MST_feature(instance)
    # Voro_adj_bool = compute_voronoi_adjacency(instance)
    # Clark_Wright_savings = compute_Clark_Wright_savings(instance)

    # x_nodes = np.array([[node.demand, instance.vehicle_capacity] for node in instance.nodes])[:,:, None] 

    demands = np.array([float(n.demand) for n in instance.nodes], dtype=np.float32)
    features = [demands] + [np.array([instance.vehicle_capacity for n in instance.nodes], dtype=np.float32)]

    # Detect CVRPTW vs CVRP
    has_tw = any(getattr(n, "ready_time", None) is not None for n in instance.nodes)

    if has_tw:
        ready = np.array([float(n.ready_time) for n in instance.nodes], dtype=np.float32)
        due   = np.array([float(n.due_time)   for n in instance.nodes], dtype=np.float32)
        serv  = np.array([float(n.service_time) for n in instance.nodes], dtype=np.float32)
        features += [ready, due, serv]

    # (N, F)
    x_nodes = np.stack(features, axis=1)[:, :, None]
    x_a = np.array([[cost] 
          for k, cost in enumerate(instance.arc_costs)])[:,:, None]
    arc_index = np.array(instance.arc_index)
    nb_vehicles = instance.nb_vehicles
    vehicle_capacity = instance.vehicle_capacity


    return (x_nodes, x_a, arc_index, nb_vehicles, vehicle_capacity)

def get_graph_additional_TW_1_features(
    sample_paths,
):
    """Translate a set of N sample instances into feature arrays for GNN.

    Features:
    - nodes: node (N x (num_clients + 1) x 1)
    - arcs: arc costs (N x m x 1)
    - arc_index : index of arcs with source and destination (N x 2 x m x 1)

    Parameters
    ----------
    sample_paths: list
        List of sample paths.

    Returns
    -------
    tuple
        "-element tuple containing node features, arc features and arc_index.

    """

    x_nodes = []
    x_connection = []
    x_arc_index = []
    nb_vehicles_list = []
    capacity_vehicle_list = []

    # Read samples from pickle files
    for sample_path in sample_paths:
        sample = load_sample(sample_path)
        # Prepare features (now returns 3 elements)
        x_no, x_a, arc_index, nb_vehicles, capacity_vehicle = get_graph_additional_TW_1_features_for_instance(
            sample["instance"], batch_dim=True
        )
        x_nodes.append(x_no)
        x_connection.append(x_a)
        x_arc_index.append(arc_index)
        nb_vehicles_list.append(nb_vehicles)
        capacity_vehicle_list.append(capacity_vehicle)

    # Concatenate along batch dimension

    return x_nodes, x_connection, x_arc_index, nb_vehicles_list, capacity_vehicle_list

def get_graph_additional_TW_3_features_for_instance(instance, batch_dim=True):
    """Extract node + arc features (CVRPTW variant 3) for a single instance."""

    MST_bool =  MST_feature(instance)
    Voro_adj_bool = compute_voronoi_adjacency(instance)
    Clark_Wright_savings = compute_Clark_Wright_savings(instance)

    # x_nodes = np.array([[node.demand, instance.vehicle_capacity] for node in instance.nodes])[:,:, None] 

    demands = np.array([float(n.demand) for n in instance.nodes], dtype=np.float32)
    features = [demands] + [np.array([instance.vehicle_capacity for n in instance.nodes], dtype=np.float32)]

    # Detect CVRPTW vs CVRP
    has_tw = any(getattr(n, "ready_time", None) is not None for n in instance.nodes)

    if has_tw:
        ready = np.array([float(n.ready_time) for n in instance.nodes], dtype=np.float32)
        due   = np.array([float(n.due_time)   for n in instance.nodes], dtype=np.float32)
        serv  = np.array([float(n.service_time) for n in instance.nodes], dtype=np.float32)
        features += [ready, due, serv]

    # (N, F)
    x_nodes = np.stack(features, axis=1)[:, :, None]
    x_a = np.array([[cost, MST_bool[k], Voro_adj_bool[k], Clark_Wright_savings[k]] 
          for k, cost in enumerate(instance.arc_costs)])[:,:, None]
    arc_index = np.array(instance.arc_index)
    nb_vehicles = instance.nb_vehicles
    vehicle_capacity = instance.vehicle_capacity


    return (x_nodes, x_a, arc_index, nb_vehicles, vehicle_capacity)

def get_graph_additional_TW_3_features(
    sample_paths,
):
    """Translate a set of N sample instances into feature arrays for GNN.

    Features:
    - nodes: node (N x (num_clients + 1) x 1)
    - arcs: arc costs (N x m x 1)
    - arc_index : index of arcs with source and destination (N x 2 x m x 1)

    Parameters
    ----------
    sample_paths: list
        List of sample paths.

    Returns
    -------
    tuple
        "-element tuple containing node features, arc features and arc_index.

    """

    x_nodes = []
    x_connection = []
    x_arc_index = []
    nb_vehicles_list = []
    capacity_vehicle_list = []

    # Read samples from pickle files
    for sample_path in sample_paths:
        sample = load_sample(sample_path)
        # Prepare features (now returns 3 elements)
        x_no, x_a, arc_index, nb_vehicles, capacity_vehicle = get_graph_additional_TW_3_features_for_instance(
            sample["instance"], batch_dim=True
        )
        x_nodes.append(x_no)
        x_connection.append(x_a)
        x_arc_index.append(arc_index)
        nb_vehicles_list.append(nb_vehicles)
        capacity_vehicle_list.append(capacity_vehicle)

    # Concatenate along batch dimension

    return x_nodes, x_connection, x_arc_index, nb_vehicles_list, capacity_vehicle_list

def get_combined_raw_arc_features_for_instance(instance, batch_dim=True):
    """Translate instance into combined arc features for baseline models.

    Features:
    - arcs: demand at source node
    - arcs: demand at destination node
    - arcs: cost of arc

    Parameters
    ----------
    instance: object
        Instance with .nodes, .arc_index, and .arc_costs
    batch_dim: bool, optional
        Whether to add batch dimension. Default is True.

    Returns
    -------
    np.array
        Arc features of shape (E, 3) or (1, E, 3) if batch_dim=True
    """
    demands = np.array([node.demand for node in instance.nodes])
    arc_index = instance.arc_index  # shape (2, E)
    arc_costs = np.array(instance.arc_costs)  # shape (E,)

    # Extract source and destination demands
    demand_i = demands[arc_index[0]]  # shape (E,)
    demand_j = demands[arc_index[1]]  # shape (E,)

    # Stack features: shape (E, 3)
    arc_features = np.stack((demand_i, demand_j, arc_costs), axis=1)

    if batch_dim:
        arc_features = arc_features[None, :, :]  # shape (1, E, 3)

    return arc_features


def get_arc_features(sample_paths, features="combined_raw_arc_features"):
    """Translate a set of N sample instances into an arc feature array.

    Parameters
    ----------
    sample_paths: list
        List of sample paths.
    features: str, optional
        Type of features. Can be 'combined_raw_arc_features'.

    Returns
    -------
    np.array
        Edge features.

    """

    feature_fun = get_combined_raw_arc_features_for_instance

    x_connection = []
    # Read samples from pickle files
    for sample_path in sample_paths:
        sample = load_sample(sample_path)
        # Prepare features
        x_connection.append(feature_fun(sample["instance"]))

    x_connection = np.concatenate(x_connection)

    return x_connection


def get_num_random_cvrp_sols(n, exp_occ=100, avg_arcs_per_sol=None):
    """
    Estimate number of random CVRP solutions needed to observe each arc with a target frequency.

    Parameters
    ----------
    n : int
        Number of client nodes (excluding depot).
    exp_occ : int
        Target frequency for each arc.
    avg_arcs_per_sol : int, optional
        Average number of arcs used per solution. 

    Returns
    -------
    int
        Estimated number of random CVRP solutions needed.
    """
    total_nodes = n + 1
    total_possible_arcs = total_nodes * (total_nodes - 1) 

    if avg_arcs_per_sol is None:
        avg_arcs_per_sol = 2 * n  

    return int(exp_occ * (total_possible_arcs / avg_arcs_per_sol))


###################################
# Data set classes
###################################


# class CVRPData(Dataset):
#     """Dataset wrapper for CVRP training data, now returning PyG Data objects.

#     Parameters
#     ----------
#     x: tuple of (x_nodes, x_connections, arc_index)
#         - x_nodes: list/array of node feature matrices
#         - x_connections: list/array of edge feature matrices
#         - arc_index: list of edge index arrays (shape [E, 2] or [2, E])
#     y: np.array
#         Labels
#     """

#     def __init__(self, x, y):
#         super(CVRPData, self).__init__()
#         self.x_nodes, self.x_connections, self.arc_index, self.nb_vehicles, self.vehicle_capacity, self.x_weights = x
#         self.y = y

#     def __getitem__(self, index):
#         # Convert numpy arrays to torch tensors
#         x_nodes = torch.tensor(self.x_nodes[index], dtype=torch.float)
#         x_connections = torch.tensor(self.x_connections[index], dtype=torch.float)
#         # x_weights_list = torch.tensor(self.x_weights[index], dtype=torch.float)
#         # Ensure arc_index is shape [2, E]
#         arc_index = torch.tensor(self.arc_index[index], dtype=torch.long)

#         y = torch.tensor(self.y[index], dtype=torch.float)
#         nb_vehicles = torch.tensor(self.nb_vehicles[index], dtype=torch.float)
#         vehicle_capacity = torch.tensor(self.vehicle_capacity[index], dtype = torch.float)
#         demands_list = torch.tensor(self.x_nodes[index][0], dtype=torch.int )

#         # Build PyG Data object
#         data = Data(
#             x=x_nodes,                # node features
#             edge_index=arc_index,     # connectivity
#             edge_attr=x_connections,  # edge features
#             y=y,                       # label
#             nb_vehicles=nb_vehicles,
#             vehicle_capacity=vehicle_capacity,
#             demands = demands_list,
#             x_weights= 0
#         )
#         return data

#     def __len__(self):
#         return len(self.y)

class CVRPData(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample = torch.load(self.file_list[idx], weights_only=False)

        # print(torch.tensor(sample["y"], dtype=torch.float))

        return Data(
            x=torch.tensor(sample["x_nodes"], dtype=torch.float),
            edge_index=torch.tensor(sample["arc_index"], dtype=torch.long),
            edge_attr=torch.tensor(sample["x_connections"], dtype=torch.float),
            y=torch.tensor(sample["y"], dtype=torch.float),
            nb_vehicles=torch.tensor(sample["nb_vehicles"], dtype=torch.float),
            vehicle_capacity=torch.tensor(sample["vehicle_capacity"], dtype=torch.float),
            demands=torch.tensor(sample["x_nodes"][0], dtype=torch.int),
        )

    
def save_checkpoint(
    policy, exp_dict, epoch, model_config, training_config, chkpnt_path
):
    """Save pytorch model parameters, experiment dict and configs.

    Parameters
    ----------
    policy: BaseSolArcPredictor
        CVRP solution arc predictor.
    exp_dict: dict or defaultdict
        Experiment dictionary.
    epoch: int
        Training epoch.
    model_config: dict
        Model config dict.
    training_config: dict
        Training config dict.
    chkpnt_path: str
        Path to save checkpoint.

    Returns
    -------
    None

    """
    if policy.input_transformer is not None:
        normalizer_state = policy.input_transformer.get_state()
    else:
        normalizer_state = None
    torch.save(
        {
            "epoch": epoch,
            "model_config": model_config,
            "state_dict": policy.model.state_dict(),
            "normalizers": normalizer_state,
            "training_config": training_config,
            "exp_dict": default_to_regular_dict(exp_dict),
        },
        chkpnt_path,
    )

def training_wrapper(
    training_config,
    model_config,
    train_dataset,
    val_dataset,
    policy,
    out_dir,
    logger,
    eval_train_data=False,
    initial_eval_display_func=None,
    running_eval_display_func=None,
):
    """Model training wrapper function.

    Parameters
    ----------
    training_config: dict
        Training config dict.
    model_config: dict
        Model config dict.
    train_dataset: CVRPData
        Training data.
    val_dataset: CVRPData
        Validation data.
    policy: BaseSolArcPredictor
        CVRP solution arc predictor to be trained.
    out_dir: str
        Path to output directory to save checkpoints and logging file.
    logger: logging.Logger
        Logger object.
    eval_train_data: bool, optional
        Indicate whether training data performance should be evaluated after every epoch. Default
        is False.
    initial_eval_display_func: Function
        Function to display initial model performance as pretty string.
    running_eval_display_func: Function
        Function to display model performance after every epoch as pretty string.

    Returns
    -------
    exp_dict: dict or defaultdict
        Experiment dictionary.
    best_period: int
        Epoch of best model performance.
    best_kpi_val: float
        Value of monitoring KPI (loss, accuracy, etc.) in best epoch.

    """
    if initial_eval_display_func is None:
        initial_eval_display_func = lambda x: x

    if running_eval_display_func is None:
        running_eval_display_func = lambda x: x

    # Training
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=training_config.train_batch_size,
        shuffle=True,
    )

    # Validation
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=training_config.val_batch_size,
        shuffle=False,
    )

    logger.info(f"Number of training samples: {len(train_dataset)}")
    logger.info(f"Number of validation samples: {len(val_dataset)}")

    #######################################################
    # Training Loop
    #######################################################

    # Pre-configure saving function
    save_chkpnt = partial(
        save_checkpoint,
        model_config=model_config,
        training_config=dict(training_config),
    )

    exp_dict = defaultdict(lambda: [])
    
    if training_config.use_gpu:
        logger.info("using gpu")

    exp_dict["device"] = "gpu" if training_config.use_gpu else "cpu"
    exp_dict["num_train_samples"] = len(train_dataset)
    exp_dict["max_num_epochs"] = training_config.max_num_epochs
    exp_dict["initial_performance"] = defaultdict(lambda: [])
    exp_dict["performance"] = defaultdict(lambda: [])

    accum_steps = 32
    policy.optimizer.zero_grad()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.model = policy.model.to(device)

    logger.info(f"Model device: {next(policy.model.parameters()).device}")

    p, _, _, _, _, _ = policy.evaluate(
        {"train": train_loader, "validation": val_loader}
    )
    for key, value in p.items():
        exp_dict["initial_performance"][key].append(value)
    logger.info(("[Initial performance] " f"{initial_eval_display_func(p)}"))

    # Training loop
    best_period = 0
    best_kpi_val = np.inf
    total_start_time = time.time()
    logger.info("Starting training loop...")

    # W&B logging is fully gated by `training_config.wandb_use`.
    if training_config.wandb_use:
        wandb.login()

    # Project that the run is recorded to (read from config.yaml)
    project = training_config.project_name
    # Dictionary with hyperparameters
    config = {
        'num_samples': training_config.num_samples,
        'validation_num_samples': training_config.validation_num_samples,
        'var_size_samples': training_config.var_size_samples,

        'train_batch_size': training_config.train_batch_size,
        'val_batch_size': training_config.val_batch_size,

        'epochs': training_config.max_num_epochs,
        'learning_rate': training_config.learning_rate,
        'momentum': training_config.momentum,
        'weight_decay': training_config.weight_decay,

        'lr_decay': training_config.lr_decay,
        'lr_decay_factor': training_config.lr_decay_factor,
        'patience': training_config.patience,

        'early_stopping': training_config.early_stopping,

        'hidden_layer_dim': model_config.hidden_layer_dim,
        'conv_hidden_layer_dim': model_config.conv_hidden_layer_dim,
        'num_conv_layers': model_config.num_conv_layers,
        'num_dense_layers': model_config.num_dense_layers,

        'features': model_config.features,
        'accum_steps': accum_steps
    }

    _run_ctx = (
        wandb.init(project=project, config=config, name=training_config.experiment_name)
        if training_config.wandb_use
        else _NoOpWandbRun()
    )
    with _run_ctx as run:
        for i in range(training_config.max_num_epochs):

            policy.model.train()

            start_time = time.time()
            start_time_process = time.process_time()

            running_loss = 0
            accum_counter = 0
            for batch in train_loader:
                batch = batch.to(device)
                # Update model parameters
                loss, _, true_loss = policy.train_step(batch)
                # running_loss += loss.cpu().item()
                running_loss += true_loss.cpu().item()
                accum_counter += 1

                if accum_counter == accum_steps:
                    policy.optimizer.step()
                    policy.optimizer.zero_grad()
                    accum_counter = 0

            if accum_counter > 0:
                policy.optimizer.step()
                policy.optimizer.zero_grad()

            running_loss = running_loss / len(train_loader)

            train_time = time.time() - start_time
            train_time_process = time.process_time() - start_time_process
            exp_dict["train_time"].append(train_time)
            exp_dict["train_time_process"].append(train_time_process)
            exp_dict["performance"]["train_loss"].append(running_loss)

            # Evaluate model performance — only the branch dictated by
            # `eval_train_data` runs, and only its KPIs are logged to W&B.
            if eval_train_data:
                (
                    p,
                    tra_running_loss,
                    tra_running_acc,
                    tra_running_rec,
                    tra_running_prec,
                    tra_running_f,
                ) = policy.evaluate({"train": train_loader})
                run.log({
                    "training_accuracy":  tra_running_acc,
                    "training_loss":      tra_running_loss,
                    "training_recall":    tra_running_rec,
                    "training_precision": tra_running_prec,
                    "training_f_score":   tra_running_f,
                })
            else:
                (
                    p,
                    va_running_loss,
                    va_running_acc,
                    va_running_rec,
                    va_running_prec,
                    va_running_f,
                ) = policy.evaluate({"validation": val_loader})
                run.log({
                    "validation_accuracy":  va_running_acc,
                    "validation_loss":      va_running_loss,
                    "validation_recall":    va_running_rec,
                    "validation_precision": va_running_prec,
                    "validation_f_score":   va_running_f,
                })
            for key, value in p.items():
                exp_dict["performance"][key].append(value)

            # y_true, y_pred = [], []
            # policy.model.eval()
            # with torch.no_grad():
            #     for batch in val_loader:
            #         batch = batch.to(device)
            #         _, outputs,_ = policy.forward_pass(batch, FW_env,
            #                         top_k, regul_lambda, max_iterations_FW)
            #         preds = (outputs > 0.5).int().cpu().numpy()  # binary case
            #         y_true.extend(batch.y.cpu().numpy())
            #         y_pred.extend(preds)

            # cm = confusion_matrix(y_true, y_pred)
            cm = 0
            exp_dict["performance"]["confusion_matrix"].append(cm)

            # Adapt learning rate based on performance stagnation
            policy.lr_scheduler_step(p[training_config.monitoring_kpi])

            # Log current epoch performance and save checkpoint
            logger.info(
                (
                    f"[Epoch {i+1}/{training_config.max_num_epochs}] "
                    f"Running train loss: {np.round(running_loss, 4)} "
                    f"{running_eval_display_func(p)} "
                    f"({datetime.timedelta(seconds=train_time)})"
                )
            )
            save_chkpnt(
                policy,
                exp_dict,
                i + 1,
                chkpnt_path=os.path.join(out_dir, "checkpoint.pth.tar"),
            )

            # Check if model performance has improved
            if p[training_config.monitoring_kpi] < best_kpi_val * (
                1 - training_config.opt_threshold
            ):
                logger.info("Better model found!")
                best_kpi_val = p[training_config.monitoring_kpi]
                best_period = i
                save_chkpnt(
                    policy,
                    exp_dict,
                    i + 1,
                    chkpnt_path=os.path.join(out_dir, "best_checkpoint.pth.tar"),
                )

            # Early stopping if model performance has not improved over several epochs
            if i - best_period >= training_config.early_stopping:
                logger.info(f"No improvement since period {best_period + 1} -> Stopping")
                break

    total_train_time = time.time() - total_start_time
    logger.info(
        f"Training finished after {datetime.timedelta(seconds=total_train_time)}."
    )
    return exp_dict, best_period, best_kpi_val

############################
# Model pre- and postprocessing
############################


# def load_arc_predictor_model(model_path, get_feature_fun=True, get_label_fun=False):
#     """Helper function to load edge predictor model.

#     Parameters
#     ----------
#     model_path: str
#         Path to model checkpoint, containing model weights, normalizer states, and configs.
#     get_feature_fun: bool, optional
#         Indicate whether feature extraction function should be returned in addition to arc
#         predictor model. Default is True.
#     get_label_fun: bool, optional
#         Indicate whether label extraction function should be returned in addition to arc predictor
#         model. Default is True.

#     Returns
#     -------
#     tuple
#         1-element, 2-element, or 3-element tuple, comprising arc predictor model and potentially
#         feature extraction function and label extraction function.

#     """
#     return_vals = []

#     chkpnt = torch.load(model_path, map_location="cpu")
#     model_config = chkpnt["model_config"]
#     model_state = chkpnt["state_dict"]
#     normalizer_state = chkpnt["normalizers"]

#     # Configure policy
#     if model_config.model == "gcnn":
#         policy_fun = GCNNSolArcPredictor
#     else:
#         raise ValueError

#     # Configure normalizer
#     input_transformer = None
#     if model_config.normalization == "standard":
#         assert (
#             normalizer_state is not None
#         ), "Normalization parameters expected but are None"
#         normalizers = tuple([StandardNormalizer(*vals) for vals in normalizer_state])
#         input_transformer = MultiInputNormalizer(normalizers)

#     # Load model and parameters
#     policy = policy_fun(
#         model_config=model_config,
#         input_transformer=input_transformer,
#     )
#     policy.model.load_state_dict(model_state)
#     policy.model.eval()

#     if not get_feature_fun and not get_label_fun:
#         return policy

#     return_vals = [policy]

#     if get_feature_fun:
#         print("feature : ", model_config.features)
#         # Configure feature extraction function
#         if model_config.features == "raw":
#             feature_fun = get_graph_raw_features_for_instance
#         else:
#             if model_config.features=="additional_features":
#                 feature_fun = get_graph_additional_features_for_instance
#             else:
#                 raise ValueError
#         return_vals.append(feature_fun)

#     if get_label_fun:
#         # Configure label extraction function
#         if model_config.prediction_task == "binary_classification":
#             label_fun = partial(get_raw_target_for_instance, binary_target=True)
#         else:
#             raise ValueError
#         return_vals.append(label_fun)

#     return tuple(return_vals)


def get_raw_targets(
    sample_paths,
    binary_target=True,
    output_dim=True,
):
    """Translate a set of N sample solutions into labels for learning.

    Parameters
    ----------
    sample_paths: list
        List of sample paths.
    binary_target: bool, optional
        Indicate whether target should be binary (1 if flow >0, 0 otherwise).
    batch_dim: bool, optional
        Indicate whether batch dimension should be added.

    Returns
    -------
    np.array
        Target array.

    """
    y = []

    # Read samples from pickle files
    for sample_path in sample_paths:
        sample = load_sample(sample_path)
        sol = sol_to_list(sample["solution"], sample["instance"].arc_lookup)
        # Prepare target
        if output_dim:
            sol = sol[:, None]
        if binary_target:
            sol = np.where(sol > 0, 1, 0)
        y.append(sol)

    y = [np.array(target) for target in y]

    return y

def load_arc_predictor_model(model_path, get_feature_fun=True, get_label_fun=False):
    """Helper function to load arc predictor model.

    Parameters
    ----------
    model_path: str
        Path to model checkpoint, containing model weights, normalizer states, and configs.
    get_feature_fun: bool, optional
        Indicate whether feature extraction function should be returned in addition to arc
        predictor model. Default is True.
    get_label_fun: bool, optional
        Indicate whether label extraction function should be returned in addition to arc predictor
        model. Default is True.

    Returns
    -------
    tuple
        1-element, 2-element, or 3-element tuple, comprising arc predictor model and potentially
        feature extraction function and label extraction function.

    """
    return_vals = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    chkpnt = torch.load(model_path, map_location="cpu")
    model_config = chkpnt["model_config"]
    model_state = chkpnt["state_dict"]
    normalizer_state = chkpnt["normalizers"]

    # Configure policy
    if model_config.model == "gcnn":
        policy_fun = GCNNSolArcPredictor
    else:
        raise ValueError

    # Configure normalizer
    input_transformer = None
    if model_config.normalization == "standard":
        assert (
            normalizer_state is not None
        ), "Normalization parameters expected but are None"
        normalizers = tuple([StandardNormalizer(*vals) for vals in normalizer_state])
        input_transformer = MultiInputNormalizer(normalizers)

    # Load model and parameters
    policy = policy_fun(
        model_config=model_config,
        input_transformer=input_transformer,
    )
    policy.model.load_state_dict(model_state)
    policy.model.eval()
    policy.model.to(device)

    if not get_feature_fun and not get_label_fun:
        return policy

    print(f"Model device: {next(policy.model.parameters()).device}")

    return_vals = [policy]

    if get_feature_fun:
        # Configure feature extraction function
        if model_config.features == "graph_raw":
            feature_fun = get_graph_raw_features_for_instance
        else:
            if model_config.features == "graph_additional_1":
                feature_fun = get_graph_additional_1_features_for_instance
            elif model_config.features == "graph_additional_2":
                feature_fun = get_graph_additional_2_features_for_instance
            elif model_config.features == "graph_additional_3":
                feature_fun = get_graph_additional_3_features_for_instance
            elif model_config.features == "graph_additional_TW_1":
                feature_fun = get_graph_additional_TW_1_features_for_instance
            elif model_config.features == "graph_additional_TW_3":
                feature_fun = get_graph_additional_TW_3_features_for_instance
            else:
                raise ValueError
        return_vals.append(feature_fun)

    if get_label_fun:
        # Configure label extraction function
        if model_config.prediction_task == "binary_classification":
            label_fun = partial(get_raw_target_for_instance, binary_target=True)
        else:
            raise ValueError
        return_vals.append(label_fun)

    print(return_vals)

    return tuple(return_vals)

def save_dataset_to_pt(sample_paths, out_dir, feature_extractor):
    """Materialise per-sample feature/label tensors as `.pt` files in `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)

    for i, path in enumerate(sample_paths):
        x_nodes, x_connections, arc_index, nb_vehicles, vehicle_capacity = feature_extractor([path])
        y = get_raw_targets([path], binary_target=True, output_dim=True)

        sample = {
            "x_nodes": x_nodes[0],
            "x_connections": x_connections[0],
            "arc_index": arc_index[0],
            "nb_vehicles": nb_vehicles[0],
            "vehicle_capacity": vehicle_capacity[0],
            "y": y[0],
        }

        torch.save(sample, os.path.join(out_dir, f"sample_{i:05d}.pt"))

def load_labels_from_files(file_list):
    """Concatenate target labels (y) across the given list of `.pt` sample files."""
    labels = []
    for f in file_list:
        sample = torch.load(f, weights_only=False)
        labels.append(sample["y"])
    return labels
