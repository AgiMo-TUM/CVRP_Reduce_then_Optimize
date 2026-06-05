""" Data utility functions. """

import gzip
import numpy as np
import pickle as pkl

# from core.utils.fctp import FCTP
# from core.utils.fctp import CapacitatedFCTP
# from core.utils.fctp import FixedStepFCTP

## change
from core.utils.cvrp import CVRP
from core.utils.cvrp import CVRP_node
##end change


# def dict_to_instance(instance_dict):
#     """Convert instance dict into proper FCTP class object.

#     Parameters
#     ----------
#     instance_dict: dict
#         Dictionary containing instance attributes.

#     Returns
#     -------
#     FCTP
#         Some FCTP instance object.

#     """
#     if instance_dict["instance_type"] == "fctp":
#         del instance_dict["instance_type"]
#         return FCTP(**instance_dict)
#     elif instance_dict["instance_type"] == "c-fctp":
#         del instance_dict["instance_type"]
#         return CapacitatedFCTP(**instance_dict)
#     elif instance_dict["instance_type"] == "fs-fctp":
#         del instance_dict["instance_type"]
#         return FixedStepFCTP(**instance_dict)
#     else:
#         raise ValueError


# def load_instance(instance_path):
#     """Load instance from path.

#     Parameters
#     ----------
#     instance_path: str
#         Path to pickle file with instance data.

#     Returns
#     -------
#     instance: FCTP
#         FCTP instance.

#     """
#     with gzip.open(instance_path, "rb") as file:
#         instance_dict = pkl.load(file)
#     return dict_to_instance(instance_dict)


def load_sample(sample_path):
    """Load sample from path.

    Parameters
    ----------
    sample_path: str
        Path to pickle file with sample data.

    Returns
    -------
    dict
        Sample dictionary.

    """
    with gzip.open(sample_path, "rb") as file:
        sample = pkl.load(file)
    sample["instance"], _ = dict_to_instance(sample)
    return sample



##Changes

def dict_to_instance(instance_dict):
    """Convert instance dict into proper CVRP class object.

    Parameters
    ----------
    instance_dict: dict
        Dictionary containing instance attributes.

    Returns
    -------
    CVRP
        Some CVRP instance object.
    """
    # if instance_dict["instance_type"] == "cvrp":
    if True:
        # Remove helper key
        instance_dict = dict(instance_dict)  # make a copy
        instance_dict.pop("instance_type", None)

        # Rebuild CVRP_node objects from dicts
        nodes = [CVRP_node(**n) for n in instance_dict["instance"]["nodes"]]

        # Ensure numpy arrays for arc_index and arc_costs
        arc_index = np.array(instance_dict["instance"]["arc_index"])
        arc_costs = np.array(instance_dict["instance"]["arc_costs"])

        return CVRP(
            nodes=nodes,
            arc_index=arc_index,
            vehicle_capacity=instance_dict["instance"]["vehicle_capacity"],
            arc_costs=arc_costs,
            nb_vehicles=instance_dict["instance"]["nb_vehicles"],
        ), instance_dict["solution"]
    else:
        raise ValueError



def load_instance(instance_path):
    """Load instance from path.

    Parameters
    ----------
    instance_path: str
        Path to pickle file with instance data.

    Returns
    -------
    instance: CVRP
        CVRP instance.

    """
    with gzip.open(instance_path, "rb") as file:
        instance_dict = pkl.load(file)
    return dict_to_instance(instance_dict)


# def load_sample(sample_path):
#     """Load sample from path.

#     Parameters
#     ----------
#     sample_path: str
#         Path to pickle file with sample data.

#     Returns
#     -------
#     dict
#         Sample dictionary.

#     """
#     with gzip.open(sample_path, "rb") as file:
#         sample = pkl.load(file)
#     sample["instance"] = dict_to_instance(sample["instance"])
#     return sample