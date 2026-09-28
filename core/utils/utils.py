""" General utility functions. """

from collections import defaultdict
from collections.abc import MutableMapping
import random
import string
import itertools
import csv

def default_to_regular_dict(d):
    """Helper function to recursively convert defaultdict into a regular Python dict.

    Parameters
    ----------
    d: defaultdict or dict
        Original dictionary.

    Returns
    -------
    d: dict
        Python dictionary.

    """
    if isinstance(d, defaultdict):
        d = {k: default_to_regular_dict(v) for k, v in d.items()}
    return d


def flatten_dict(d, parent_key="", sep="_"):
    """Helper function to recursively flatten nested dictionary.

    Parameters
    ----------
    d: defaultdict or dict
        Original dictionary.
    parent_key: str, optional
        Parent key. Deafult is ''.
    sep: str, optional
        Seperator to seperate keys. Default is '_'.

    Returns
    -------
    d: dict
        Flattened dictionary.

    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_random_string(k=8):
    """Get random string of specified length.

    Parameters
    ----------
    k: int, optional
        Length of random string.

    Returns
    -------
    str
        Randomly generated string.

    """
    return "".join(random.choices(string.ascii_lowercase, k=k))

def generate_hyperparameter_csv(): 

    conv_layers = [1, 2, 3, 4]
    hidden_dims = [20, 32, 64]
    batch_sizes = [32, 64]
    learning_rates = [1e-3, 1e-4]
    dense_layers = [1, 2]
    features = ["graph_raw", "graph_additional_1", "graph_additional_2", "graph_additional_3"]

    # Cartesian product
    all_configs = list(itertools.product(
        conv_layers,
        hidden_dims,
        batch_sizes,
        learning_rates,
        dense_layers,
        features
    ))

    output_file = "hyperparams.csv"

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=';')
        
        # Header in your required format
        writer.writerow(["experiment_id", "conv_layers", "hidden_dims", "batch_sizes", "learning_rates", "dense_layers", "features"])
        
        # Write each experiment
        for i, (conv, hid, bs, lr, dense, feat) in enumerate(all_configs):
            experiment_id = f"exp_{i:04d}"
            writer.writerow([experiment_id, conv, hid , bs, lr, dense, feat])

    print(f"Generated {len(all_configs)} experiments into {output_file}")
