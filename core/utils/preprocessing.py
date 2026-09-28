""" Utility functions for pre-processing FCTP data. """

from copy import deepcopy

import networkx as nx
import numpy as np

from core.utils.postprocessing import list_to_dict
from core.utils.cvrp import CVRP


def get_cvrp_graph(arc_index):
    """Translate set of TP connections into networkx graph.

    Parameters
    ----------
    arc_index: tuple of np.array
        arc_index[0][k], arc_index[1][k] give the endpoints of arc k.
        Shape: (2, E) where E = number of arcs.

    Returns
    -------
    graph: nx.DiGraph
        CVRP graph including specified connections.

    """
    graph = nx.DiGraph()

    E = len(arc_index[0])

    for k in range(E):
        i, j = arc_index[0][k], arc_index[1][k]
        graph.add_edge(i, j)
    return graph