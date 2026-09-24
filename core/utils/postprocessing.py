""" Utility functions for post-processing solutions. """

from collections import defaultdict

import networkx as nx
import numpy as np

def sol_to_list(solution, arc_lookup, shape=None):
    """Convert a solution-dict into a solution-list.

    Parameters
    ----------
    solution: dict
        Solution dictionary containing acivation variables for the ith arc.
    arc_lookup: dict
        Index dictionary with key (i,j)
    shape: tuple, optional
        Target shape. If it is not provided, it is derived from the dictionary keys.

    Returns
    -------
    sol_list: 1D np.array
        Solution list containing activation variables for ith arc.

    """
    if shape is None:
        shape = len(arc_lookup)
    sol_list = np.zeros(shape)
    for  (i, j), v in solution.items():
        sol_list[arc_lookup[(i,j)]] = v
    return sol_list


def list_to_dict(sol_list, arc_index):
    """Convert a solution-list into a solution-dict.

    Parameters
    ----------
    sol_list: 1D np.array
        Solution list containing activation variable for ith-arc.

    Returns
    -------
    dict
        Solution dictionary containing activation variable for arc (i,j).

    """
    
    return {
        (arc_index[0][k], arc_index[1][k]): sol_list[k]
        for k in range(len(sol_list))
        if sol_list[k] != 0
    }

def sol_to_graph(solution):
    """Convert a solution-dict into a directed networkx graph.

    Parameters
    ----------
    solution: dict
        Solution dictionary containing activation variable for arc (i,j).

    Returns
    -------
    graph: nx.DiGraph
        Directed graph containing the activated arcs for a CVRP instance.
    """
    graph = nx.DiGraph()

    nodes = {i for (i, j) in solution.keys()} | {j for (i, j) in solution.keys()}
    graph.add_nodes_from(nodes)

    for (i, j), v in solution.items():
        if v > 0:
            graph.add_edge(i, j, activation=v)

    return graph