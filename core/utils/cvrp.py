# """Classes and functionalities to represent FCTP instances."""
##Changes
"""Classes and functionalities to represent CVRP instances."""
##End changes

import numpy as np


# class FCTP:
#     """Class representing FCTP instance.

#     Attributes
#     ----------
#     supply: np.array
#         Supply vector.
#     demand: np.array
#         Demand vector.
#     var_costs: np.array
#         Variable cost matrix.
#     fix_costs: np.array
#         Fixed-cost matrix.
#     supplier_locations: np.array, optional
#         Supplier locations (m x 2).
#     customer_locations: np.array, optional
#         Customer locations (m x 2).

#     """

#     def __init__(
#         self,
#         supply,
#         demand,
#         var_costs,
#         fix_costs,
#         supplier_locations=None,
#         customer_locations=None,
#     ):
#         self.supply = supply
#         self.demand = demand
#         self.var_costs = var_costs
#         self.fix_costs = fix_costs
#         self.supplier_locations = supplier_locations
#         self.customer_locations = customer_locations

#     @property
#     def m(self):
#         """Number of supply nodes."""
#         return len(self.supply)

#     @property
#     def n(self):
#         """Number of demand nodes."""
#         return len(self.demand)

#     def to_dict(self):
#         attributes = vars(self)
#         attributes["instance_type"] = "fctp"
#         return attributes

#     def eval_sol_dict(self, solution):
#         """Compute total costs for solution provided as dict.

#         Parameters
#         ----------
#         solution: dict
#             Solution dictionary containing flows from supplier i to customer j.

#         Returns
#         -------
#         float or int
#             Total costs incurred by solution.

#         """
#         v_costs = sum([v * self.var_costs[i, j] for (i, j), v in solution.items()])
#         f_costs = sum([self.fix_costs[i, j] for (i, j), v in solution.items() if v > 0])
#         return v_costs + f_costs

#     def eval_sol_matrix(self, solution):
#         """Compute total costs for solution provided as matrix.

#         Parameters
#         ----------
#         solution: np.array
#             Solution matrix (m x n) containing flows from supplier i to customer j.

#         Returns
#         -------
#         float or int
#             Total costs incurred by solution.

#         """
#         return np.sum(
#             solution * self.var_costs + np.where(solution > 0, 1, 0) * self.fix_costs,
#         )


# class CapacitatedFCTP(FCTP):
#     """Class representing capacitated FCTP instance.

#     Attributes
#     ----------
#     supply: np.array
#         Supply vector.
#     demand: np.array
#         Demand vector.
#     var_costs: np.array
#         Variable cost matrix (m x n).
#     fix_costs: np.array
#         Fixed-cost matrix (m x n).
#     edge_capacities: np.array
#         Edge capacity matrix (m x n).
#     supplier_locations: np.array, optional
#         Supplier locations (m x 2).
#     customer_locations: np.array, optional
#         Customer locations (m x 2).

#     """

#     def __init__(
#         self,
#         supply,
#         demand,
#         var_costs,
#         fix_costs,
#         edge_capacities,
#         supplier_locations=None,
#         customer_locations=None,
#     ):
#         super().__init__(
#             supply, demand, var_costs, fix_costs, supplier_locations, customer_locations
#         )

#         self.edge_capacities = edge_capacities

#     def to_dict(self):
#         attributes = vars(self)
#         attributes["instance_type"] = "c-fctp"
#         return attributes


# class FixedStepFCTP(FCTP):
#     """Class representing capacitated FCTP instance with fixed-step costs.

#     Attributes
#     ----------
#     supply: np.array
#         Supply vector.
#     demand: np.array
#         Demand vector.
#     var_costs: np.array
#         Variable cost matrix (m x n).
#     fix_costs: np.array
#         Fixed-cost matrix (m x n).
#     vehicle_capacities: np.array
#         Vehicle capacity matrix (m x n).
#     supplier_locations: np.array, optional
#         Supplier locations (m x 2).
#     customer_locations: np.array, optional
#         Customer locations (m x 2).

#     """

#     def __init__(
#         self,
#         supply,
#         demand,
#         var_costs,
#         fix_costs,
#         vehicle_capacities,
#         supplier_locations=None,
#         customer_locations=None,
#     ):
#         super().__init__(
#             supply, demand, var_costs, fix_costs, supplier_locations, customer_locations
#         )

#         self.vehicle_capacities = vehicle_capacities

#     def to_dict(self):
#         attributes = vars(self)
#         attributes["instance_type"] = "fs-fctp"
#         return attributes

#     def eval_sol_dict(self, solution):
#         """Compute total costs for solution provided as dict.

#         Parameters
#         ----------
#         solution: dict
#             Solution dictionary containing flows from supplier i to customer j.

#         Returns
#         -------
#         float or int
#             Total costs incurred by solution.

#         """
#         v_costs = sum([v * self.var_costs[i, j] for (i, j), v in solution.items()])
#         f_costs = sum(
#             [
#                 self.fix_costs[i, j] * np.ceil(v / self.vehicle_capacities[i, j])
#                 for (i, j), v in solution.items()
#                 if v > 0
#             ]
#         )
#         return v_costs + f_costs

#     def eval_sol_matrix(self, solution):
#         """Compute total costs for solution provided as matrix.

#         Parameters
#         ----------
#         solution: np.array
#             Solution matrix (m x n) containing flows from supplier i to customer j.

#         Returns
#         -------
#         float or int
#             Total costs incurred by solution.

#         """
#         v_costs = solution * self.var_costs
#         f_costs = np.ceil(solution / self.vehicle_capacities) * self.fix_costs
#         return np.sum(v_costs + f_costs)
    

##Changes

class CVRP_node:
    """Class representing node for CVRP instances

    Attributes
    ----------

    node_id: int
        Node id.
    demand: float
        Node demand.
    x: float
        Node x-coordinate.
    y: float
        Node y-coordinate    

    """

    def __init__(
            self,
            node_id,
            demand,
            x = None,
            y = None
    ):
        self.node_id = node_id
        self.demand = demand
        self.x = x
        self.y = y

    def to_dict(self):
        attributes = vars(self)
        return attributes
    
    

class CVRP:
    """Class representing CVRP instance.

    Attributes
    ----------
        nodes : list of CVRP_nodes
            List of nodes including depot and clients.
        arc_index : 2D np.array
            List of arcs with source and destination
        vehicle_capacity: float
            Maximum capacity of ecah vehicle.
        arc_costs : np array
            Variable cost matrix.
        nb_vehicles : int
            Number of vehicles.
        arc_lookup : dict
            Retrive the index of an arc with the pair (i,j)

    """

    def __init__(
        self,
        nodes,
        arc_index,
        vehicle_capacity,
        arc_costs,
        nb_vehicles
    ):
        self.nodes = nodes
        self.arc_index = arc_index
        self.vehicle_capacity = vehicle_capacity
        self.arc_costs = arc_costs
        self.nb_vehicles = nb_vehicles
        self.arc_lookup = {
            (int(i), int(j)): idx
            for idx, (i, j) in enumerate(zip(arc_index[0], arc_index[1]))
        }


    @property
    def n(self):
        """Number of clients nodes."""
        return len(self.nodes[1:])
    
    @property
    def m(self):
        """Number of arcs."""
        return len(self.arc_index[0])
    
    @property
    def depot(self):
        """Depot node."""
        for node in self.nodes:
            if node.demand == 0:
                return node
        raise ValueError("Depot node not found (no node with demand == 0)")
    
    @property
    def clients(self):
        """Client nodes list."""
        return [node for node in self.nodes if node.demand > 0]
    
    @property
    def demands(self):
        """Demand values."""
        return [node.demand for node in self.nodes]
    @property
    def arc_list(self):
        """List of arcs"""
        return [(int(src), int(dst)) for src, dst in zip(self.arc_index[0], self.arc_index[1])]
    

    def to_dict(self):

        return {
            "nodes" : [node.to_dict() for node in self.nodes],
            "arc_index" : self.arc_index,
            # "arc_lookup" : self.arc_lookup,
            "vehicle_capacity" : self.vehicle_capacity,
            "arc_costs" : self.arc_costs,
            "nb_vehicles" : self.nb_vehicles,
            "instance_type" :  "cvrp"
        }

    def eval_sol_dict(self, solution):
        """Compute total costs for solution provided as dict.

        Parameters
        ----------
        solution: dict
            Solution dictionary containing booleans for the activation of arcs.

        Returns
        -------
        float or int
            Total costs incurred by solution.

        """

        return sum(
            v * self.arc_costs[self.arc_lookup[(i, j)]]
            for (i, j), v in solution.items()
            if (i, j) in self.arc_lookup
        )

    def eval_sol_list(self, solution):
        """Compute total costs for solution provided as matrix.

        Parameters
        ----------
        solution: np.array
            Solution list containing booleans for the activation of arc (i,j).

        Returns
        -------
        float or int
            Total costs incurred by solution.

        """
        return np.sum(
            solution * self.arc_costs,
        )

