# """ LP/MIP models and helper functions to solve TP and FCTP instances. """

##Changes
""" LP/MIP models and helper functions to solve CVRP instances. """
##End changes

import random

# import gurobipy as gp
import networkx as nx
import numpy as np
from VRPSolverEasy.src import solver
from time import time




##################################################
# Helper functions
##################################################


def sol_vals(var_dict):
    """Translate solution into a dictionary.

    Parameters
    ----------
    var_dict: dict
        Dictionary containing model variables.

    Returns
    -------
    dict
        Dictionary containing variable values.

    """
    return {k: np.round(v.X) for k, v in var_dict.items()}




def cvrp(demands, arc_index, arc_costs, nb_vehicles, vehicle_capacity, sol_dict = None):
    """Generate Gurobi model for the CVRP.

    Parameters
    ----------
    demands: 1D np.array or list
        A list of customer demands, a null demand corresponds to the depot.
    arc_index : 2D np.array
        List of arcs with source and destination
    costs: 1D np.array
        Costs for using the ith arc.
    nb_vehicles: int
        Number of vehicles.
    vehicle_capacity : int or float
        Uniform capacity for the vehicles.
    relax : bool, optional
        Indicate whether the cost variables should be linearized. Default is False.


    Returns
    -------
    m: gp.Model
        Gurobi model of TP instance.
    x: dict
        Dictionary of arc activation variables, keyed by (node.id(i), node.id(j)).
    u: dict
        Dictionnary of used capacity after delivering city j, keyed by node.id(j).

    """



    m = gp.Model()

    arc_list = [(int(src), int(dst)) for src, dst in zip(arc_index[0], arc_index[1])]

    cost_dict = {(int(src), int(dst)): arc_costs[k]
             for k, (src, dst) in enumerate(zip(arc_index[0], arc_index[1]))}
    # print(cost_dict)

    x = m.addVars(arc_list, obj=cost_dict, vtype=gp.GRB.BINARY, name="x")

    if sol_dict is not None:
        for arc, value in sol_dict.items():
            if arc in x:
                x[arc].start = value

    # The activation variables tell whether an arc is used to travel from a place to another

    u = m.addVars(len(demands), vtype=gp.GRB.CONTINUOUS, name="u")

    # For each pair of customers, first MTZ inequality
    for i, demand_i in enumerate(demands):
        for j, demand_j in enumerate(demands):
            if i != j and demand_i > 0 and demand_j > 0 and (i, j) in arc_list:
                m.addConstr(
                    u[j] - u[i] >= demand_j - vehicle_capacity * (1 - x[i, j]),
                    name=f"MTZ1_{i}_{j}"
                )

    # For each customer, second MTZ inequality
    for j, demand_j in enumerate(demands):
        if(demand_j > 0):
            m.addConstr(u[j] >= demand_j, name=f"MTZ2_lb{j}")
            m.addConstr(u[j] <= vehicle_capacity, name=f"MTZ2_ub{j}")

    # Constraint for the number of vehicles leaving the depot
    for i, demand in enumerate(demands):
        if(demand == 0): 
            m.addConstr(
                gp.quicksum(x[i, j] for j in range(len(demands)) 
                            if (i, j) in x and i != j ) <= nb_vehicles,
                name=f"CV{i}",
            )
            break
    # Leaving customer constraint
    for i, demand in enumerate(demands):
        if(demand > 0):
            m.addConstr(
                gp.quicksum(x[i, j] for j in range(len(demands))
                            if (i, j) in x and i != j) == 1,
                name=f"LC{i}"
            )
    
    #Serving customer constraint
    for j, demand in enumerate(demands):
        if(demand > 0):
            m.addConstr(
                gp.quicksum(x[i, j] for i in range(len(demands))
                            if (i, j) in x and i != j) == 1,
                name=f"SC{j}"
            )

    return m, x, u


def cvrp_subset_connections(demands, arc_index, arc_costs, nb_vehicles,
                            vehicle_capacity, connections, relax = False):
    """Generate Gurobi model for the TP.

    Parameters
    ----------
    demands: 1D np.array or list
        A list of customer demands, a null demand corresponds to the depot.
    arc_index : 2D np.array
        A list of arcs with source and destination
    costs: 2D np.array
        Costs for using arc from  i to j.
    nb_vehicles: int
        Number of vehicles.
    vehicle_capacity : int or float
        Uniform capacity for the vehicles.
    connections: 1D np.array
        Binary list indicating whether connection should be included.
    relax : bool, optional
        Indicate whether the cost variables should be linearized. Default is False.


    Returns
    -------
    m: gp.Model
        Gurobi model of TP instance.
    x: dict
        Dictionary of arc activation variables, keyed by (node.id(i), node.id(j)).
    u: dict
        Dictionnary of used capacity after delivering city j, keyed by node.id(j).

    """

    m = gp.Model()


    cost_dict = {(int(src), int(dst)): arc_costs[k]
             for k, (src, dst) in enumerate(zip(arc_index[0], arc_index[1]))
             if connections[k]}

    arc_list = [
        (int(i), int(j))
        for idx, (i, j) in enumerate(zip(arc_index[0], arc_index[1]))
        if connections[idx]
    ]

    # The activation variables tell whether an arc is used to travel from a place to another
    if relax:
        x = m.addVars(arc_list, obj=cost_dict, vtype=gp.GRB.CONTINUOUS, name="x")
    else:
        x = m.addVars(arc_list, obj=cost_dict, vtype=gp.GRB.BINARY, name="x")

    u = m.addVars(len(demands), vtype=gp.GRB.CONTINUOUS, name="u")

    # For each pair of customers, first MTZ inequality
    for i, demand_i in enumerate(demands):
        for j, demand_j in enumerate(demands):
            if(i!=j and demand_i > 0 and demand_j > 0 and (i, j) in x):
                m.addConstr(u[j] - u[i] >= demand_j - vehicle_capacity*(1 - x[i, j]),
                            name = f"MTZ1{i,j}")

    # For each customer, second MTZ inequality
    for j, demand_j in enumerate(demands):
        if(demand_j > 0):
            m.addConstr(u[j] >= demand_j, name=f"MTZ2_lb{j}")
            m.addConstr(u[j] <= vehicle_capacity, name=f"MTZ2_ub{j}")

    # Constraint for the number of vehicles leaving the depot
    for i, demand in enumerate(demands):
        if(demand == 0): 
            m.addConstr(
                gp.quicksum(x[i, j] for j in range(len(demands)) 
                            if (i, j) in x and i != j) <= nb_vehicles,
                name=f"CV{i}",
            )
            break
    # Leaving customer constraint
    for i, demand in enumerate(demands):
        if(demand > 0):
            m.addConstr(
                gp.quicksum(x[i, j] for j in range(len(demands))
                            if (i, j) in x and i != j) == 1,
                name=f"LC{i}"
            )
    
    #Serving customer constraint
    for j, demand in enumerate(demands):
        if(demand > 0):
            m.addConstr(
                gp.quicksum(x[i, j] for i in range(len(demands))
                            if (i, j) in x and i != j) == 1,
                name=f"SC{j}"
            )

    return m, x, u


def cvrp_via_VRP_Easy(demands, arc_index, arc_costs, nb_vehicles,
                            vehicle_capacity, connections, time_limit=5000,
                            upper_bound=None, cluster=None):
    
    start_time = time()

    cost_dict = {(int(src), int(dst)): arc_costs[k]
             for k, (src, dst) in enumerate(zip(arc_index[0], arc_index[1]))
             if connections[k]}

    arc_list = [
        (int(i), int(j))
        for idx, (i, j) in enumerate(zip(arc_index[0], arc_index[1]))
        if connections[idx]
    ]

    sol_dict = {(int(src), int(dst)): 0
            for (src, dst) in (zip(arc_index[0], arc_index[1]))}
    
    model = solver.Model()

    # add vehicle type
    model.add_vehicle_type(id=1,
                           start_point_id=0,
                           end_point_id=0,
                           max_number=int(nb_vehicles),
                           capacity=int(vehicle_capacity),
                           var_cost_dist=1
                           )
    # add depot
    model.add_depot(id=0)

    if cluster!=None:
        for i in range(1, len(demands)):
            model.add_customer(id=cluster[i],
                            demand=int(demands[i])
                            )
    else:
        for i in range(1, len(demands)):
            model.add_customer(id=i,
                            demand=int(demands[i])
                            )
        
    for arc in arc_list:
        model.add_link(start_point_id=arc[0],
                       end_point_id=arc[1],
                       distance=int(cost_dict[arc])
                       )
        
    # model.set_parameters(print_level=-2)
    # if upper_bound!= None:
    #     print("upper_bound = ", upper_bound)
    #     model.set_parameters(time_limit=time_limit,
    #                      solver_name="CPLEX", print_level=-2, upper_bound=upper_bound)
    # else:
    #     model.set_parameters(time_limit=time_limit,
    #                      solver_name="CPLEX", print_level=-2)

    model.set_parameters(time_limit=50, print_level=-2)
    model.solve()
    build_solver_runtime  = time() - start_time

    # if model.solution.is_defined :
    #     print(f"""Statistics :
    #     best lower bound : { model.statistics.best_lb } 
        
    #     solution time : {model.statistics.solution_time}
        
    #     number of nodes : {model.statistics.nb_branch_and_bound_nodes}
        
    #     solution value : {model.solution.value}

    #     root lower bound : {model.statistics.root_lb}

    #     root root time : {model.statistics.root_time}.
    #     """)
    #     print(f"Status : {model.status}.\n")
    #     print(f"Message : {model.message}.\n")   
    #     for route in model.solution.routes:            
    #         print(f"Vehicle Type id : {route.vehicle_type_id}.")
    #         print(f"Ids : {route.point_ids}.")
    #         print(f"Load : {route.cap_consumption}.\n")
    #         for i in range(len(route.point_ids) - 1):
    #             if route.point_ids[i] < route.point_ids[i+1]:
    #                 sol_dict[(route.point_ids[i], route.point_ids[i+1])] = 1
    #             else :
    #                 sol_dict[(route.point_ids[i+1], route.point_ids[i])] = 1
    return sol_dict, model.statistics.solution_time, model.solution.value, model.statistics.best_lb, model.status, build_solver_runtime

def cvrpTW_via_VRP_Easy(cvrptw_nodes, arc_index, arc_costs, nb_vehicles,
                            vehicle_capacity, connections, undirected=False, time_limit=5000,
                            upper_bound=None):
    
    start_time = time()
    demands = [node.demand for node in cvrptw_nodes]
    cost_dict = {(int(src), int(dst)): arc_costs[k]
             for k, (src, dst) in enumerate(zip(arc_index[0], arc_index[1]))
             if connections[k]}

    arc_list = [
        (int(i), int(j))
        for idx, (i, j) in enumerate(zip(arc_index[0], arc_index[1]))
        if connections[idx]
    ]


    sol_dict = {(int(src), int(dst)): 0
            for (src, dst) in (zip(arc_index[0], arc_index[1]))}
    
    model = solver.Model()

    model.add_vehicle_type(id=1,
                           start_point_id=0,
                           end_point_id=0,
                           max_number=int(nb_vehicles),
                           capacity=int(vehicle_capacity),
                           tw_begin=int(cvrptw_nodes[0].ready_time),
                           tw_end=int(cvrptw_nodes[0].due_time),
                           var_cost_dist=1
                           )

    model.add_depot(id=0,
                    service_time=int(cvrptw_nodes[0].service_time),
                    tw_begin=int(cvrptw_nodes[0].ready_time),
                    tw_end=int(cvrptw_nodes[0].due_time)
                   )

    for i in range(1, len(demands)):
        model.add_customer(id=i,
                           demand=int(demands[i]),
                           service_time=int(cvrptw_nodes[i].service_time),
                           tw_begin=int(cvrptw_nodes[i].ready_time),
                           tw_end=int(cvrptw_nodes[i].due_time),
                          )
        
    for arc in arc_list:
        model.add_link(start_point_id=arc[0],
                       end_point_id=arc[1],
                       distance=int(cost_dict[arc]),
                       time=int(cost_dict[arc])
                       )
        if undirected:
            model.add_link(start_point_id=arc[1],
                       end_point_id=arc[0],
                       distance=int(cost_dict[arc]),
                       time=int(cost_dict[arc])
                       )
        
    # model.set_parameters(print_level=-2)
    # if upper_bound!= None:
    #     print("upper_bound = ", upper_bound)
    #     model.set_parameters(time_limit=time_limit,
    #                      solver_name="CPLEX", print_level=-2, upper_bound=upper_bound)
    # else:
    #     model.set_parameters(time_limit=time_limit,
    #                      solver_name="CPLEX", print_level=-2)
    if upper_bound!= None:
        print("upper_bound = ", upper_bound)
        model.set_parameters(time_limit=time_limit, print_level=-2, upper_bound=upper_bound)
    else:
        model.set_parameters(time_limit=time_limit,
                         solver_name="CPLEX", print_level=-2)

    
    model.solve()
    build_solver_runtime = time() - start_time

    if model.solution.is_defined :
        print(f"""Statistics :
        best lower bound : { model.statistics.best_lb } 
        
        solution time : {model.statistics.solution_time}
        
        number of nodes : {model.statistics.nb_branch_and_bound_nodes}
        
        solution value : {model.solution.value}

        root lower bound : {model.statistics.root_lb}

        root root time : {model.statistics.root_time}.
        """)
        print(f"Status : {model.status}.\n")
        print(f"Message : {model.message}.\n")   
        for route in model.solution.routes:            
            print(f"Vehicle Type id : {route.vehicle_type_id}.")
            print(f"Ids : {route.point_ids}.")
            print(f"Load : {route.cap_consumption}.\n")
            for i in range(len(route.point_ids) - 1):
                sol_dict[(route.point_ids[i], route.point_ids[i+1])] = 1

    return sol_dict, model.statistics.solution_time, model.solution.value, model.statistics.best_lb, model.status, build_solver_runtime