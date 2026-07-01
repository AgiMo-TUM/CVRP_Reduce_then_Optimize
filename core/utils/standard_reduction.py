import numpy as np
def standard_pruning(cvrp_instance, used_method, relevant, top_k = 18):

    standard_pruning_count = 0
    n = cvrp_instance.n
    relevant_pruning = {(int(src), int(dst)): True
             for k, (src, dst) in enumerate(zip(cvrp_instance.arc_index[0], cvrp_instance.arc_index[1]))}

    if used_method == "demand_capacity":
        for key in relevant:
            relevant_pruning[key] = True
        for i in range(1,n-1):
            for j in range(i+1, n):
                if cvrp_instance.nodes[i].demand + cvrp_instance.nodes[j].demand > cvrp_instance.vehicle_capacity:
                    relevant_pruning[(i,j)] = False
                    standard_pruning_count += 1
                    if (j, i) in relevant_pruning:
                        relevant_pruning[(j,i)] = False
                        standard_pruning_count += 1

    if used_method == "depot_inequality":
        for key in relevant_pruning:
            relevant_pruning[key] = True
        cost_dict = {(int(src), int(dst)): cvrp_instance.arc_costs[k]
             for k, (src, dst) in enumerate(zip(cvrp_instance.arc_index[0], cvrp_instance.arc_index[1]))}
        depot_index  = 0
        for i in range(1,n):
            for j in range(i+1, n+1):
                # print(cost_dict[(i,j)]," ", cost_dict[(depot_index, i)], " ", cost_dict[(depot_index,j)])
                if cost_dict[(i,j)] > cost_dict[(depot_index, i)] + cost_dict[(depot_index,j)]:
                    relevant_pruning[(i,j)] = False
                    standard_pruning_count += 1
                    if (j, i) in relevant_pruning:
                        relevant_pruning[(j, i)] = False
                        standard_pruning_count += 1

    # if used_method == "nearest_neighbours":

    #     for key in relevant_pruning:
    #         relevant_pruning[key] = False

    #     for i in range(1, n + 1):

    #         outgoing = [] 
    #         for arc_id, (u, v) in enumerate(zip(cvrp_instance.arc_index[0], cvrp_instance.arc_index[1])): 
    #             if u == i:
    #                 outgoing.append(((u,v), cvrp_instance.arc_costs[arc_id]))
            
    #         outgoing_sorted = sorted(outgoing, key=lambda x: x[1]) 
    #         for (u,v), _ in outgoing_sorted[:top_k]:
    #             if (u, v) in relevant_pruning:
    #                 relevant_pruning[(u, v)] = True
    #     standard_pruning_count = sum(1 for k, v in relevant_pruning.items() if not v)

    #     for k,(u,v) in enumerate(zip(cvrp_instance.arc_index[0], cvrp_instance.arc_index[1])):
    #         relevant[k] = relevant_pruning[(u,v)]

    if used_method == "nearest_neighbours":

        for key in relevant_pruning:
            relevant_pruning[key] = False

        for i in range(0, n + 1):

            outgoing = [] 
            for arc_id, (u, v) in enumerate(zip(cvrp_instance.arc_index[0], cvrp_instance.arc_index[1])): 
                if u == i or v == i:
                    outgoing.append(((u,v), cvrp_instance.arc_costs[arc_id]))
            
            outgoing_sorted = sorted(outgoing, key=lambda x: x[1]) 
            for (u,v), _ in outgoing_sorted[:top_k]:
                if (u, v) in relevant_pruning:
                    relevant_pruning[(u, v)] = True
        standard_pruning_count = sum(1 for k, v in relevant_pruning.items() if not v)

        for k,(u,v) in enumerate(zip(cvrp_instance.arc_index[0], cvrp_instance.arc_index[1])):
            relevant[k] = relevant_pruning[(u,v)]

        num_arc_preds = np.sum(relevant)


    return num_arc_preds






# import itertools
# import csv

# conv_layers = [1, 2, 3, 4]
# hidden_dims = [20, 32, 64]
# batch_sizes = [32, 64]
# learning_rates = [1e-3, 1e-4]
# dense_layers = [1, 2]
# features = ["graph_raw", "graph_additional_1", "graph_additional_2", "graph_additional_3"]

# # Cartesian product
# all_configs = list(itertools.product(
#     conv_layers,
#     hidden_dims,
#     batch_sizes,
#     learning_rates,
#     dense_layers,
#     features
# ))

# output_file = "hyperparams.csv"

# with open(output_file, "w", newline="") as f:
#     writer = csv.writer(f, delimiter=';')
    
#     # Header in your required format
#     writer.writerow(["experiment_id", "conv_layers", "hidden_dims", "batch_sizes", "learning_rates", "dense_layers", "features"])
    
#     # Write each experiment
#     for i, (conv, hid, bs, lr, dense, feat) in enumerate(all_configs):
#         experiment_id = f"exp_{i:04d}"
#         writer.writerow([experiment_id, conv, hid , bs, lr, dense, feat])

# print(f"Generated {len(all_configs)} experiments into {output_file}")
