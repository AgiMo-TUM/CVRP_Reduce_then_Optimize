import numpy as np
from scipy.spatial import Voronoi


def MST_feature(cvrp_instance):
    
    weighted_edges = []

    parent = list(range(cvrp_instance.n + 1)) #depot

    for k, (u,v) in enumerate (cvrp_instance.arc_list):
        if u < v :
            weighted_edges.append((u,v, cvrp_instance.arc_costs[k]))


    weighted_edges = sorted(weighted_edges, key = lambda x : x[2])

    mst = []

    for u, v, w in weighted_edges:
        or_u = u
        or_v = v
        while parent[u] != u:
            u = parent[u]
        while parent[v] != v:
            v = parent[v]
        if u!=v:
            parent[u] = v
            mst.append((or_u,or_v))

    mst_bool_list = []

    for (u,v) in cvrp_instance.arc_list:
        mst_bool_list.append((u,v) in mst or (v,u) in mst )
    return mst_bool_list

def compute_voronoi_adjacency(cvrp_instance):

    coords = np.array([(node.x, node.y) for node in cvrp_instance.nodes])

    vor = Voronoi(coords)

    adjacency = []

    for i, j in vor.ridge_points:
        if i < j:

            adjacency.append((i,j))

    adjacency_bool_list = []

    for (u,v) in cvrp_instance.arc_list:
        adjacency_bool_list.append((u,v) in adjacency or (v,u) in adjacency )

    return adjacency_bool_list

def compute_Clark_Wright_savings(cvrp_instance):

    Clark_Wright_savings = []

    cost_dict = {(int(src), int(dst)): cvrp_instance.arc_costs[k]
             for k, (src, dst) in enumerate(cvrp_instance.arc_list)}
    
    for (u,v) in cvrp_instance.arc_list:
        if (u !=0 and v!=0):
            Clark_Wright_savings.append(cost_dict[(0,u)] + cost_dict[(0,v)] - cost_dict[(u,v)])
        else:
            if (u==0):
                Clark_Wright_savings.append(cost_dict[(0,v)])
            if (v==0):
                Clark_Wright_savings.append(cost_dict[(u,0)])

    return Clark_Wright_savings

