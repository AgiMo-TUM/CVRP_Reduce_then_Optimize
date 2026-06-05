import gurobipy as gp
from gurobipy import GRB
import random
import math
import time

#start timer
t_total_start = time.time()


# =========================
# DATA GENERATION (same spirit as your model)
# =========================

n = 100
K = 30
random.seed(0)

V = list(range(n))
E = [(i, j) for i in V for j in V if i < j]

# pair weights theta_ij >= 0
theta = {(i, j): random.uniform(0, 1) for (i, j) in E}

# demands
d = {i: random.randint(1, 10) for i in V}
total_demand = sum(d.values())

Q = 100  # large-enough capacity (can be made smaller)
print(f"Total demand = {total_demand}, Q = {Q}")

# =========================
# HELPER: pattern handling
# =========================

def pattern_cost(pattern, theta):
    """Return cost c_p = sum_{i<j in pattern} theta_ij."""
    cost = 0.0
    patt_list = list(pattern)
    for idx_i in range(len(patt_list)):
        i = patt_list[idx_i]
        for idx_j in range(idx_i + 1, len(patt_list)):
            j = patt_list[idx_j]
            if (i, j) in theta:
                cost += theta[i, j]
            elif (j, i) in theta:
                cost += theta[j, i]
    return cost

def build_initial_patterns(V, K, Q, d):
    """
    Build K initial feasible patterns that partition V.
    Simple sequential split, respecting Q (since Q is large here, it's trivial).
    """
    patterns = []
    start = 0
    remaining_customers = len(V)

    for kk in range(K):
        remaining_clusters = K - kk
        # allocate roughly equal number of customers to each remaining cluster
        size = math.ceil(remaining_customers / remaining_clusters)
        patt = []
        capacity_used = 0
        while len(patt) < size and start < len(V):
            i = V[start]
            if capacity_used + d[i] <= Q:
                patt.append(i)
                capacity_used += d[i]
            else:
                # Q is large in your setting, but just in case:
                break
            start += 1

        patterns.append(tuple(sorted(patt)))
        remaining_customers = len(V) - start

    return patterns

# =========================
# BUILD INITIAL PATTERNS
# =========================

patterns = build_initial_patterns(V, K, Q, d)
patterns = [p for p in patterns if len(p) > 0]
print("Initial patterns:", patterns)
all_customers_in_patterns = set()
for p in patterns:
    all_customers_in_patterns.update(p)

missing = set(V) - all_customers_in_patterns
print("Missing customers in initial patterns:", missing)

# store unique patterns
pattern_set = set(patterns)   # use tuples (sorted) so we can hash them
pattern_list = list(pattern_set)

# precompute costs
pattern_costs = {p: pattern_cost(p, theta) for p in pattern_list}

# =========================
# BUILD RESTRICTED MASTER PROBLEM (RMP)
# =========================

RMP = gp.Model("DW_RMP")
RMP.Params.OutputFlag = 1  # set to 0 to silence Gurobi

# Cover constraints: each customer i must be covered exactly once
cover_constr = {}
for i in V:
    lhs = gp.LinExpr()                 # empty LHS
    cover_constr[i] = RMP.addConstr(lhs == 1.0, name=f"cover_{i}")

# Cluster count constraint: select exactly K patterns
lhs = gp.LinExpr()
cluster_constr = RMP.addConstr(lhs == float(K), name="cluster_count")

# Pattern variables λ_p (continuous in CG phase)
lambda_vars = {}

def add_pattern_to_RMP(p):
    """
    Add a pattern p as a column to the RMP.
    """
    global RMP, lambda_vars
    if p in lambda_vars:
        return

    col = gp.Column()

    # coefficient 1 in each cover constraint of customers in p
    for i in p:
        col.addTerms([1.0], [cover_constr[i]])

    # coefficient 1 in cluster count constraint
    col.addTerms([1.0], [cluster_constr])

    var = RMP.addVar(
        obj=pattern_costs[p],
        vtype=GRB.CONTINUOUS,  # continuous in DW LP
        name=f"lambda_{len(lambda_vars)}",
        column=col
    )
    lambda_vars[p] = var

# add initial patterns
for p in pattern_list:
    add_pattern_to_RMP(p)

RMP.update()


# =========================
# CHECK INITIAL RMP FEASIBILITY
# =========================
RMP.optimize()
print("Initial RMP status:", RMP.status)

if RMP.status != GRB.OPTIMAL:
    print("Initial RMP is infeasible or not optimal. Status =", RMP.status)
    print("Computing IIS for debugging...")
    RMP.computeIIS()
    RMP.write("rmp_initial.ilp")
    print("IIS written to rmp_initial.ilp")
    raise SystemExit("Stopping because initial RMP is infeasible.")

# =========================
# COLUMN GENERATION LOOP
# =========================

def solve_subproblem(pi, sigma, theta, d, Q, V, E):
    """
    Subproblem:
    min   sum_{i<j} theta_ij y_ij - sum_i pi_i x_i - sigma
    s.t.  sum_i d_i x_i <= Q
          y_ij <= x_i, y_ij <= x_j, y_ij >= x_i + x_j - 1
          x_i, y_ij binary
    """
    sub = gp.Model("Subproblem")
    sub.Params.OutputFlag = 0

    x = sub.addVars(V, vtype=GRB.BINARY, name="x")
    y = sub.addVars(E, vtype=GRB.BINARY, name="y")

    # capacity constraint
    sub.addConstr(gp.quicksum(d[i] * x[i] for i in V) <= Q, name="capacity")

    # linking y = x_i AND x_j
    sub.addConstrs((y[i, j] >= x[i] + x[j] - 1 for (i, j) in E), name="lb_y")
    sub.addConstrs((y[i, j] <= x[i] for (i, j) in E), name="ub_y_i")
    sub.addConstrs((y[i, j] <= x[j] for (i, j) in E), name="ub_y_j")

    # objective: internal cost - duals - sigma
    obj = gp.quicksum(theta[i, j] * y[i, j] for (i, j) in E) \
          - gp.quicksum(pi[i] * x[i] for i in V) \
          - sigma

    sub.setObjective(obj, GRB.MINIMIZE)
    sub.optimize()

    if sub.status != GRB.OPTIMAL:
        return None, None, None

    # build pattern from x
    pattern = tuple(sorted(i for i in V if x[i].X > 0.5))
    reduced_cost = sub.ObjVal

    return pattern, reduced_cost, sub

# Column generation
t_cg_start = time.time() #CG timer start
EPS = 1e-6
max_iter = 10
for it in range(max_iter):
    print(f"\n=== Column generation iteration {it} ===")
    RMP.optimize()
    if RMP.status == GRB.OPTIMAL:
        print("RMP LP objective value:", RMP.ObjVal)
    else:
        print("RMP is not optimal CG:", RMP.status)

    # Get dual prices
    pi = {i: cover_constr[i].Pi for i in V}
    sigma = cluster_constr.Pi

    print("Duals pi:", pi)
    print("Dual sigma:", sigma)

    # Solve subproblem
    pattern, red_cost, sub = solve_subproblem(pi, sigma, theta, d, Q, V, E)

    if pattern is None:
        print("Subproblem not solved optimally.")
        break

    print("Subproblem pattern:", pattern, "reduced cost:", red_cost)

    # Stopping condition: no negative reduced cost
    if red_cost is None or red_cost > -EPS or len(pattern) == 0:
        print("No improving pattern found. Stopping column generation.")
        break

    # If new pattern, add to pattern set and RMP
    p_tuple = tuple(sorted(pattern))
    if p_tuple not in pattern_set:
        pattern_set.add(p_tuple)
        pattern_costs[p_tuple] = pattern_cost(p_tuple, theta)
        print("Adding new pattern:", p_tuple, "with cost", pattern_costs[p_tuple])
        add_pattern_to_RMP(p_tuple)
        RMP.update()
    else:
        print("Generated pattern already exists. Stopping.")
        break

t_cg = time.time() - t_cg_start
print(f"\nTotal column generation time: {t_cg:.4f} seconds")

print("\n=== End of column generation ===")
for v in RMP.getVars():
    v.vtype = GRB.BINARY

RMP.update()
RMP.optimize()
if RMP.status == GRB.OPTIMAL:
    print("RMP LP objective value:", RMP.ObjVal)
else:
    print("RMP is not optimal. Status =", RMP.status)
# =========================
# OPTIONAL: INTEGER MASTER WITH GENERATED COLUMNS
# =========================

print("\n=== Solving final integer master (on generated columns) ===")

# change variables to binary and re-solve as MIP (no further CG)
for var in lambda_vars.values():
    var.vtype = GRB.BINARY
RMP.update()
RMP.optimize()


if RMP.status == GRB.OPTIMAL:
    print("Final integer objective:", RMP.ObjVal)
    chosen_patterns = [p for p, var in lambda_vars.items() if var.X > 0.5]
    print("Chosen patterns (clusters):")
    for idx, p in enumerate(chosen_patterns):
        print(f"Cluster {idx}: {p}")
else:
    print("Integer master not optimal. Status:", RMP.status)

#end timer
t_total = time.time() - t_total_start
print(f"\n==============================")
print(f" TOTAL DW SOLVE TIME: {t_total:.4f} seconds")
print(f"==============================")