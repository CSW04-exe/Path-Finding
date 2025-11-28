# ---------------------------------------------------------------
# Author : Carter Ward
# Class  : CS 330-1
# Date   : 11/07/2025
#
# Program 3: A* pathfinding on Adventure Bay.
# Loads the map node and connection files, builds the graph, then
# runs A* to find shortest paths between a few node pairs. The
# results are written to a text file for checking and plotting.
# ---------------------------------------------------------------

import csv
import math
from collections import defaultdict, deque

# ---------------------------------------------------------------
# File paths (update if needed)
# ---------------------------------------------------------------
NODES_FILE = "CS 330, Pathfinding, Graph AB Nodes v3.txt"
CONS_FILE  = "CS 330, Pathfinding, Graph AB Connections v3.txt"
OUT_FILE   = "pathfinding_output.txt"

# Required test pairs (start, goal)
TESTS = [(1, 29), (1, 38), (11, 1), (33, 66), (58, 43)]

# ---------------------------------------------------------------
# Load nodes (keep id, x, z, name)
# ---------------------------------------------------------------
def load_nodes(path):
    nodes = {}  # id -> {"x": float, "z": float, "name": str}
    with open(path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.reader(f, delimiter=",", quotechar='"', skipinitialspace=True)
        for row in rdr:
            if not row or row[0].startswith("#"):  # skip comments/blank
                continue
            if row[0].strip() != "N":
                continue
            node_id = int(row[1])
            x = float(row[7])
            z = float(row[8])
            name = row[11].replace("\\n", " ").strip()
            nodes[node_id] = {"x": x, "z": z, "name": name}
    return nodes

# ---------------------------------------------------------------
# Load directed edges (A -> B with cost)
# ---------------------------------------------------------------
def load_connections(path):
    edges = defaultdict(list)  # from_id -> list[(to_id, cost)]
    all_edges = []             # for printing
    with open(path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.reader(f, delimiter=",", quotechar='"', skipinitialspace=True)
        for row in rdr:
            if not row or row[0].startswith("#"):
                continue
            if row[0].strip() != "C":
                continue
            from_id = int(row[2])
            to_id   = int(row[3])
            cost    = float(row[4])
            edges[from_id].append((to_id, cost))
            all_edges.append((from_id, to_id, cost))
    return edges, all_edges

# ---------------------------------------------------------------
# Heuristic: straight-line distance in (x,z)
# ---------------------------------------------------------------
def h(nodes, a, b):
    dx = nodes[b]["x"] - nodes[a]["x"]
    dz = nodes[b]["z"] - nodes[a]["z"]
    return math.hypot(dx, dz)

# ---------------------------------------------------------------
# A* (with "reopen-closed" on improvement, per lecture)
# Returns (path_list, total_cost). Empty path means no route.
# ---------------------------------------------------------------
def astar(nodes, edges, start, goal):
    open_set = {start}
    closed_set = set()

    g_cost = defaultdict(lambda: math.inf)
    f_cost = defaultdict(lambda: math.inf)
    prev   = {}

    g_cost[start] = 0.0
    f_cost[start] = h(nodes, start, goal)

    while open_set:
        # node in Open with the lowest f (tie-break on id)
        current = min(open_set, key=lambda n: (f_cost[n], n))

        if current == goal:
            # rebuild path
            path = deque([goal])
            while path[0] in prev:
                path.appendleft(prev[path[0]])
            return list(path), g_cost[goal]

        open_set.remove(current)
        closed_set.add(current)

        # relax outgoing edges
        for to_id, cost in edges.get(current, []):
            tentative_g = g_cost[current] + cost

            # better route to a Closed node → reopen it
            if to_id in closed_set and tentative_g < g_cost[to_id]:
                closed_set.remove(to_id)
                open_set.add(to_id)

            # add unseen neighbors to Open
            if to_id not in open_set and to_id not in closed_set:
                open_set.add(to_id)
            # if not an improvement, skip
            elif tentative_g >= g_cost[to_id]:
                continue

            # record best-so-far route
            prev[to_id]   = current
            g_cost[to_id] = tentative_g
            f_cost[to_id] = tentative_g + h(nodes, to_id, goal)

    return [], math.inf  # no path

# ---------------------------------------------------------------
# Output formatting helpers
# ---------------------------------------------------------------
def fmt_node_line(nid, nd):
    name = nd["name"]
    name_part = f' "{name}"' if name else ""
    return f"N {nid} x={nd['x']:.1f} z={nd['z']:.1f}{name_part}"

def fmt_edge_line(fr, to, w):
    return f"C {fr} -> {to}  cost={w:.1f}"

def fmt_path_line(start, goal, path, cost):
    seq = " ".join(str(n) for n in path) if path else "(no path)"
    cost_str = "∞" if math.isinf(cost) else f"{cost:.1f}"
    return f"Path {start} -> {goal}: {seq}   cost={cost_str}"

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    nodes = load_nodes(NODES_FILE)
    edges, all_edges = load_connections(CONS_FILE)

    with open(OUT_FILE, "w", encoding="utf-8") as out:
        out.write("CS 330 — Pathfinding Adventure Bay (A*)\n")
        out.write("Graph summary\n\n")

        # nodes table
        out.write("Nodes\n")
        for nid in sorted(nodes.keys()):
            out.write(fmt_node_line(nid, nodes[nid]) + "\n")
        out.write("\n")

        # connections table
        out.write("Connections\n")
        for fr, to, w in sorted(all_edges, key=lambda t: (t[0], t[1])):
            out.write(fmt_edge_line(fr, to, w) + "\n")
        out.write("\n")

        # requested paths
        out.write("Requested paths\n")
        for s, g in TESTS:
            path, cost = astar(nodes, edges, s, g)
            out.write(fmt_path_line(s, g, path, cost) + "\n")

    print(f"Done. Results written to {OUT_FILE}")

# ---------------------------------------------------------------
if __name__ == "__main__":
    main()
