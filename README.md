# Path-Finding

## Purpose

This is Program 3 for CS 330 (Artificial Intelligence), a course assignment where I implement the A* search algorithm and use it to find optimal routes through a graph. The graph represents a fictional map called "Adventure Bay," made up of numbered nodes (locations) connected by weighted, directed edges (paths between them, with a travel cost). The assignment exists to prove out A* as a real, working algorithm rather than just a concept on a whiteboard — reading a graph from data files, searching it, and reporting exact paths and costs for a required set of start/goal pairs.

## Problem and Approach

The problem is: given the Adventure Bay graph (66 nodes, over 100 directed connections) and five specific (start, goal) node pairs — (1, 29), (1, 38), (11, 1), (33, 66), (58, 43) — find the lowest-cost path for each pair and report the full path and its total cost.

My approach was to separate the work into three clean stages in `WardCS330Program3.py`:

1. **Parse** the two provided data files (`CS 330, Pathfinding, Graph AB Nodes v3.txt` and `CS 330, Pathfinding, Graph AB Connections v3.txt`) into in-memory Python structures — a dictionary of nodes (each with an x/z coordinate and an optional name) and an adjacency list of directed edges with costs.
2. **Search** the graph with A*, using the node coordinates to compute a straight-line (Euclidean) distance heuristic between the current node and the goal. This is the standard "admissible heuristic" approach for A* — it never overestimates the true remaining cost, since a straight line is never longer than any path the graph could offer, which is what makes A* still find the optimal path while exploring fewer nodes than plain Dijkstra.
3. **Report** by writing a self-contained summary — the full node table, the full connection table, and the resulting path/cost for every required pair — to `pathfinding_output.txt`.

I followed the lecture's version of A* closely, including the "reopen a closed node" rule: if a cheaper route to an already-closed node is found later, the node is pulled back into the open set instead of being left with a stale, non-optimal cost. That detail matters on a graph like this one, which has asymmetric edge costs (e.g. `C 1 -> 2 cost=5.0` but `C 2 -> 1 cost=30.0`), so the first path found to a node isn't always its cheapest.

## Structure and Methodologies

- **Graph representation**: an adjacency list, built with `collections.defaultdict(list)`, mapping each node id to a list of `(neighbor_id, cost)` tuples. This is a natural fit for a sparse graph like Adventure Bay (66 nodes, ~113 directed edges) — far less wasteful than an adjacency matrix would be here.
- **Open/closed sets**: plain Python `set()` objects for the A* open and closed lists. `min()` with a key function picks the lowest-`f-cost` node out of the open set each iteration (ties broken by node id for determinism). This is a simple approach rather than a heap-based priority queue — it's easy to read and correct, at the cost of O(n) node selection instead of O(log n); fine at this graph's scale.
- **Cost tracking**: `g_cost` and `f_cost` are `defaultdict(lambda: math.inf)` so any unseen node defaults to infinite cost without needing to be pre-initialized. `prev` is a plain dict used to reconstruct the final path by walking backward from the goal once it's reached, using a `collections.deque` for efficient left-appends.
- **Heuristic**: `math.hypot(dx, dz)` — straight-line distance between two nodes' (x, z) map coordinates.
- **Dependencies**: only the Python standard library — `csv` (to parse the node/connection files, which are essentially comma-separated records), `math`, and `collections` (`defaultdict`, `deque`). No third-party packages.

## Process

1. **Parse the nodes file.** `load_nodes()` reads `CS 330, Pathfinding, Graph AB Nodes v3.txt` with `csv.reader`, skips comment (`#`) and blank lines, and keeps only rows starting with `"N"`. From each row it pulls the node id, x/z coordinates, and an optional display name (some nodes are landmarks like "Katie's Pet Parlor" or "Foggy Bottom"; most are unnamed waypoints).
2. **Parse the connections file.** `load_connections()` reads `CS 330, Pathfinding, Graph AB Connections v3.txt` the same way, keeping rows starting with `"C"`, and builds the directed adjacency list plus a flat list of all edges (for printing later).
3. **Run A* on each required pair.** `main()` loops over the five `(start, goal)` pairs in `TESTS` and calls `astar()` for each one, which returns the node sequence and total cost (or an empty path / infinite cost if no route exists).
4. **Write the report.** Everything — the full sorted node table, the full sorted connection table, and each path result — is written to `pathfinding_output.txt` in one pass, so the output file is a complete, self-checkable record of both the graph that was loaded and the answers produced from it.

Comparing my output to `CS 330, Pathfinding, Trace Adventure Bay EXAMPLE.txt`: the example file is the instructor-provided scenario template — it shows the same graph (same node/connection format, including the intermediate search-state columns like status/cost-so-far/heuristic that a manual/trace run would fill in) and lists the same five required path queries, but with the paths and costs left as placeholders (`path= 1 ....... 29 cost= ?`). My program's job was effectively to fill in those blanks programmatically: `pathfinding_output.txt` contains the same five pairs with the actual computed paths and total costs, which is the concrete, checkable deliverable the example was modeling.

## Outcome

Running the program against the Adventure Bay graph produced the following results (from `pathfinding_output.txt`):

| Start → Goal | Path | Cost |
|---|---|---|
| 1 → 29 | 1 2 3 4 5 6 7 8 10 11 12 13 14 15 16 29 | 298.0 |
| 1 → 38 | 1 2 3 4 5 6 7 8 10 11 35 37 38 | 301.0 |
| 11 → 1 | 11 12 13 14 15 16 17 18 19 20 21 22 23 3 2 1 | 309.0 |
| 33 → 66 | 33 13 14 15 16 17 18 46 47 64 66 | 201.0 |
| 58 → 43 | 58 48 49 50 51 52 53 44 38 40 43 | 208.0 |

A few things stand out: the 1 → 29 and 11 → 1 routes both funnel through the long bridge chain (nodes 2–8) because that's the only way across that part of the map even though several of those edges are individually expensive; and the 1 → 2 vs. 2 → 1 cost asymmetry (5.0 vs. 30.0) confirms the graph is genuinely directed, which is exactly the kind of case the "reopen a closed node" rule in my A* implementation is there to handle correctly.

This project gave me hands-on practice with graph algorithms beyond the pseudocode level — building an adjacency list from raw structured text, implementing A* with a real admissible heuristic instead of a toy example, and validating the algorithm's correctness by checking that every found path is actually connected in the source data and that direction/cost asymmetries are respected. Getting the "reopen closed nodes" edge case right, and having the output file double as a self-contained proof of both the input graph and the results, are the parts I'm most satisfied with — it means anyone (including a grader) can check my answers against the raw data without re-running anything.

## How to run

```
python WardCS330Program3.py
```

Run it from this directory so it can find the two input files by their relative names. It prints a one-line confirmation and (re)writes `pathfinding_output.txt`.
