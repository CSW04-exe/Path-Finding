# Path-Finding — A* Search Over Adventure Bay

A Python implementation of the A* search algorithm that finds shortest paths across a hand-built map graph ("Adventure Bay"), written for a CS 330 (Analysis of Algorithms) assignment.

## 1. Purpose

This project exists to implement and demonstrate a working, from-scratch A* pathfinding algorithm — one of the core graph-search algorithms covered in an algorithms course — on a nontrivial, realistically laid-out map rather than a toy grid. The goal was to take a graph representation of a fictional town (Adventure Bay, a map with 66 named and unnamed locations such as docks, houses, a train station, and a ski resort) and compute the actual shortest routes between specific location pairs, the same kind of problem that powers GPS navigation, game NPC movement, and network routing.

Beyond satisfying the assignment brief, the project was a chance to see A* end-to-end: not just the search loop itself, but the surrounding work of parsing a real (if simplified) data format, building a usable in-memory graph from it, and producing output that could be checked by hand against the map.

## 2. Problem and approach

The assignment (CS 330, Program 3) was assigned with a fixed dataset and a fixed set of test cases, rather than being self-chosen:

- **Input data**: two flat text files describing the graph — `CS 330, Pathfinding, Graph AB Nodes v3.txt` (66 node records with id, x/z coordinates, and an optional location name) and `CS 330, Pathfinding, Graph AB Connections v3.txt` (directed edges with an explicit travel cost, which is **not** simply the Euclidean distance between the two endpoints — the graph has asymmetric costs, e.g. `C 1 -> 2 cost=5.0` but `C 2 -> 1 cost=30.0`).
- **Required output**: shortest paths (sequence of node IDs and total cost) for five specific start/goal pairs: `(1, 29)`, `(1, 38)`, `(11, 1)`, `(33, 66)`, `(58, 43)`.

The approach was to:
1. Parse both files into an adjacency-list graph, ignoring comment lines (`#`) and any record whose tag isn't `N` (node) or `C` (connection).
2. Implement A* by hand, using straight-line (Euclidean) distance between a node's `(x, z)` coordinates and the goal's as the heuristic — an admissible estimate since it's a lower bound on any real path cost.
3. Run the five required searches and write a single consolidated report (`pathfinding_output.txt`) containing the full node table, the full edge table, and the resulting paths with their total costs, so the results could be manually spot-checked against the map layout described in the accompanying trace document.

## 3. Structure and methodologies

**Language & dependencies**: Pure Python 3, standard library only (`csv`, `math`, `collections.defaultdict`, `collections.deque`). No external packages, frameworks, or visualization libraries are used — the entire project runs with a single `python WardCS330Program3.py` invocation and no `pip install` step.

**Data structures**:
- **Graph representation**: an adjacency list built with `defaultdict(list)`, mapping each node id to a list of `(neighbor_id, edge_cost)` tuples — directed and weighted, since the source data encodes asymmetric costs per direction.
- **Node table**: a plain `dict` keyed by node id, storing `x`, `z` coordinates and a display name, used both for heuristic distance calculations and for pretty-printing.
- **Open/closed sets**: implemented as plain Python `set`s rather than a binary heap. The "lowest f-cost node in Open" step is done with `min(open_set, key=lambda n: (f_cost[n], n))`, a linear scan rather than a priority queue — a reasonable tradeoff at this graph's scale (66 nodes) but the clearest place a production implementation would swap in `heapq`.
- **Cost tables**: `g_cost` and `f_cost` are `defaultdict(lambda: math.inf)`, so any never-visited node behaves correctly as "infinitely far" without pre-populating every id.
- **Path reconstruction**: a `prev` dictionary (child → parent) walked backwards from the goal and assembled with a `deque` (`appendleft`) for O(1) prepends while rebuilding the path in order.

**Algorithm**: A* search with an explicit "reopen a closed node if a cheaper route is found" step — an addition beyond the textbook-minimal version, needed because this graph's costs aren't symmetric or purely distance-based, so a node marked "closed" isn't necessarily final until every incoming improvement has been checked.

**Files in the repo**:
- `WardCS330Program3.py` — the graph loader, A* implementation, and report writer.
- `CS 330, Pathfinding, Graph AB Nodes v3.txt` / `...Connections v3.txt` — the provided input dataset (66 nodes, 100+ directed edges), authored by the course instructor.
- `CS 330, Pathfinding, Trace Adventure Bay EXAMPLE.txt` — a worked example trace used to verify the algorithm's step-by-step behavior against expected instructor output.
- `pathfinding_output.txt` — the generated report: full node dump, full edge dump, and the five required shortest paths with total cost.

## 4. Process

The commit and file history tells a fairly linear, assignment-driven story:

1. **Start from the given data.** The two graph text files and the worked trace example were the starting point — the map, node names, and edge costs were fixed by the instructor, so the first task was understanding the record format (fields for status, cost-so-far, heuristic, total, previous-node, coordinates, and plot hints — most of which are scratch fields meant to be filled in by a *plotting* tool and aren't needed by the search itself).
2. **Build the parser first.** `load_nodes` and `load_connections` were written to defensively skip comment lines and any row not tagged `N` or `C`, since the raw files include a header comment block explaining every field.
3. **Implement A* against the trace example.** The heuristic (`h`) uses `math.hypot` on the `(x, z)` fields — an intentional choice since those coordinates already represent physical placement on the map, making Euclidean distance a valid, admissible lower bound. The core loop follows the standard open/closed-set formulation, checked step-by-step against the "EXAMPLE" trace file to confirm the algorithm reopened and re-costed nodes the same way the instructor's worked example did.
4. **Wire up the five required test pairs.** `TESTS = [(1, 29), (1, 38), (11, 1), (33, 66), (58, 43)]` was hard-coded per the assignment spec, and `main()` was written to loop over them and call `astar` for each.
5. **Build a single readable report instead of just printing to stdout.** Rather than only printing pass/fail per test, the script writes out the *entire* graph (every node, every edge) alongside the five paths into `pathfinding_output.txt` — turning the deliverable into something that could be handed in and manually audited line-by-line against the map.
6. **Sanity-check by hand.** With named landmarks in the data (docks, "Porter's Cafe", "Farmer Umi's farm", "Jake's snowboarding resort"), it was possible to visually reason about whether a returned route "made sense" geographically, not just trust the cost numbers.

## 5. Outcome

Running the program against the provided graph produces five verified shortest paths, e.g.:

| From → To | Path (node IDs) | Total cost |
|---|---|---|
| 1 → 29 | 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 29 | 298.0 |
| 1 → 38 | 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 35, 37, 38 | 301.0 |
| 11 → 1 | 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 3, 2, 1 | 309.0 |
| 33 → 66 | 33, 13, 14, 15, 16, 17, 18, 46, 47, 64, 66 | 201.0 |
| 58 → 43 | 58, 48, 49, 50, 51, 52, 53, 44, 38, 40, 43 | 208.0 |

Notably, `11 → 1` and `1 → 29`/`1 → 38` don't reuse the same edges in reverse — a direct byproduct of the graph's asymmetric edge costs, and a good concrete check that the implementation is genuinely respecting direction and cost rather than just finding *a* connecting path.

From building this, the main things demonstrated and learned were:

- **Translating a textbook algorithm into working code against messy, real-format input.** The graph wasn't handed over as clean adjacency lists — it came as a CSV-like format with comment headers, scratch fields, and quoted names containing literal `\n` escapes, so a meaningful chunk of the work was robust parsing before the algorithm ever ran.
- **Why the closed-set "reopen" step matters.** Because edge costs are directional and not simply proportional to distance, a naive A* that never revisits a closed node can settle for a suboptimal path. Handling that correctly — and verifying it against the instructor's worked trace — reinforced the difference between the "clean graph" version of A* taught in lecture and what's needed once assumptions like symmetric costs no longer hold.
- **The cost/complexity tradeoff in data structure choice.** Using a `set` with a linear `min()` scan for the open set instead of a heap was a conscious, appropriate simplification at 66 nodes, but the code makes clear where a `heapq`-based priority queue would be the next improvement for a much larger graph.
- **Value of writing a full, human-auditable report** rather than trusting the algorithm blindly: dumping every node and edge alongside the computed paths made it possible to manually trace a path on the map and confirm the algorithm's output was actually correct, not just plausible.
