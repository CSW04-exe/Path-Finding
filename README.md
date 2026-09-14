# Path-Finding — A* Search

**Type:** Individual project
**Contributor:** Carter Ward
**Course:** CS 330-1 (Artificial Intelligence / Game AI) — Program 3
**Completed:** 11/07/2025

## Purpose

This is Program 3 for CS 330 (Artificial Intelligence): implementing the A* search algorithm to find optimal routes through "Adventure Bay," a fictional map of 66 nodes connected by weighted, directed edges. The goal was to prove A* out as working code — reading a graph from data files, searching it, and reporting exact paths and costs — rather than leaving it as pseudocode.

## Problem and Approach

Given the Adventure Bay graph and five required (start, goal) pairs — (1, 29), (1, 38), (11, 1), (33, 66), (58, 43) — find the lowest-cost path for each and report the full path and total cost. I used a Euclidean-distance heuristic (straight-line distance between node coordinates), which is admissible since it never overestimates true remaining cost. I also followed the "reopen a closed node" rule: if a cheaper route to an already-closed node turns up later, it's pulled back into the open set rather than kept at a stale cost — important here since edge costs are asymmetric (e.g. 1→2 costs 5.0, but 2→1 costs 30.0).

## Structure and Methodologies

- Adjacency list built with `collections.defaultdict(list)`, mapping each node to `(neighbor, cost)` tuples
- Plain `set()` objects for the A* open/closed lists, with `min()` selecting the lowest f-cost node each iteration
- `g_cost`/`f_cost` as `defaultdict(lambda: math.inf)` so unseen nodes default to infinite cost
- Standard library only: `csv` (parsing node/connection files), `math`, `collections`

## Process

1. Parse the nodes file into a dictionary of ids, coordinates, and optional names
2. Parse the connections file into the directed adjacency list
3. Run A* on each of the five required start/goal pairs
4. Write a report with the node table, connection table, and each path/cost result

## Outcome

The program produced the following results:

| Start → Goal | Path | Cost |
|---|---|---|
| 1 → 29 | 1 2 3 4 5 6 7 8 10 11 12 13 14 15 16 29 | 298.0 |
| 1 → 38 | 1 2 3 4 5 6 7 8 10 11 35 37 38 | 301.0 |
| 11 → 1 | 11 12 13 14 15 16 17 18 19 20 21 22 23 3 2 1 | 309.0 |
| 33 → 66 | 33 13 14 15 16 17 18 46 47 64 66 | 201.0 |
| 58 → 43 | 58 48 49 50 51 52 53 44 38 40 43 | 208.0 |

This gave me hands-on practice with graph algorithms beyond pseudocode: building an adjacency list from raw text, implementing A* with a real admissible heuristic, and correctly handling directed, asymmetric-cost edges via the reopen-closed-node rule.

## How to run

```
python WardCS330Program3.py
```

Run from this directory so it can find the input files by their relative names.
