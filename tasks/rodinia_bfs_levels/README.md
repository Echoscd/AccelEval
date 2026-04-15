# Rodinia BFS Level-Synchronous Traversal

## Problem Background

This task computes single-source shortest path distances on an unweighted directed graph using
level-synchronous breadth-first search (BFS). Starting from a source vertex, the algorithm expands
one frontier at a time and assigns each reachable vertex its minimum hop count.

This benchmark is adapted from the Rodinia BFS benchmark family. Rodinia stores each node as a
pair `(starting, no_of_edges)` plus a flat edge list, then repeatedly updates three boolean masks:
current frontier, next frontier, and visited.

## Algorithm Source

This ORBench task is based on the real Rodinia BFS code structure:
- Rodinia benchmark suite repository (`yuhc/gpu-rodinia` / mirror `THU-DSP-LAB/gpu-rodinia`)
- BFS benchmark listed under OpenMP benchmarks in the Rodinia suite
- The BFS kernel structure is also visible in the historical Rodinia/OpenACC BFS file discussed on
  NVIDIA Developer Forums, which shows the same `Node { starting, no_of_edges }` representation and
  frontier-mask update pattern.

The ORBench version keeps the same graph representation idea and the same level-synchronous mask-based
BFS logic, while converting input/output into ORBench binary format.

## Why It Fits GPU Acceleration

- **Massive frontier parallelism**: every active frontier vertex can expand its outgoing edges concurrently.
- **Irregular but parallel graph traversal**: each level touches many edges in parallel.
- **Persistent graph data**: node metadata and edge arrays can stay on device across iterations.
- **Frontier compaction / bitmap opportunities**: masks and visited arrays map naturally to parallel primitives.

The main bottlenecks are irregular memory access to the edge list and load imbalance across frontier sizes.

## Input Format

Binary file `input.bin` (ORBench v2 format):

| Tensor | Type | Shape | Description |
|---|---|---:|---|
| `node_start` | int32 | `[num_nodes]` | CSR row start for each node |
| `node_degree` | int32 | `[num_nodes]` | Out-degree for each node |
| `edge_dst` | int32 | `[num_edges]` | Flat adjacency list destinations |

| Parameter | Type | Description |
|---|---|---|
| `num_nodes` | int64 | Number of graph nodes |
| `num_edges` | int64 | Number of directed edges |
| `source` | int64 | BFS source vertex |
| `seed` | int64 | RNG seed used by the generator |

## Output Format

`expected_output.txt` contains `num_nodes` lines.
Each line is the BFS distance from `source` to one node:

```text
Format: "%d\n" per line
```

Unreachable nodes are reported as `-1`.

## Data Sizes

| Size | num_nodes | approx avg out-degree |
|---|---:|---:|
| small | 20,000 | 6 |
| medium | 100,000 | 8 |
| large | 250,000 | 8 |

## Notes for ORBench Integration

- Recommended interface mode: `init_compute`
- Correctness check: exact integer comparison
- Graph is stored in Rodinia-style node metadata + edge list form
