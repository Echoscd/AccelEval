# Hotspot 2D Thermal Simulation

## Problem Background

This task simulates transient heat diffusion on a 2D chip surface with a power dissipation field.
At each time step, every grid cell exchanges heat with its neighbors and the ambient environment,
while also receiving local heat input from the power matrix.

The benchmark uses the classic Rodinia Hotspot finite-difference formulation for thermal simulation.
It is a canonical structured-grid stencil workload.

## Algorithm Source

This task is adapted from the real GitHub benchmark source:
- Repository: `yuhc/gpu-rodinia`
- Benchmark: `openmp/hotspot/hotspot_openmp.cpp`
- Suite: Rodinia Benchmark Suite (University of Virginia)

The ORBench version keeps the same physical constants, the same transient update equations,
and the same edge/corner boundary handling, while converting the benchmark to ORBench input/output format.

## Why It Fits GPU Acceleration

- **Embarrassingly parallel within each iteration**: each cell update is independent once the previous temperature field is fixed.
- **Regular stencil access**: most cells read a small local neighborhood.
- **Tiling-friendly**: shared-memory tiles with halo cells naturally reduce global memory traffic.
- **Double buffering**: GPU implementations can swap two device grids each iteration.

The main bottlenecks are repeated global-memory traffic and boundary-condition handling.

## Input Format

Binary file `input.bin` (ORBench v2 format):

| Tensor | Type | Shape | Description |
|---|---|---:|---|
| `temp0` | float32 | `[rows * cols]` | Initial temperature field, row-major |
| `power` | float32 | `[rows * cols]` | Power dissipation field, row-major |

| Parameter | Type | Description |
|---|---|---|
| `rows` | int64 | Number of grid rows |
| `cols` | int64 | Number of grid columns |
| `iters` | int64 | Number of transient iterations |
| `seed` | int64 | RNG seed used by the generator |

## Output Format

`expected_output.txt` contains `rows * cols` lines.
Each line is the final temperature of one grid cell in row-major order:

```text
Format: "%.6e\n" per line
```

## Data Sizes

| Size | rows | cols | iterations |
|---|---:|---:|---:|
| small | 256 | 256 | 50 |
| medium | 768 | 768 | 100 |
| large | 1536 | 1536 | 150 |

## Notes for ORBench Integration

- Recommended interface mode: `init_compute`
- Correctness check: floating-point numerical comparison
- Suggested tolerance: `atol = 1e-3`, `rtol = 1e-4`
