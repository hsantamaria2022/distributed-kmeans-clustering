# Distributed K-Means Clustering (MPI + OpenMP)

A high-performance implementation of the K-means clustering algorithm using **MPI** for distributed computing across multiple nodes and **OpenMP** for shared-memory parallelism within each node.

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Rank 0: Data Loading                │
│         (reads binary file, distributes)        │
└─────────────────┬───────────────────────────────┘
                  │ MPI_Scatterv
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
┌────────┐  ┌────────┐  ┌────────┐
│ Rank 0 │  │ Rank 1 │  │ Rank N │
│ OpenMP │  │ OpenMP │  │ OpenMP │
│ threads│  │ threads│  │ threads│
└───┬────┘  └───┬────┘  └───┬────┘
    │            │            │
    └────────────┼────────────┘
                 │ MPI_Alltoallv (point migration)
                 ▼
         Iterate until convergence
```

Each MPI process owns one cluster. On every iteration:

1. Each process computes its local centroid (OpenMP-parallel)
2. Centroids are shared via `MPI_Allgather`
3. Each point is reassigned to the nearest centroid
4. Points migrate to their new owner process via `MPI_Alltoallv`
5. Global statistics (min, max, mean, variance) are computed via MPI reductions
6. Convergence is reached when <5% of points change cluster

## Key Features

- **Hybrid parallelism**: MPI (inter-node) + OpenMP (intra-node)
- **Dynamic load balancing**: points migrate each iteration
- **N-dimensional**: works with arbitrary-dimensional data
- **Binary I/O**: efficient data format for large datasets

## Prerequisites

- C++ compiler with C++11 support
- MPI implementation (OpenMPI or MPICH)
- OpenMP support (included with GCC)

```bash
# Ubuntu/Debian
sudo apt install mpich libmpich-dev

# macOS
brew install open-mpi
```

## Build & Run

```bash
# Build
make

# Generate sample data (4 clusters, 500 points each)
./generate_data 4 500

# Run with 4 MPI processes
mpirun -np 4 ./kmeans

# Optional: control threads per process
export OMP_NUM_THREADS=4
mpirun -np 4 ./kmeans
```

## Example Output

```
Number of rows: 2000
Number of dimensions: 2
Rank 0 received 500 points
Rank 1 received 500 points
Rank 2 received 500 points
Rank 3 received 500 points

Centroids iter: 0
Group 0: [ 3.21 -7.84 ]
Group 1: [ -12.05 14.32 ]
Group 2: [ 8.67 9.11 ]
Group 3: [ -5.43 -11.20 ]

Change percentage: 62.5%. iter: 0
...
Convergence reached at iteration 8 with 3.2% global changes.

--- Global Statistics ---
Dim 0: Min=-14.2 Max=10.8 Mean=-1.4 Variance=72.3
Dim 1: Min=-13.5 Max=16.1 Mean=1.1 Variance=98.7

Total execution time: 0.041 s
```

## Data Format

The binary input file (`dataset.bin`) structure:

| Field       | Type     | Description           |
|-------------|----------|-----------------------|
| nPoints     | uint32_t | Number of data points |
| nDimensions | uint32_t | Dimensions per point  |
| data        | float[]  | Row-major point data  |

## Technologies

- **C++11** — Core language
- **MPI** — Distributed memory parallelism (Scatterv, Allgather, Alltoallv, Reduce)
- **OpenMP** — Shared memory parallelism (parallel for, reductions, critical sections)

## License

MIT
