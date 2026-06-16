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
