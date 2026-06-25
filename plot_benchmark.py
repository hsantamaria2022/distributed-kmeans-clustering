"""Generate speedup plot from benchmark data."""
import matplotlib.pyplot as plt

# Baseline: 1 process, 1 thread
baseline = 0.589

# OpenMP scaling (NP=1, varying threads)
omp_threads = [1, 2, 4]
omp_times = [0.589, 0.327, 0.386]

# MPI scaling (NT=1, varying processes)
mpi_procs = [1, 2, 4]
mpi_times = [0.589, 0.328, 0.281]

omp_speedup = [baseline / t for t in omp_times]
mpi_speedup = [baseline / t for t in mpi_times]

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(omp_threads, omp_speedup, 'o-', linewidth=2.2, markersize=9,
        color='#2196F3', label='OpenMP (1 process)')
ax.plot(mpi_procs, mpi_speedup, 's-', linewidth=2.2, markersize=9,
        color='#4CAF50', label='MPI (1 thread)')
ax.plot([1, 2, 4], [1, 2, 4], '--', color='gray', alpha=0.5,
        linewidth=1.5, label='Ideal (linear)')

ax.set_xlabel("Parallelism Level (threads or processes)", fontsize=12)
ax.set_ylabel("Speedup (×)", fontsize=12)
ax.set_title("K-Means Clustering — Parallel Speedup\n1M points, 2D, 4 clusters (single node)",
             fontsize=12, fontweight='bold')
ax.set_xticks([1, 2, 4])
ax.set_yticks([1, 2, 3, 4])
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, 4.5)
ax.set_ylim(0, 4.5)

for t, s in zip(omp_threads, omp_speedup):
    ax.annotate(f'{s:.2f}×', (t, s), textcoords="offset points",
                xytext=(10, -5), fontsize=10)
for t, s in zip(mpi_procs, mpi_speedup):
    ax.annotate(f'{s:.2f}×', (t, s), textcoords="offset points",
                xytext=(10, 5), fontsize=10)

plt.tight_layout()
plt.savefig("benchmark_speedup.png", dpi=150, bbox_inches='tight')
print("Saved benchmark_speedup.png")
