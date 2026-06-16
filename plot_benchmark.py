"""Generate speedup plot from real benchmark data."""
import matplotlib.pyplot as plt

# Real measurements: 500K points, 2D, 4 clusters (single node)
threads = [1, 2, 4]
times = [0.351211, 0.279453, 0.156293]  # 1 MPI process
baseline = times[0]
speedups = [baseline / t for t in times]

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(threads, speedups, 'o-', linewidth=2.2, markersize=9,
        color='#2196F3', label='Measured (1 node, OpenMP)')
ax.plot(threads, threads, '--', color='gray', alpha=0.5,
        linewidth=1.5, label='Ideal (linear)')

ax.set_xlabel("OpenMP Threads", fontsize=12)
ax.set_ylabel("Speedup (×)", fontsize=12)
ax.set_title("K-Means Clustering — OpenMP Speedup\n500K points, 2D, 4 clusters",
             fontsize=12, fontweight='bold')
ax.set_xticks(threads)
ax.set_yticks([1, 2, 3, 4])
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, 4.5)
ax.set_ylim(0, 4.5)

# Annotate speedup values
for t, s in zip(threads, speedups):
    ax.annotate(f'{s:.2f}×', (t, s), textcoords="offset points",
                xytext=(10, -5), fontsize=10)

plt.tight_layout()
plt.savefig("benchmark_speedup.png", dpi=150, bbox_inches='tight')
print("Saved benchmark_speedup.png")
