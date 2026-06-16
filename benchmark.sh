#!/bin/bash
# Benchmark: measures execution time varying MPI processes and OMP threads
# Outputs results to benchmark_results.csv

DATASET_POINTS=50000
DATASET_CLUSTERS=4
OUTPUT="benchmark_results.csv"

echo "processes,threads,time_seconds" > "$OUTPUT"

./generate_data $DATASET_CLUSTERS $DATASET_POINTS

for NP in 1 2 4; do
    for NT in 1 2 4 8; do
        export OMP_NUM_THREADS=$NT
        TIME=$(mpirun --oversubscribe -np $NP ./kmeans 2>/dev/null | grep "Total execution time" | awk '{print $4}')
        echo "$NP,$NT,$TIME" >> "$OUTPUT"
        echo "NP=$NP, NT=$NT -> ${TIME}s"
    done
done

echo "Results saved to $OUTPUT"
