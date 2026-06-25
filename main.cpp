/**
 * Distributed K-Means Clustering using MPI + OpenMP
 * 
 * Points are statically distributed across processes. Each process
 * computes partial sums for all K centroids, combined via MPI_Allreduce.
 * K is configurable and independent of the number of processes.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <mpi.h>
#include <omp.h>

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);

    double t0 = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(size < 1){
        if(rank == 0)
            std::cerr << "Error: at least 1 MPI process required\n";
        MPI_Finalize();
        return 1;
    }

    int K = 4; // default number of clusters
    if(argc > 1) K = atoi(argv[1]);

    uint32_t nPoints, dimensions; 
    std::vector<float> globalPoints;   
    
    // --- Phase 1: Rank 0 reads binary dataset ---
    if(rank==0){
        std::ifstream file("dataset.bin", std::ios::binary);

        if(!file){
            std::cout << "Error reading binary file" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        file.read((char*)&nPoints, sizeof(uint32_t));
        file.read((char*)&dimensions, sizeof(uint32_t));

        globalPoints.resize(nPoints * dimensions);

        std::cout << "Number of rows: " << nPoints << std::endl;
        std::cout << "Number of dimensions: " << dimensions << std::endl;

        file.read((char*)globalPoints.data(), nPoints * dimensions * sizeof(float));
    }
    
    // --- Phase 2: Distribute data across processes ---
    MPI_Bcast(&nPoints, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dimensions, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    int base = nPoints / size;
    int remainder = nPoints % size;

    std::vector<int> sendcounts(size);
    std::vector<int> offset(size);

    for(int r=0; r<size; r++){
        sendcounts[r] = (r < remainder ? base + 1 : base) * dimensions;
    }
    
    offset[0] = 0;
    for(int r=1; r<size; r++){
        offset[r] = offset[r-1] + sendcounts[r-1];
    }

    int localCount = sendcounts[rank];
    std::vector<float> localPoints(localCount);

    MPI_Scatterv(globalPoints.data(), sendcounts.data(), offset.data(), 
                MPI_FLOAT, localPoints.data(), localCount, MPI_FLOAT, 0, 
                MPI_COMM_WORLD);
    
    // Free global data on rank 0 after scatter
    globalPoints.clear();
    globalPoints.shrink_to_fit();

    int localN = localCount / dimensions;
    std::cout << "Rank " << rank << " received " << localN << " points" << std::endl;

    // --- Phase 3: Iterative K-means loop ---
    int maxIter = 2000;
    std::vector<int> localGroups(localN, 0);
    std::vector<float> globalCentroids(K * dimensions, 0.0f);

    // Initialize centroids: Rank 0 picks K evenly spaced points
    {
        std::vector<float> seeds(K * dimensions);
        if(rank == 0){
            std::ifstream file("dataset.bin", std::ios::binary);
            file.seekg(2 * sizeof(uint32_t));
            int step = nPoints / K;
            for(int c = 0; c < K; c++){
                file.seekg(2 * sizeof(uint32_t) + (long)(c * step) * dimensions * sizeof(float));
                file.read((char*)&seeds[c * dimensions], dimensions * sizeof(float));
            }
        }
        MPI_Bcast(seeds.data(), K * dimensions, MPI_FLOAT, 0, MPI_COMM_WORLD);
        globalCentroids = seeds;
    }

    for(int iter = 0; iter < maxIter; iter++){

        // Assign each local point to nearest centroid
        int localChanges = 0;

        #pragma omp parallel for schedule(static) reduction(+:localChanges)
        for(int i = 0; i < localN; i++){
            float bestDist = 1e30f;
            int bestGroup = -1;
            const float* point = &localPoints[i * dimensions];

            for(int c = 0; c < K; c++){
                float dist = 0.0f;
                const float* cent = &globalCentroids[c * dimensions];
                for(uint32_t d = 0; d < dimensions; d++){
                    float diff = point[d] - cent[d];
                    dist += diff * diff;
                }
                if(dist < bestDist){
                    bestDist = dist;
                    bestGroup = c;
                }
            }

            if(localGroups[i] != bestGroup) localChanges++;
            localGroups[i] = bestGroup;
        }

        // Check convergence globally
        int globalChanges = 0;
        MPI_Allreduce(&localChanges, &globalChanges, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        float changeRatio = (float)globalChanges / (float)nPoints;
        if(rank == 0){
            std::cout << "Change percentage: " << changeRatio * 100.0 << "%. iter: " << iter << std::endl;
        }

        if (iter > 0 && changeRatio < 0.001f) {
            if (rank == 0){
                std::cout << "Convergence reached at iteration " << iter
                        << " with " << changeRatio * 100.0
                        << "% global changes.\n";
            }
            break;
        }

        // Compute partial sums for each centroid locally
        std::vector<double> localSums(K * dimensions, 0.0);
        std::vector<int> localCounts(K, 0);

        #pragma omp parallel
        {
            std::vector<double> tSums(K * dimensions, 0.0);
            std::vector<int> tCounts(K, 0);

            #pragma omp for schedule(static)
            for(int i = 0; i < localN; i++){
                int g = localGroups[i];
                tCounts[g]++;
                const float* row = &localPoints[i * dimensions];
                for(uint32_t d = 0; d < dimensions; d++)
                    tSums[g * dimensions + d] += row[d];
            }

            #pragma omp critical
            {
                for(int c = 0; c < K; c++){
                    localCounts[c] += tCounts[c];
                    for(uint32_t d = 0; d < dimensions; d++)
                        localSums[c * dimensions + d] += tSums[c * dimensions + d];
                }
            }
        }

        // Reduce across all processes
        std::vector<double> globalSums(K * dimensions);
        std::vector<int> globalCounts(K);
        MPI_Allreduce(localSums.data(), globalSums.data(), K * dimensions, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(localCounts.data(), globalCounts.data(), K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // Update centroids
        for(int c = 0; c < K; c++){
            if(globalCounts[c] > 0){
                for(uint32_t d = 0; d < dimensions; d++)
                    globalCentroids[c * dimensions + d] = (float)(globalSums[c * dimensions + d] / globalCounts[c]);
            }
            // If empty cluster, centroid stays unchanged
        }

        if(rank == 0 && (iter == 0 || iter % 10 == 0)){
            std::cout << "\nCentroids iter: " << iter << "\n";
            for(int c = 0; c < K; c++){
                std::cout << "Group " << c << ": [ ";
                for(uint32_t d = 0; d < dimensions; d++)
                    std::cout << globalCentroids[c * dimensions + d] << " ";
                std::cout << "]\n";
            }
        }
    }

    // --- Phase 5: Final statistics (only once, after convergence) ---
    localN = localCount / dimensions;
    int nt = omp_get_max_threads();

    std::vector<std::vector<float>>  tMin(nt, std::vector<float>(dimensions,  1e9f));
    std::vector<std::vector<float>>  tMax(nt, std::vector<float>(dimensions, -1e9f));
    std::vector<std::vector<double>> tSum(nt, std::vector<double>(dimensions, 0.0));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float*  myMin = tMin[tid].data();
        float*  myMax = tMax[tid].data();
        double* mySum = tSum[tid].data();

        #pragma omp for schedule(static)
        for(int i = 0; i < localN; i++){
            const float* row = &localPoints[i * dimensions];
            for(int d = 0; d < (int)dimensions; d++){
                float v = row[d];
                if(v < myMin[d]) myMin[d] = v;
                if(v > myMax[d]) myMax[d] = v;
                mySum[d] += v;
            }
        }
    }

    std::vector<float>  minLocal(dimensions,  1e9f);
    std::vector<float>  maxLocal(dimensions, -1e9f);
    std::vector<double> sumLocal(dimensions, 0.0);
    for(int t = 0; t < nt; t++){
        for(int d = 0; d < (int)dimensions; d++){
            if(tMin[t][d] < minLocal[d]) minLocal[d] = tMin[t][d];
            if(tMax[t][d] > maxLocal[d]) maxLocal[d] = tMax[t][d];
            sumLocal[d] += tSum[t][d];
        }
    }

    std::vector<float> minGlobal(dimensions), maxGlobal(dimensions);
    std::vector<double> sumGlobal(dimensions);
    MPI_Reduce(minLocal.data(), minGlobal.data(), dimensions, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(maxLocal.data(), maxGlobal.data(), dimensions, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(sumLocal.data(), sumGlobal.data(), dimensions, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    std::vector<double> mean(dimensions);
    if(rank == 0){
        for(int d = 0; d < (int)dimensions; d++)
            mean[d] = sumGlobal[d] / nPoints;
    }
    MPI_Bcast(mean.data(), dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Variance pass
    std::vector<double> varLocal(dimensions, 0.0);
    #pragma omp parallel
    {
        std::vector<double> myVar(dimensions, 0.0);
        #pragma omp for schedule(static)
        for(int i = 0; i < localN; i++){
            const float* row = &localPoints[i * dimensions];
            for(int d = 0; d < (int)dimensions; d++){
                double diff = row[d] - mean[d];
                myVar[d] += diff * diff;
            }
        }
        #pragma omp critical
        {
            for(int d = 0; d < (int)dimensions; d++)
                varLocal[d] += myVar[d];
        }
    }

    std::vector<double> varGlobal(dimensions);
    MPI_Reduce(varLocal.data(), varGlobal.data(), dimensions, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double t1 = MPI_Wtime();

    if(rank == 0){
        std::cout << "\n--- Global Statistics ---" << std::endl;
        for(int d = 0; d < (int)dimensions; d++){
            std::cout << "Dim " << d << ":"
                      << " Min=" << minGlobal[d]
                      << " Max=" << maxGlobal[d]
                      << " Mean=" << mean[d]
                      << " Variance=" << (varGlobal[d] / nPoints) << "\n";
        }
        std::cout << "\nTotal execution time: " << (t1-t0) << " s" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
