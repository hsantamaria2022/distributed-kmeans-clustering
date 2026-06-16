/**
 * Distributed K-Means Clustering using MPI + OpenMP
 * 
 * Each MPI process owns one cluster. Points migrate between processes
 * based on nearest-centroid assignment until convergence (<5% changes).
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
    std::vector<int> localGroups(localN);
    std::vector<float> globalCentroids(size * dimensions);

    for(int iter = 0; iter < maxIter; iter++){
        localN = localCount / dimensions;
        localGroups.resize(localN);

        // Compute local centroid (mean of all local points)
        std::vector<float> centroid(dimensions, 0.0f);

        if (localN > 0){
            #pragma omp parallel
            {
                std::vector<float> local_sum(dimensions, 0.0f);
                #pragma omp for schedule(static)
                for (int i = 0; i < localN; i++){
                    const float* row = &localPoints[i * dimensions];
                    for (uint32_t d = 0; d < dimensions; d++){
                        local_sum[d] += row[d];
                    }
                }
                #pragma omp critical
                {
                    for (uint32_t d = 0; d < dimensions; d++){
                        centroid[d] += local_sum[d];
                    }
                }
            }

            float inv = 1.0f / localN;
            for(uint32_t d = 0; d < dimensions; d++){
                centroid[d] *= inv;
            }
        }

        // Share centroids across all processes
        MPI_Allgather(centroid.data(), dimensions, MPI_FLOAT, 
                      globalCentroids.data(), dimensions, MPI_FLOAT, MPI_COMM_WORLD);

        if(rank == 0 && (iter == 0 || iter % 10 == 0)){
            std::cout << "\nCentroids iter: " << iter << "\n";
            for(int r = 0; r < size; r++){
                std::cout << "Group " << r << ": [ ";
                for(uint32_t d = 0; d < dimensions; d++){
                    std::cout << globalCentroids[r*dimensions + d] << " ";
                }
                std::cout << "]\n";
            }
        }

        // Assign each point to nearest centroid (parallel)
        int localChanges = 0;

        #pragma omp parallel for schedule(static) reduction(+:localChanges)
        for(int i = 0; i < localN; i++){
            float bestDist = 1e30f;
            int bestGroup = -1;
            const float* point = &localPoints[i * dimensions];

            for(int c = 0; c < size; c++){
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

            localGroups[i] = bestGroup;
            if(bestGroup != rank) localChanges++;
        }

        // Check convergence
        int globalChanges = 0;
        MPI_Allreduce(&localChanges, &globalChanges, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        float changeRatio = (float)globalChanges / (float)nPoints;
        if(rank == 0){
            std::cout << "Change percentage: " << changeRatio * 100.0 << "%. iter: " << iter << std::endl;
        }

        if (changeRatio < 0.05f) {
            if (rank == 0){
                std::cout << "Convergence reached at iteration " << iter
                        << " with " << changeRatio * 100.0 
                        << "% global changes.\n";
            }
            break;
        }

        // --- Phase 4: Migrate points to their assigned process ---
        std::vector<int> sendCounts(size, 0);

        for(int i = 0; i < localN; i++){
            sendCounts[localGroups[i]] += dimensions;
        }

        std::vector<int> sdispls(size);
        sdispls[0] = 0;
        for(int r = 1; r < size; r++){
            sdispls[r] = sdispls[r-1] + sendCounts[r-1];
        }

        int totalSend = sdispls[size-1] + sendCounts[size-1];
        std::vector<float> sendbuf(totalSend);
        std::vector<int> pos(size, 0);
        
        for(int i = 0; i < localN; i++){
            int dest = localGroups[i];
            int off = sdispls[dest] + pos[dest];
            const float* row = &localPoints[i * dimensions];

            for(uint32_t d = 0; d < dimensions; d++){
                sendbuf[off + d] = row[d];
            }
            pos[dest] += dimensions;
        }

        // Exchange points between all processes
        std::vector<int> recvcounts(size);
        MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        std::vector<int> rdispls(size);
        rdispls[0] = 0;
        for(int r = 1; r < size; r++){
            rdispls[r] = rdispls[r-1] + recvcounts[r-1];
        }

        int totalRecv = rdispls[size-1] + recvcounts[size-1];
        localPoints.resize(totalRecv);

        MPI_Alltoallv(sendbuf.data(), sendCounts.data(), sdispls.data(), MPI_FLOAT, 
                      localPoints.data(), recvcounts.data(), rdispls.data(), MPI_FLOAT, MPI_COMM_WORLD);

        localCount = totalRecv;
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
