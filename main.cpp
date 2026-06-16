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

        for(uint32_t i=0; i<nPoints*dimensions; i++){
            float temp;
            file.read((char*)&temp, sizeof(float));
            globalPoints[i] = temp;
        }
    }
    
    // --- Phase 2: Distribute data across processes ---
    MPI_Bcast(&nPoints, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dimensions, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Compute per-process chunk sizes for uneven distribution
    int base = nPoints / size;
    int remainder = nPoints % size;

    std::vector<int> sendcounts(size);
    std::vector<int> offset(size);

    for(int r=0; r<size; r++){
        int pointsForR = (r < remainder ? base + 1 : base);
        sendcounts[r] = pointsForR * dimensions;
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
    
    int localN = localCount / dimensions;
    std::vector<int> localGroups;

    std::cout << "Rank " << rank << " received " << localN << " points" << std::endl;

    // --- Phase 3: Iterative K-means loop ---
    int maxIter = 2000;

    for(int iter = 0; iter < maxIter; iter++){
        localN = localCount / dimensions;
        localGroups.assign(localN, -1);

        // Compute local centroid (mean of all local points)
        std::vector<float> centroid(dimensions, 0.0f);

        if (localN > 0){
            #pragma omp parallel for reduction(+:centroid[:dimensions]) schedule(static)
            for (int i=0; i<localN; i++){
                for (uint32_t d=0; d<dimensions; d++){
                    centroid[d] += localPoints[i * dimensions + d];
                }
            }

            for(uint32_t d=0; d<dimensions; d++){
                centroid[d] /= localN;
            }
        }

        // Share centroids across all processes
        std::vector<float> globalCentroids(size*dimensions);
        MPI_Allgather(centroid.data(), dimensions, MPI_FLOAT, 
                      globalCentroids.data(), dimensions, MPI_FLOAT, MPI_COMM_WORLD);

        if(rank==0){
            std::cout << "\nCentroids iter: " << iter << "\n";
            for(int r=0; r<size; r++){
                std::cout << "Group " << r << ": [ ";
                for(uint32_t d=0; d<dimensions; d++){
                    std::cout << globalCentroids[r*dimensions + d] << " ";
                }
                std::cout << "]\n";
            }
        }

        // Assign each point to nearest centroid (parallel)
        #pragma omp parallel for schedule(static)
        for(int i=0; i<localN; i++){
            float bestDist = 1e30f;
            int bestGroup = -1;

            for(int c = 0; c<size; c++){
                float dist = 0.0f;

                for(uint32_t d=0; d<dimensions; d++){
                    float diff = localPoints[i*dimensions+d] - globalCentroids[c*dimensions+d];
                    dist += diff * diff;
                }

                if(dist < bestDist){
                    bestDist = dist;
                    bestGroup = c;
                }
            }

            localGroups[i] = bestGroup;
        }

        // Count points assigned to a different cluster than current rank
        int localChanges = 0;

        #pragma omp parallel for reduction(+:localChanges)
        for(int i=0; i < localN; i++){
            if (localGroups[i] != rank){
                localChanges++;
            }
        }

        int globalChanges = 0;
        MPI_Allreduce(&localChanges, &globalChanges, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        float changeRatio = (float)globalChanges / (float)nPoints;
        if(rank==0){
            std::cout << "\nChange percentage: " << changeRatio * 100.0 << "%. iter: " << iter << std::endl;
        }

        // Convergence: stop when less than 5% of points change cluster
        if (changeRatio < 0.05f) {
            if (rank == 0){
                std::cout << "Convergence reached at iteration " << iter
                        << " with " << changeRatio * 100.0 
                        << "% global changes.\n";
            }
            break;
        }

        // --- Phase 4: Migrate points to their assigned process ---

        // Count how many floats to send to each process
        std::vector<int> sendCounts(size, 0);

        #pragma omp parallel
        {
            std::vector<int> threadCounts(size, 0);

            #pragma omp for nowait
            for(int i=0; i<localN; i++){
                threadCounts[localGroups[i]] += dimensions;
            }

            #pragma omp critical 
            {
                for(int r=0; r<size; r++){
                    sendCounts[r] += threadCounts[r];
                }
            }
        }

        // Build send buffer with points grouped by destination
        std::vector<int> sdispls(size);
        sdispls[0]=0;
        for(int r=1; r<size; r++){
            sdispls[r] = sdispls[r-1] + sendCounts[r-1];
        }

        int totalFloatsToSend = sdispls[size-1] + sendCounts[size-1];
        std::vector<float> sendbuf(totalFloatsToSend);
        std::vector<int> pos(size, 0);
        
        for(int i=0; i<localN; i++){
            int dest = localGroups[i];
            int off = sdispls[dest] + pos[dest];

            for(uint32_t d=0; d<dimensions; d++){
                sendbuf[off+d] = localPoints[i * dimensions + d];
            }

            pos[dest] += dimensions;
        }

        // Exchange points between all processes
        std::vector<int> recvcounts(size);
        MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        std::vector<int> rdispls(size);
        rdispls[0]=0;
        for(int r=1; r<size; r++){
            rdispls[r] = rdispls[r-1] + recvcounts[r-1];
        }

        int totalRecv = rdispls[size-1] + recvcounts[size-1];
        std::vector<float> recvbuf(totalRecv);

        MPI_Alltoallv(sendbuf.data(), sendCounts.data(), sdispls.data(), MPI_FLOAT, 
                      recvbuf.data(), recvcounts.data(), rdispls.data(), MPI_FLOAT, MPI_COMM_WORLD);

        // Update local data with received points
        localPoints = recvbuf;
        localCount = totalRecv;
        localN = localCount / dimensions;

        // --- Phase 5: Compute global statistics (min, max, mean, variance) ---
        int nt = omp_get_max_threads();

        std::vector<std::vector<float>>  tMin(nt, std::vector<float>(dimensions,  1e9f));
        std::vector<std::vector<float>>  tMax(nt, std::vector<float>(dimensions, -1e9f));
        std::vector<std::vector<double>> tSum(nt, std::vector<double>(dimensions, 0.0));

        // Pass 1: parallel min, max, sum (thread-local buffers, no sync needed)
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            float*  myMin = tMin[tid].data();
            float*  myMax = tMax[tid].data();
            double* mySum = tSum[tid].data();

            #pragma omp for schedule(static)
            for(int i=0; i<localN; i++){
                const float* row = &localPoints[i * dimensions];
                for(int d=0; d<(int)dimensions; d++){
                    float v = row[d];
                    if(v < myMin[d]) myMin[d] = v;
                    if(v > myMax[d]) myMax[d] = v;
                    mySum[d] += v;
                }
            }
        }

        // Reduce thread-local results
        std::vector<float>  minLocal(dimensions,  1e9f);
        std::vector<float>  maxLocal(dimensions, -1e9f);
        std::vector<double> sumLocal(dimensions, 0.0);
        for(int t=0; t<nt; t++){
            for(int d=0; d<(int)dimensions; d++){
                if(tMin[t][d] < minLocal[d]) minLocal[d] = tMin[t][d];
                if(tMax[t][d] > maxLocal[d]) maxLocal[d] = tMax[t][d];
                sumLocal[d] += tSum[t][d];
            }
        }

        // MPI reduction across processes
        std::vector<float> minGlobal(dimensions), maxGlobal(dimensions);
        std::vector<double> sumGlobal(dimensions);
        MPI_Reduce(minLocal.data(), minGlobal.data(), dimensions, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(maxLocal.data(), maxGlobal.data(), dimensions, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(sumLocal.data(), sumGlobal.data(), dimensions, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // Compute global mean and broadcast for variance pass
        std::vector<double> mean(dimensions);
        if(rank==0){
            for(int d=0; d<(int)dimensions; d++)
                mean[d] = sumGlobal[d] / nPoints;
        }
        MPI_Bcast(mean.data(), dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Pass 2: parallel variance computation
        for(int t=0; t<nt; t++)
            std::fill(tSum[t].begin(), tSum[t].end(), 0.0);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            double* myVar = tSum[tid].data();

            #pragma omp for schedule(static)
            for(int i=0; i<localN; i++){
                const float* row = &localPoints[i * dimensions];
                for(int d=0; d<(int)dimensions; d++){
                    double diff = row[d] - mean[d];
                    myVar[d] += diff * diff;
                }
            }
        }

        std::vector<double> varLocal(dimensions, 0.0);
        for(int t=0; t<nt; t++)
            for(int d=0; d<(int)dimensions; d++)
                varLocal[d] += tSum[t][d];

        std::vector<double> varGlobal(dimensions);
        MPI_Reduce(varLocal.data(), varGlobal.data(), dimensions, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if(rank==0){
            std::cout << "\n--- Global Statistics ---" << std::endl;
            for(int d=0; d<(int)dimensions; d++){
                std::cout << "Dim " << d << ":"
                          << " Min=" << minGlobal[d]
                          << " Max=" << maxGlobal[d]
                          << " Mean=" << mean[d]
                          << " Variance=" << (varGlobal[d] / nPoints) << "\n";
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    double t1 = MPI_Wtime();

    if(rank == 0){
        std::cout << "\nTotal execution time: " << (t1-t0) << " s" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
