#include <iostream>
#include <fstream>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <mpi.h>
#include <omp.h>

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);

    double t0 = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    uint32_t nPuntos, dimensiones; 
    std::vector<float> puntosGlobales;   
    
    // Lectura binario (solo rank 0)
    if(rank==0){

        std::ifstream file("salida", std::ios::binary);

        if(!file){
            std::cout << "Error reading binary file" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        file.read((char*)&nPuntos, sizeof(uint32_t));
        file.read((char*)&dimensiones, sizeof(uint32_t));

        puntosGlobales.resize(nPuntos * dimensiones);

        std::cout << "Numero de filas: " << nPuntos << std::endl;
        std::cout << "Numero de dimensiones: " << dimensiones << std::endl;

        for(int i=0; i<nPuntos*dimensiones; i++){
            float temp;
            file.read((char*)&temp, sizeof(float));
            puntosGlobales[i] = temp;
        }
    }
    
    // Broadcast de datos
    MPI_Bcast(&nPuntos, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dimensiones, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Scatterv inicial
    int base = nPuntos / size;
    int resto = nPuntos % size;

    std::vector<int> sendcounts(size);
    std::vector<int> offset(size);

    for(int r=0; r<size; r++){
        int puntos_r = (r < resto ? base + 1 : base);
        sendcounts[r] = puntos_r * dimensiones;
    }
    
    offset[0] = 0;
    for(int r=1; r<size; r++){
        offset[r] = offset[r-1] + sendcounts[r-1];
    }

    int localCount = sendcounts[rank];
    std::vector<float> puntosLocales(localCount);

    MPI_Scatterv(puntosGlobales.data(), sendcounts.data(), offset.data(), 
                MPI_FLOAT, puntosLocales.data(), localCount, MPI_FLOAT, 0, 
                MPI_COMM_WORLD);
    
    int localN = localCount / dimensiones;
    std::vector<int> gruposLocales;

    std::cout << "Rank " << rank << " ha recibido " << localCount / dimensiones << " puntos" << std::endl;

    // Bucle principal K-means
    int maxIter = 2000;

    for(int iter = 0; iter < maxIter; iter++){
        // Paso 3: Calcular centroide local
        localN = localCount / dimensiones;
        
        gruposLocales.assign(localN, -1);

        float* centroide = new float[dimensiones]();

        if (localN > 0){
            #pragma omp parallel for reduction(+:centroide[:dimensiones]) schedule(static)
            for (int i=0; i<localN; i++){
                for (int d=0; d<dimensiones; d++){
                    centroide[d] += puntosLocales[i * dimensiones + d];
                }
            }

            for(int d=0; d<dimensiones; d++){
                centroide[d] /= localN;
            }
        }
        

        // Compartir centroides
        std::vector<float> centroidesGlobales(size*dimensiones);

        MPI_Allgather(centroide, dimensiones, MPI_FLOAT, centroidesGlobales.data(), dimensiones, MPI_FLOAT, MPI_COMM_WORLD);

        //liberar espacio
        delete[] centroide;

        if(rank==0){
            std::cout << "\nCentroides iter: " << iter << "\n";
            for(int r=0; r<size; r++){
                std::cout << "Grupo " << r << ": [ ";
                for(int d=0; d<dimensiones; d++){
                    std::cout << centroidesGlobales[r*dimensiones + d] << " ";
                }
                std::cout << "]\n";
            }
        }


        // Paso 4: Reasignar cada punto al centroide más cercano
        int numThreads = omp_get_max_threads();
        std::vector<double> tiempos(numThreads, 0.0);
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            double start = omp_get_wtime();

            #pragma omp for schedule(static)
            for(int i=0; i<localN; i++){
                float mejorDist = 1e308;
                int mejorGrupo = -1;

                for(int c = 0; c<size; c++){
                    float dist = 0.0;

                    for(int d =0; d<dimensiones; d++){
                        float diff = puntosLocales[i*dimensiones +d] - centroidesGlobales[c*dimensiones +d];
                        dist += diff * diff;
                    }

                    if(dist<mejorDist){
                        mejorDist = dist;
                        mejorGrupo = c;
                    }
                }

                gruposLocales[i] = mejorGrupo;
            }

            double end = omp_get_wtime();
            tiempos[tid] =  end-start;
        }

        // if(rank == 0){
        //     std::cout << "\n--- Tiempos por hilos (iter " << iter << ") ---\n";

        //     double max_t = 0.0;
        //     double min_t = 1e9;

        //     for(int i=0; i<numThreads; i++){
        //         std::cout << "Hilo " << i << ": " << tiempos[i] << " s\n";

        //         if(tiempos[i] > max_t) max_t = tiempos[i];
        //         if(tiempos[i] < min_t) min_t = tiempos[i];
        //     }

        //     std::cout << "Desbalance: " << (max_t - min_t) << " s" << std::endl;
        // }
        

        // Contar cambios locales: puntos asignados a un grupo distinto al rank actual
        int cambiosLocales = 0;

        #pragma omp parallel for reduction(+:cambiosLocales)
        for(int i=0; i < localN; i++){
            if (gruposLocales[i] != rank){
                cambiosLocales++;
            }
        }
        

        // Reducir cambios globales
        int cambiosGlobales = 0;
        MPI_Allreduce(&cambiosLocales, &cambiosGlobales, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // Calcular porcentaje global
        float porcentajeCambios = (float)cambiosGlobales / (float)nPuntos;
        if(rank==0){
            std::cout << "\nPorcentaje cambios: " << porcentajeCambios * 100.0 << "%. iter: " << iter << std::endl;
        }

        // Condición de parada
        if (porcentajeCambios < 0.05) {
            if (rank == 0){
                std::cout << "Convergencia alcanzada en iteración " << iter
                        << " con " << porcentajeCambios * 100.0 
                        << "% de cambios globales.\n";
            }
            break;
        }

        //Paso 5: Mandar puntos al rank correspondiente

        //Contar cuántos puntos enviar a cada proceso
        std::vector<int> sendCounts(size, 0);

        #pragma omp parallel
        {
            std::vector<int> localCounts(size,0);

            #pragma omp for nowait
            for(int i=0; i<localN; i++){
                int destino = gruposLocales[i]; //el grupo es el rank destino
                localCounts[destino] += dimensiones; //enviamos 'dimensiones' floats
            }

            #pragma omp critical 
            {
                for(int r=0; r<size; r++){
                    sendCounts[r] += localCounts[r];
                }
            }
        }

        //Construir los offset
        std::vector<int> sdispls(size);
        sdispls[0]=0;
        for(int r=1; r<size; r++){
            sdispls[r] = sdispls[r-1] + sendCounts[r-1];
        }

        //Construir buffer envio
        int totalFloatsToSend = sdispls[size-1] + sendCounts[size-1];
        std::vector<float> sendbuf(totalFloatsToSend);

        std::vector<int> pos(size, 0);
        
        for(int i=0; i<localN; i++){
            int destino = gruposLocales[i];
            int offset = sdispls[destino] + pos[destino];

            for(int d=0; d<dimensiones; d++){
                sendbuf[offset+d] = puntosLocales[i * dimensiones + d];
            }

            pos[destino] += dimensiones;
        }

        //Intercambiar puntos con MPI_Alltoallv
        std::vector<int> recvcounts(size);
        MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        std::vector<int> rdispls(size);
        rdispls[0]=0;
        for(int r=1; r<size; r++){
            rdispls[r] = rdispls[r-1] + recvcounts[r-1];
        }

        int totalRecv = rdispls[size-1] + recvcounts[size-1];
        std::vector<float> recvbuf(totalRecv);

        MPI_Alltoallv(sendbuf.data(), sendCounts.data(), sdispls.data(), MPI_FLOAT, recvbuf.data(), recvcounts.data(), rdispls.data(), MPI_FLOAT, MPI_COMM_WORLD);

        // Actualizar puntosLocales, localCount, localN
        puntosLocales = recvbuf;
        localCount = totalRecv;
        localN = localCount / dimensiones;

        // Calculo estadísticas globales (min,max,media,varianza) - OpenMP buffers por hilo
        int nt = omp_get_max_threads();

        std::vector<std::vector<float>>  tMin(nt, std::vector<float>(dimensiones,  1e9f));
        std::vector<std::vector<float>>  tMax(nt, std::vector<float>(dimensiones, -1e9f));
        std::vector<std::vector<double>> tSum(nt, std::vector<double>(dimensiones, 0.0));

        // Pase 1: min, max, suma (paralelo, sin sincronización)
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            float*  myMin = tMin[tid].data();
            float*  myMax = tMax[tid].data();
            double* mySum = tSum[tid].data();

            #pragma omp for schedule(static)
            for(int i=0; i<localN; i++){
                const float* row = &puntosLocales[i * dimensiones];
                for(int d=0; d<(int)dimensiones; d++){
                    float v = row[d];
                    if(v < myMin[d]) myMin[d] = v;
                    if(v > myMax[d]) myMax[d] = v;
                    mySum[d] += v;
                }
            }
        }

        // Reducción secuencial de hilos -> local
        std::vector<float>  minLocal(dimensiones,  1e9f);
        std::vector<float>  maxLocal(dimensiones, -1e9f);
        std::vector<double> sumLocal(dimensiones, 0.0);
        for(int t=0; t<nt; t++){
            for(int d=0; d<(int)dimensiones; d++){
                if(tMin[t][d] < minLocal[d]) minLocal[d] = tMin[t][d];
                if(tMax[t][d] > maxLocal[d]) maxLocal[d] = tMax[t][d];
                sumLocal[d] += tSum[t][d];
            }
        }

        // Reducir min, max, suma con MPI
        std::vector<float> minGlobal(dimensiones), maxGlobal(dimensiones);
        std::vector<double> sumGlobal(dimensiones);
        MPI_Reduce(minLocal.data(), minGlobal.data(), dimensiones, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(maxLocal.data(), maxGlobal.data(), dimensiones, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(sumLocal.data(), sumGlobal.data(), dimensiones, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // Calcular media global y broadcast para pase 2
        std::vector<double> media(dimensiones);
        if(rank==0){
            for(int d=0; d<(int)dimensiones; d++)
                media[d] = sumGlobal[d] / nPuntos;
        }
        MPI_Bcast(media.data(), dimensiones, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Pase 2: varianza (paralelo, numéricamente estable)
        for(int t=0; t<nt; t++)
            std::fill(tSum[t].begin(), tSum[t].end(), 0.0);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            double* myVar = tSum[tid].data();

            #pragma omp for schedule(static)
            for(int i=0; i<localN; i++){
                const float* row = &puntosLocales[i * dimensiones];
                for(int d=0; d<(int)dimensiones; d++){
                    double diff = row[d] - media[d];
                    myVar[d] += diff * diff;
                }
            }
        }

        std::vector<double> varLocal(dimensiones, 0.0);
        for(int t=0; t<nt; t++)
            for(int d=0; d<(int)dimensiones; d++)
                varLocal[d] += tSum[t][d];

        std::vector<double> varGlobal(dimensiones);
        MPI_Reduce(varLocal.data(), varGlobal.data(), dimensiones, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if(rank==0){
            std::cout << "\n--- Estadísticas globales ---" << std::endl;
            for(int d=0; d<(int)dimensiones; d++){
                std::cout << "Dim " << d << ":" << std::endl;
                std::cout << "  Min      = " << minGlobal[d] << "\n";
                std::cout << "  Max      = " << maxGlobal[d] << "\n";
                std::cout << "  Media    = " << media[d] << "\n";
                std::cout << "  Varianza = " << (varGlobal[d] / nPuntos) << "\n\n";
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    double t1 = MPI_Wtime();
    double total = t1-t0;

    if(rank ==0){
        std::cout << "\nTiempo total del programa: " << total << " s" << std::endl;
    }

    MPI_Finalize();
    return 0;
}