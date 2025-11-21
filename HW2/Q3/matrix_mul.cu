#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include <chrono>
#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>

// Matrix sizes: A(SIZE_I, SIZE_K) B(SIZE_K,SIZE_J) C(SIZE_I,SIZE_J)
/**
#if 1
constexpr size_t SIZE_I = 1024;
constexpr size_t SIZE_K = 1024;
constexpr size_t SIZE_J = 1024;
#else
constexpr size_t SIZE_I = 513;
constexpr size_t SIZE_K = 8192;
constexpr size_t SIZE_J = 1023;
#endif
*/


#ifndef MAT_CONF
#define MAT_CONF 8
#endif

#if   MAT_CONF == 1
// 256 x 256
constexpr size_t SIZE_I = 256;
constexpr size_t SIZE_K = 256;
constexpr size_t SIZE_J = 256;

#elif MAT_CONF == 2
// 512 x 512
constexpr size_t SIZE_I = 512;
constexpr size_t SIZE_K = 512;
constexpr size_t SIZE_J = 512;

#elif MAT_CONF == 3
// 1024 x 1024
constexpr size_t SIZE_I = 1024;
constexpr size_t SIZE_K = 1024;
constexpr size_t SIZE_J = 1024;

#elif MAT_CONF == 4
// 1536 x 1536
constexpr size_t SIZE_I = 1536;
constexpr size_t SIZE_K = 1536;
constexpr size_t SIZE_J = 1536;

#elif MAT_CONF == 5
// 2048 x 2048
constexpr size_t SIZE_I = 2048;
constexpr size_t SIZE_K = 2048;
constexpr size_t SIZE_J = 2048;

#elif MAT_CONF == 6
// 3072 x 3072
constexpr size_t SIZE_I = 3072;
constexpr size_t SIZE_K = 3072;
constexpr size_t SIZE_J = 3072;

#elif MAT_CONF == 7
constexpr size_t SIZE_I = 1846;
constexpr size_t SIZE_K = 1846;
constexpr size_t SIZE_J = 1846;

#elif MAT_CONF == 8
constexpr size_t SIZE_I = 2049;
constexpr size_t SIZE_K = 2049;
constexpr size_t SIZE_J = 2049;
#else
#error "MAT_CONF must be between 1 and 8"
#endif


constexpr size_t STRIPE_I = 4;
constexpr size_t STRIPE_J = 4;
constexpr size_t WARP_SIZE = 32;

constexpr size_t REPETITIONS = 5;
constexpr int RND = 42;
constexpr size_t maxBlocks = 2048;


using dataType = float;
using timingType = std::chrono::duration<double, std::milli>;

// error-checking
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " : " << cudaGetErrorString(err) << std::endl;        \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)


__global__ void basic_gemmm(const dataType* matrixA, const dataType* matrixB, 
                                    dataType* matrixC, const size_t size_i, const size_t size_j,
                                    const size_t size_k)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gridSize = blockDim.x * gridDim.x;
    const int totalElements = size_i * size_j;
    for(int el = idx; el<totalElements; el+=gridSize){
        dataType sum = 0.0;
        const int i = el / size_j;
        const int j = el % size_j; 
        for(int k=0; k<size_k; ++k){
            sum += matrixA[size_k * i + k] * matrixB[size_j * k + j];
        }
        matrixC[el] = sum;
    }
}

__global__ void tiled_gemm(const dataType* matrixA, const dataType* matrixB, 
                                    dataType* matrixC, const size_t size_i, const size_t size_j,
                                    const size_t size_k, 
                                    const size_t tile_i, const size_t tile_j, const size_t tile_k)
{
    extern __shared__ dataType sharedMem[];
    dataType* sharedMatrixA = sharedMem; // tile_i * tile_k elements of matrixA
    dataType* sharedMatrixB = sharedMem + tile_i * tile_k; // tile_k * tile_j elements of matrixB 

    // number of tiles in each dimension i,j,k --> ensure to cover entire matrices
    const int num_patches_i = (size_i + tile_i - 1) / tile_i;
    const int num_pathces_j = (size_j + tile_j - 1) / tile_j;
    const int num_pathces_k = (size_k + tile_k - 1) / tile_k;
    const int totPatches =  num_patches_i * num_pathces_j;
    const int num_blocks = gridDim.x;

    // assumes tiles multiple of stripes
    // assumes blockDim.x = num_threads_i * num_threads_j
    const int num_threads_i = tile_i / STRIPE_I;
    const int num_threads_j = tile_j / STRIPE_J;
    const int idx_ti = threadIdx.x / num_threads_j;
    const int idx_tj = threadIdx.x % num_threads_j;

    // loop over all patches to cover entire matrixC
    for(int patchIdx=blockIdx.x; patchIdx<totPatches; patchIdx+=num_blocks){
        const int patch_i = patchIdx / num_pathces_j;
        const int patch_j = patchIdx % num_pathces_j;
        const int offset_i = patch_i * tile_i;
        const int offset_j = patch_j * tile_j;
        
        // local sum in register
        dataType sum[STRIPE_I * STRIPE_J] = {0.0};
        // loop over shared index k
        for(int patch_k=0; patch_k<num_pathces_k; ++patch_k){
            const int offset_k = patch_k * tile_k;
            // load patch = tile_i * tile_k matrixA
            for(int sidx=threadIdx.x; sidx<tile_i*tile_k; sidx+=blockDim.x){
                const int local_i = offset_i + sidx / tile_k;
                const int local_k = offset_k + sidx % tile_k;
                if( local_i < size_i && local_k < size_k)
                    sharedMatrixA[sidx] = matrixA[ local_i * size_k + local_k];
                else
                    sharedMatrixA[sidx] = 0.0;
            }
            // load patch = tile_k * tile_j matrixB
            for(int sidx=threadIdx.x; sidx<tile_k*tile_j; sidx+=blockDim.x){
                const int local_k = offset_k + sidx / tile_j;
                const int local_j = offset_j + sidx % tile_j;
                if( local_j < size_j && local_k < size_k)
                    sharedMatrixB[sidx] = matrixB[ local_k * size_j + local_j];
                else
                    sharedMatrixB[sidx] = 0.0;
            }
            __syncthreads();
            // compute partial multiplication and store in each thread register
            #pragma unroll
            for(int idx_k=0; idx_k<tile_k; ++idx_k){
                for(int idx_i=0; idx_i<STRIPE_I; ++idx_i){
                    for(int idx_j=0; idx_j<STRIPE_J; ++idx_j)
                    sum[idx_i * STRIPE_J + idx_j] += 
                    sharedMatrixA[(idx_i * num_threads_i + idx_ti) * tile_k + idx_k] * 
                    sharedMatrixB[tile_j * idx_k + idx_j*num_threads_j + idx_tj];
                }
            }
            __syncthreads();
        }
        // copy matrixC elements in the global memory
        #pragma unroll
        for(int idx_i=0; idx_i<STRIPE_I; ++idx_i){
            for(int idx_j=0; idx_j<STRIPE_J; ++idx_j){
                const int idx_i_c = patch_i * tile_i + (idx_i * num_threads_i + idx_ti); 
                const int idx_j_c = patch_j * tile_j + (idx_j * num_threads_j + idx_tj);
                if( idx_i_c < size_i && idx_j_c < size_j)
                matrixC[idx_i_c * size_j + idx_j_c] = sum[idx_i * STRIPE_J + idx_j];
            }
        }
    }
}


__host__ void fill_random(dataType* matrix, const size_t size, std::mt19937& rng)
{
    std::uniform_real_distribution<dataType> dist(0, 1);
    for (size_t idx=0; idx<size; ++idx) {
        matrix[idx] = dist(rng);
    }
}

__host__ dataType check_matrix(dataType* matrixHost, dataType* matrixD, const size_t size)
{
    dataType Linf =0.0;
    for(int idx=0; idx<size; ++idx){
        const dataType delta = std::fabs(matrixHost[idx]-matrixD[idx]);
        if( delta > Linf) Linf = delta;
    }
    return Linf;
}

// CPU matrix_mul serial implementation
__host__ void matrix_mul_cpu(const dataType* matrixA, const dataType* matrixB, 
                                    dataType* matrixC, const size_t size_i, const size_t size_j,
                                    const size_t size_k)
{
    for(int i=0; i<size_i; ++i){
        for(int j=0; j<size_j; ++j){
            dataType sum = 0.0;
            for(int k=0; k<size_k; ++k){
                sum += matrixA[size_k * i + k] * matrixB[size_j * k + j];
            }
            matrixC[size_j * i + j] = sum;
        }
    }
}


int main(int argc, char** argv)
{
    using clock = std::chrono::high_resolution_clock;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Matrix sizes: A ("<<SIZE_I <<"x"<< SIZE_K <<")" 
        << " B ("<<SIZE_K <<"x"<< SIZE_J <<")"
        << " C ("<<SIZE_I <<"x"<< SIZE_J <<") \n" <<  std::endl;

    // host allocation
    dataType* matrixA_host = new dataType[SIZE_I*SIZE_K];
    dataType* matrixB_host = new dataType[SIZE_J*SIZE_K];
    dataType* matrixC_host = new dataType[SIZE_I*SIZE_J];
    dataType* matrixC_dtohost = new dataType[SIZE_I*SIZE_J];
    std::mt19937 rng(RND);
    fill_random(matrixA_host, SIZE_I*SIZE_K, rng);
    fill_random(matrixB_host, SIZE_J*SIZE_K, rng);
    std::memset(matrixC_host,0,SIZE_I*SIZE_J*sizeof(dataType));
    std::memset(matrixC_dtohost,0,SIZE_I*SIZE_J*sizeof(dataType));
    
    // device allocation
    dataType* matrixA_d = nullptr;
    dataType* matrixB_d = nullptr;
    dataType* matrixC_d = nullptr;
    CUDA_CHECK(cudaMalloc(&matrixA_d, SIZE_I * SIZE_K * sizeof(dataType)));
    CUDA_CHECK(cudaMalloc(&matrixB_d, SIZE_J * SIZE_K * sizeof(dataType)));
    CUDA_CHECK(cudaMalloc(&matrixC_d, SIZE_I * SIZE_J * sizeof(dataType)));
    CUDA_CHECK(cudaMemcpy(matrixA_d, matrixA_host, SIZE_I * SIZE_K * sizeof(dataType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(matrixB_d, matrixB_host, SIZE_J * SIZE_K * sizeof(dataType), cudaMemcpyHostToDevice));
    // CPU execution
#if 1
    {
        matrix_mul_cpu(matrixA_host, matrixB_host, matrixC_host,SIZE_I,SIZE_J,SIZE_K);
        auto start = clock::now();
        const size_t repCPU = 2;
        for(size_t i=0; i<REPETITIONS; ++i){
            matrix_mul_cpu(matrixA_host, matrixB_host, matrixC_host,SIZE_I,SIZE_J,SIZE_K);
        }
        auto end = clock::now();
        const timingType time_matrix_mul_cpu = end - start;
        //std::cout << "matrix_mul CPU = "<< sum_cpu << std::endl;
        std::cout << "\nExecution time CPU: " << time_matrix_mul_cpu.count()/REPETITIONS << " ms" << std::endl;
    }
#endif
    // GPU execution basic
    {
        const size_t threadsPerBlock = 256;
        const size_t bb = ( SIZE_I*SIZE_J + threadsPerBlock -1 )/threadsPerBlock;
        const size_t blocks = std::min(bb,maxBlocks);      
        std::cout <<"\nGPU basic - threadsPerBlock="<<threadsPerBlock 
        << " blocks=" << blocks << std::endl;
        CUDA_CHECK(cudaMemset(matrixC_d, 0, SIZE_I * SIZE_J * sizeof(dataType)));
        basic_gemmm<<<blocks,threadsPerBlock>>>(matrixA_d, matrixB_d, matrixC_d,
                                            SIZE_I,SIZE_J,SIZE_K);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = clock::now();
        for(size_t i=0; i<REPETITIONS; ++i){
            CUDA_CHECK(cudaMemset(matrixC_d, 0, SIZE_I * SIZE_J * sizeof(dataType)));
            basic_gemmm<<<blocks,threadsPerBlock>>>(matrixA_d, matrixB_d, matrixC_d,
                                                SIZE_I,SIZE_J,SIZE_K);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        auto end = clock::now();
        CUDA_CHECK(cudaMemcpy(matrixC_dtohost, matrixC_d, SIZE_I*SIZE_J* sizeof(dataType), cudaMemcpyDeviceToHost));
        const timingType time_matrix_mul_gpu_basic = end - start;
        std::cout << "Matrix mul GPU basic - Linf norm vs CPU "
                << check_matrix(matrixC_host,matrixC_dtohost, SIZE_I*SIZE_J) << std::endl;
        std::cout << "Execution time GPU basic: " << 
        time_matrix_mul_gpu_basic.count()/REPETITIONS << " ms" << std::endl;
    }


    // GPU execution tiled
#if 1
    {
        const size_t tile_i = 32, tile_j = 32, tile_k = 32;
        const size_t num_patches_i = (SIZE_I + tile_i - 1) / tile_i;
        const size_t num_pathces_j = (SIZE_J + tile_j - 1) / tile_j;
        const size_t threadsPerBlock = tile_i * tile_j / (STRIPE_I * STRIPE_J);
        const size_t blocks = std::min(maxBlocks, num_patches_i*num_pathces_j);
        const size_t sharedMemSize = sizeof(dataType) * (tile_i * tile_k + tile_k * tile_j);
        std::cout <<"\nGPU tiled_gemm tile_i="<< tile_i << " tile_j="<<tile_j
        << " tile_k=" << tile_k << " - threadsPerBlock="<<threadsPerBlock 
        << " blocks=" << blocks
        << " - sharedMem size="<< sharedMemSize 
        << " bytes - register size="<< sizeof(dataType) * STRIPE_I * STRIPE_J * threadsPerBlock
        << " bytes" << std::endl;
        CUDA_CHECK(cudaMemset(matrixC_d, 0, SIZE_I * SIZE_J * sizeof(dataType)));
        tiled_gemm<<<blocks,threadsPerBlock,sharedMemSize>>>(matrixA_d, matrixB_d, matrixC_d,
                                            SIZE_I,SIZE_J,SIZE_K,tile_i,tile_j,tile_k);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = clock::now();
        for(size_t i=0; i<REPETITIONS; ++i){
            //CUDA_CHECK(cudaMemset(matrixC_d, 0, SIZE_I * SIZE_J * sizeof(dataType)));
            tiled_gemm<<<blocks,threadsPerBlock,sharedMemSize>>>(matrixA_d, matrixB_d, matrixC_d,
                                            SIZE_I,SIZE_J,SIZE_K,tile_i,tile_j,tile_k);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        auto end = clock::now();
        CUDA_CHECK(cudaMemcpy(matrixC_dtohost, matrixC_d, SIZE_I * SIZE_J * sizeof(dataType), cudaMemcpyDeviceToHost));
        const timingType time_matrix_mul_gpu_tiled = end - start;
        std::cout << "Matrix mul GPU tiled_gemm - Linf norm vs CPU "
                << check_matrix(matrixC_host,matrixC_dtohost, SIZE_I*SIZE_J) << std::endl;
        std::cout << "Execution time GPU tiled_gemm: " << 
        time_matrix_mul_gpu_tiled.count()/REPETITIONS << " ms" << std::endl;
    }
#endif

    // GPU execution tiled
#if 1
    {
        const size_t tile_i = 32, tile_j = 64, tile_k = 32;
        const size_t num_patches_i = (SIZE_I + tile_i - 1) / tile_i;
        const size_t num_pathces_j = (SIZE_J + tile_j - 1) / tile_j;
        const size_t threadsPerBlock = tile_i * tile_j / (STRIPE_I * STRIPE_J);
        const size_t blocks = std::min(maxBlocks, num_patches_i*num_pathces_j);
        const size_t sharedMemSize = sizeof(dataType) * (tile_i * tile_k + tile_k * tile_j);
        std::cout <<"\nGPU tiled_gemm tile_i="<< tile_i << " tile_j="<<tile_j
        << " tile_k=" << tile_k << " - threadsPerBlock="<<threadsPerBlock 
        << " blocks=" << blocks
        << " - sharedMem size="<< sharedMemSize 
        << " bytes - register size="<< sizeof(dataType) * STRIPE_I * STRIPE_J * threadsPerBlock
        << " bytes" << std::endl;
        CUDA_CHECK(cudaMemset(matrixC_d, 0, SIZE_I * SIZE_J * sizeof(dataType)));
        tiled_gemm<<<blocks,threadsPerBlock,sharedMemSize>>>(matrixA_d, matrixB_d, matrixC_d,
                                            SIZE_I,SIZE_J,SIZE_K,tile_i,tile_j,tile_k);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = clock::now();
        for(size_t i=0; i<REPETITIONS; ++i){
            //CUDA_CHECK(cudaMemset(matrixC_d, 0, SIZE_I * SIZE_J * sizeof(dataType)));
            tiled_gemm<<<blocks,threadsPerBlock,sharedMemSize>>>(matrixA_d, matrixB_d, matrixC_d,
                                            SIZE_I,SIZE_J,SIZE_K,tile_i,tile_j,tile_k);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        auto end = clock::now();
        CUDA_CHECK(cudaMemcpy(matrixC_dtohost, matrixC_d, SIZE_I * SIZE_J * sizeof(dataType), cudaMemcpyDeviceToHost));
        const timingType time_matrix_mul_gpu_tiled = end - start;
        std::cout << "Matrix mul GPU tiled_gemm - Linf norm vs CPU "
                << check_matrix(matrixC_host,matrixC_dtohost, SIZE_I*SIZE_J) << std::endl;
        std::cout << "Execution time GPU tiled_gemm: " << 
        time_matrix_mul_gpu_tiled.count()/REPETITIONS << " ms" << std::endl;
    }
#endif
    
    // GPU execution tiled
#if 0
    {
        const size_t tile_i = 64, tile_j = 64, tile_k = 32;
        const size_t num_patches_i = (SIZE_I + tile_i - 1) / tile_i;
        const size_t num_pathces_j = (SIZE_J + tile_j - 1) / tile_j;
        const size_t threadsPerBlock = tile_i * tile_j / (STRIPE_I * STRIPE_J);
        const size_t blocks = std::min(maxBlocks, num_patches_i*num_pathces_j);
        const size_t sharedMemSize = sizeof(dataType) * (tile_i * tile_k + tile_k * tile_j);
        std::cout <<"\nGPU tiled_gemm tile_i="<< tile_i << " tile_j="<<tile_j
        << " tile_k=" << tile_k << " - threadsPerBlock="<<threadsPerBlock 
        << " blocks=" << blocks
        << " - sharedMem size="<< sharedMemSize 
        << " bytes - register size="<< sizeof(dataType) * STRIPE_I * STRIPE_J * threadsPerBlock
        << " bytes" << std::endl;
        CUDA_CHECK(cudaMemset(matrixC_d, 0, SIZE_I * SIZE_J * sizeof(dataType)));
        tiled_gemm<<<blocks,threadsPerBlock,sharedMemSize>>>(matrixA_d, matrixB_d, matrixC_d,
                                            SIZE_I,SIZE_J,SIZE_K,tile_i,tile_j,tile_k);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = clock::now();
        for(size_t i=0; i<REPETITIONS; ++i){
            //CUDA_CHECK(cudaMemset(matrixC_d, 0, SIZE_I * SIZE_J * sizeof(dataType)));
            tiled_gemm<<<blocks,threadsPerBlock,sharedMemSize>>>(matrixA_d, matrixB_d, matrixC_d,
                                            SIZE_I,SIZE_J,SIZE_K,tile_i,tile_j,tile_k);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        auto end = clock::now();
        CUDA_CHECK(cudaMemcpy(matrixC_dtohost, matrixC_d, SIZE_I * SIZE_J * sizeof(dataType), cudaMemcpyDeviceToHost));
        const timingType time_matrix_mul_gpu_tiled = end - start;
        std::cout << "Matrix mul GPU tiled_gemm - Linf norm vs CPU "
                << check_matrix(matrixC_host,matrixC_dtohost, SIZE_I*SIZE_J) << std::endl;
        std::cout << "Execution time GPU tiled_gemm: " << 
        time_matrix_mul_gpu_tiled.count()/REPETITIONS << " ms" << std::endl;
    }
#endif
    // GPU execution tiled
#if 1
    {
        const size_t tile_i = 64, tile_j = 128, tile_k = 32;
        const size_t num_patches_i = (SIZE_I + tile_i - 1) / tile_i;
        const size_t num_pathces_j = (SIZE_J + tile_j - 1) / tile_j;
        const size_t threadsPerBlock = tile_i * tile_j / (STRIPE_I * STRIPE_J);
        const size_t blocks = std::min(maxBlocks, num_patches_i*num_pathces_j);
        const size_t sharedMemSize = sizeof(dataType) * (tile_i * tile_k + tile_k * tile_j);
        std::cout <<"\nGPU tiled_gemm tile_i="<< tile_i << " tile_j="<<tile_j
        << " tile_k=" << tile_k << " - threadsPerBlock="<<threadsPerBlock 
        << " blocks=" << blocks
        << " - sharedMem size="<< sharedMemSize 
        << " bytes - register size="<< sizeof(dataType) * STRIPE_I * STRIPE_J * threadsPerBlock
        << " bytes" << std::endl;
        CUDA_CHECK(cudaMemset(matrixC_d, 0, SIZE_I * SIZE_J * sizeof(dataType)));
        tiled_gemm<<<blocks,threadsPerBlock,sharedMemSize>>>(matrixA_d, matrixB_d, matrixC_d,
                                            SIZE_I,SIZE_J,SIZE_K,tile_i,tile_j,tile_k);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = clock::now();
        for(size_t i=0; i<REPETITIONS; ++i){
            //CUDA_CHECK(cudaMemset(matrixC_d, 0, SIZE_I * SIZE_J * sizeof(dataType)));
            tiled_gemm<<<blocks,threadsPerBlock,sharedMemSize>>>(matrixA_d, matrixB_d, matrixC_d,
                                            SIZE_I,SIZE_J,SIZE_K,tile_i,tile_j,tile_k);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        auto end = clock::now();
        CUDA_CHECK(cudaMemcpy(matrixC_dtohost, matrixC_d, SIZE_I * SIZE_J * sizeof(dataType), cudaMemcpyDeviceToHost));
        const timingType time_matrix_mul_gpu_tiled = end - start;
        std::cout << "Matrix mul GPU tiled_gemm - Linf norm vs CPU "
                << check_matrix(matrixC_host,matrixC_dtohost, SIZE_I*SIZE_J) << std::endl;
        std::cout << "Execution time GPU tiled_gemm: " << 
        time_matrix_mul_gpu_tiled.count()/REPETITIONS << " ms" << std::endl;
    }
#endif
// GPU execution tiled

    // free memory
    delete[] matrixA_host;
    delete[] matrixB_host;
    delete[] matrixC_host;
    delete[] matrixC_dtohost;
    CUDA_CHECK(cudaFree(matrixA_d));
    CUDA_CHECK(cudaFree(matrixB_d));
    CUDA_CHECK(cudaFree(matrixC_d));

    return 0;
}
