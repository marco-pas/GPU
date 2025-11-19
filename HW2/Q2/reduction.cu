#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include <chrono>
#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>

constexpr size_t ARRAY_LEN = 512 * (1<<9);
constexpr size_t NUM_RUNS = 1;
constexpr size_t REPETITIONS = 1;
constexpr int RND = 42;
constexpr size_t maxBlocks = 1024;


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


__global__ void reduction_basic_kernel(const dataType* array, const size_t array_len, 
                                    dataType* sum)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t gridSize = blockDim.x * gridDim.x;
    dataType localsum = 0.0;
    for(size_t i = idx; i<array_len; i+=gridSize){
        localsum+=array[i];
    }
    atomicAdd(sum, localsum);
}

__global__ void reduction_smem_kernel(const dataType* array, const size_t array_len, 
                                    dataType* sum)
{
    extern __shared__ dataType sharedMem[];
    const size_t tidx = threadIdx.x;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t gridSize = blockDim.x * gridDim.x;
    
    // do local thread sum in register
    dataType localsum = 0.0;
    for(size_t i = idx; i<array_len; i+=gridSize){
        localsum+=array[i];
    }
    // load local sum in shared mem
    sharedMem[tidx] = localsum;
    __syncthreads();
    
    // obtain partial sum in shared mem
    // shared mem must be a power of 2
    for(size_t size = blockDim.x/2; size > 0; size = size/2){
        if(tidx < size) sharedMem[tidx] += sharedMem[tidx+size];
        __syncthreads();
    }
    // reduce the partial sums to global sum
    if(tidx==0) atomicAdd(sum, sharedMem[0]);
}


__host__ void fill_uniform(dataType* array, const size_t array_len, std::mt19937& rng)
{
    std::uniform_real_distribution<dataType> dist(0, 1);
    for (size_t idx=0; idx<array_len; ++idx) {
        array[idx] = dist(rng);
    }
}

// CPU reduction serial implementation
__host__ float reduction_cpu(const dataType* array, const size_t array_len)
{
    dataType sum = 0.0;
    for(size_t idx=0; idx<array_len; ++idx){
        sum+=array[idx];
    }
    return sum;
}


int main(int argc, char** argv)
{
    using clock = std::chrono::high_resolution_clock;
    std::cout << std::fixed << std::setprecision(6);

    for(size_t iter=0; iter<NUM_RUNS; ++iter){
        const size_t mult = pow(2,iter); 
        const size_t current_array_len = ARRAY_LEN * mult;
        std::cout << "\nArray size = " << current_array_len << std::endl;
        // host allocation
        dataType* array_host = new dataType[current_array_len];
        dataType* sum_dtohost_basic = new dataType[1];
        dataType* sum_dtohost_smem = new dataType[1];
        std::mt19937 rng(RND);
        fill_uniform(array_host,current_array_len,rng);

        // device allocation
        dataType* array_d = nullptr;
        dataType* sum_d_basic = nullptr;
        dataType* sum_d_smem = nullptr;
        CUDA_CHECK(cudaMalloc(&array_d, current_array_len * sizeof(dataType)));
        CUDA_CHECK(cudaMalloc(&sum_d_basic, sizeof(dataType)));
        CUDA_CHECK(cudaMalloc(&sum_d_smem,  sizeof(dataType)));
        CUDA_CHECK(cudaMemcpy(array_d, array_host, current_array_len * sizeof(dataType), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(sum_d_basic, 0, sizeof(dataType)));
        CUDA_CHECK(cudaMemset(sum_d_smem, 0,  sizeof(dataType)));

        // GPU execution basic
        {
            const size_t threadsPerBlock = 256;
            const size_t blocks = (current_array_len + threadsPerBlock - 1) / threadsPerBlock;      
            CUDA_CHECK(cudaMemset(sum_d_basic, 0, sizeof(dataType)));
            reduction_basic_kernel<<<std::min(blocks,maxBlocks),threadsPerBlock>>>(array_d,current_array_len,sum_d_basic);
            CUDA_CHECK(cudaDeviceSynchronize());
            auto start = clock::now();
            for(size_t i=0; i<REPETITIONS; ++i){
                CUDA_CHECK(cudaMemset(sum_d_basic, 0, sizeof(dataType)));
                reduction_basic_kernel<<<std::min(blocks,maxBlocks),threadsPerBlock>>>(array_d,current_array_len,sum_d_basic);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());
            auto end = clock::now();
            CUDA_CHECK(cudaMemcpy(sum_dtohost_basic, sum_d_basic, sizeof(dataType), cudaMemcpyDeviceToHost));
            const timingType time_reduction_gpu_basic = end - start;
            std::cout << "Reduction GPU basic = "<< *sum_dtohost_basic << std::endl;
            std::cout << "Execution time GPU basic: " << 
            time_reduction_gpu_basic.count()/REPETITIONS << " ms" << std::endl;
        }

        
        // GPU execution shared
        {
            const size_t threadsPerBlock = 256;
            const size_t blocks = (current_array_len + threadsPerBlock - 1) / threadsPerBlock;
            const size_t sharedMemSize = sizeof(dataType) * threadsPerBlock;   
            CUDA_CHECK(cudaMemset(sum_d_smem, 0, sizeof(dataType)));
            reduction_smem_kernel<<<std::min(blocks,maxBlocks),threadsPerBlock,sharedMemSize>>>(array_d,current_array_len,sum_d_smem);
            CUDA_CHECK(cudaDeviceSynchronize());
            auto start = clock::now();
            for(size_t i=0; i<REPETITIONS; ++i){
                CUDA_CHECK(cudaMemset(sum_d_smem, 0, sizeof(dataType)));
                reduction_smem_kernel<<<std::min(blocks,maxBlocks),threadsPerBlock,sharedMemSize>>>(array_d,current_array_len,sum_d_smem);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());
            auto end = clock::now();
            CUDA_CHECK(cudaMemcpy(sum_dtohost_smem, sum_d_smem, sizeof(dataType), cudaMemcpyDeviceToHost));
            const timingType time_reduction_gpu_smem = end - start;
            std::cout << "Reduction GPU shared = "<< *sum_dtohost_smem << std::endl;
            std::cout << "Execution time GPU shared: " << 
            time_reduction_gpu_smem.count()/REPETITIONS << " ms" << std::endl;
        }
        
        // CPU execution
        dataType sum_cpu = 0.0;
        {
            sum_cpu = reduction_cpu(array_host,current_array_len);
            auto start = clock::now();
            for(size_t i=0; i<REPETITIONS; ++i){
                sum_cpu = reduction_cpu(array_host,current_array_len);
            }
            auto end = clock::now();
            const timingType time_reduction_cpu = end - start;
            std::cout << "Reduction CPU = "<< sum_cpu << std::endl;
            std::cout << "Execution time CPU: " << time_reduction_cpu.count()/REPETITIONS << " ms" << std::endl;
        }
        
        // check result correctness
        const dataType tol = 1e-3;
        const bool error_flag1 = std::fabs(sum_cpu - *sum_dtohost_basic) < tol ? 0 : 1;
        const bool error_flag2 = std::fabs(sum_cpu - *sum_dtohost_smem) < tol ? 0 : 1;
        if(error_flag1 || error_flag2){
            std::cout<< "Reductions not equal"<<std::endl;
        }else{
            std::cout<< "Reductions are equal, good!"<<std::endl;
        }

        // free memory
        delete[] array_host;
        CUDA_CHECK(cudaFree(array_d));
        CUDA_CHECK(cudaFree(sum_d_basic));
        CUDA_CHECK(cudaFree(sum_d_smem));
    }

    return 0;
}
