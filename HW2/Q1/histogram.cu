#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include <chrono>
#include <random>
#include <fstream>
#include <sstream>
#include <string>
constexpr size_t NUM_BINS = 4096;
// lengths 1024, 10240, 102400, 1024000
constexpr size_t ARRAY_LEN = 1024;
constexpr size_t SATURATION = 127;
constexpr int RND = 42;

constexpr bool NORMAL_DISTRIBUTION = true;

using histType = std::uint32_t;
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



__global__ void saturate_histogram_kernel(histType* histogram, const size_t num_bins, 
                                        const size_t saturation)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t gridSize = blockDim.x * gridDim.x;
    for(size_t i = idx; i<num_bins; i+=gridSize){
        histogram[i] = histogram[i] > saturation ? saturation : histogram[i];
    }
}

__global__ void histogram1D_basic_kernel(const int* array, histType* histogram,
                                    const size_t array_len, const size_t num_bins)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t gridSize = blockDim.x * gridDim.x;
    for(size_t i = idx; i<array_len; i+=gridSize){
        atomicAdd(histogram + array[i], 1);
    }
}

__global__ void histogram1D_smem_kernel(const int* array, histType* histogram,
                                    const size_t array_len, const size_t num_bins)
{
    extern __shared__ histType sharedMem[];
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t gridSize = blockDim.x * gridDim.x;
    // initialize to 0 the shared memory
    for(size_t i = threadIdx.x; i<num_bins; i+=blockDim.x){
        sharedMem[i] = 0;
    }
    __syncthreads();
    
    // accumlate in shared memmory partial histograms, each block has its own full histogram
    for(size_t i = idx; i<array_len; i+=gridSize){
        atomicAdd(sharedMem + array[i], 1);
    }
    __syncthreads();
    
    // reduce the partial histograms to the global histogram
    for(size_t i = threadIdx.x; i<num_bins; i+=blockDim.x){
        atomicAdd(histogram + i, sharedMem[i]);
    }
}



__host__ void fill_uniform(int* array, const size_t array_len, 
                            const size_t num_bins, std::mt19937& rng)
{
    std::uniform_int_distribution<int> dist(0, num_bins - 1);
    for (size_t idx=0; idx<array_len; ++idx) {
        array[idx] = dist(rng);
    }
}

__host__ void fill_normal(int* array, const size_t array_len, 
                            const size_t num_bins, std::mt19937& rng)
{
    double mean   = (num_bins - 1) / 2.0;
    double std = num_bins / 8.0;
    std::normal_distribution<double> dist(mean, std);

    for (size_t idx=0; idx<array_len; ++idx) {
        double x = dist(rng);
        // Round to nearest integer bin and clamp
        int v = static_cast<int>(std::lround(x));
        if (v < 0) v = 0;
        else if (v >= num_bins) v = num_bins - 1;
        array[idx] = v;
    }
}
// CPU histogram serial implementation, assuming histogram is set to zero
__host__ void histogram_cpu_saturation(const int* array, histType* histogram, 
                                    const size_t array_len, const size_t num_bins, 
                                    const size_t saturation)
{
    for(size_t idx=0; idx<array_len; ++idx){
        histogram[array[idx]]++;
    }
    for(size_t idx=0; idx<num_bins; ++idx){
        histogram[idx] = histogram[idx] > saturation ? saturation : histogram[idx];
    }
}

__host__ int check_histogram(const histType* histogram_host, const histType* histogram_d, const size_t num_bins)
{
    bool error_flag = 0;
    for(size_t idx=0; idx<num_bins; ++idx){
        if(histogram_host[idx] != histogram_d[idx]){
            std::cerr<<" Hist CPU and Hist GPU are different at idx "<< idx <<
            " hist CPU=" << histogram_host[idx] << " hist GPU="<< histogram_d[idx] <<std::endl;
            error_flag = 1;
        }
    }
    return error_flag;
}


__host__ bool write_to_disk(const histType* histogram, const size_t array_len, const size_t num_bins, 
                    const std::string& dist_name)
{
    std::ostringstream oss;
    oss << "histogram_" << array_len << "_bins_" << num_bins
        << "_" << dist_name << ".txt";
    const std::string filename = oss.str();

    std::ofstream ofs(filename);
    if (!ofs) {
        std::cerr << "Error: could not open file '" << filename << "' for writing" << std::endl;
        return false;
    }
    for (std::size_t i = 0; i < num_bins; ++i) {
        ofs << histogram[i] << '\n';
    }
    return true;
}

int main(int argc, char** argv)
{
    using clock = std::chrono::high_resolution_clock;

    // host allocation
    int* array_host = new int[ARRAY_LEN];
    histType* histogram_host = new histType[NUM_BINS];
    histType* histogram_dtohost_basic = new histType[NUM_BINS];
    histType* histogram_dtohost_smem = new histType[NUM_BINS];
    std::memset(histogram_host, 0, NUM_BINS * sizeof(histType));
    std::memset(histogram_dtohost_basic, 0, NUM_BINS * sizeof(histType));
    std::memset(histogram_dtohost_smem, 0, NUM_BINS * sizeof(histType));
    std::mt19937 rng(RND);
    std::string dist_name;

    // initialize input array
    if constexpr(NORMAL_DISTRIBUTION){
        dist_name = "normal";
        fill_normal(array_host,ARRAY_LEN,NUM_BINS,rng);
    }else{
        dist_name = "uniform";
        fill_uniform(array_host,ARRAY_LEN,NUM_BINS,rng);
    }



    // device allocation
    int* array_d = nullptr;
    histType* histogram_d_basic = nullptr;
    histType* histogram_d_smem = nullptr;
    CUDA_CHECK(cudaMalloc(&array_d, ARRAY_LEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&histogram_d_basic, NUM_BINS * sizeof(histType)));
    CUDA_CHECK(cudaMalloc(&histogram_d_smem, NUM_BINS * sizeof(histType)));
    CUDA_CHECK(cudaMemcpy(array_d, array_host, ARRAY_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(histogram_d_basic, 0, NUM_BINS * sizeof(histType)));
    CUDA_CHECK(cudaMemset(histogram_d_smem, 0, NUM_BINS * sizeof(histType)));

    // GPU execution basic
    {
        const size_t threadsPerBlock = 256;
        const size_t blocksHist = (ARRAY_LEN + threadsPerBlock - 1) / threadsPerBlock;
        const size_t blocksSaturate = (NUM_BINS + threadsPerBlock - 1) / threadsPerBlock;
        auto start = clock::now();
        histogram1D_basic_kernel<<<blocksHist,threadsPerBlock>>>(array_d,histogram_d_basic,ARRAY_LEN,NUM_BINS);
        saturate_histogram_kernel<<<blocksSaturate,threadsPerBlock>>>(histogram_d_basic,NUM_BINS,SATURATION);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = clock::now();
        const timingType time_hist_gpu_basic = end - start;
        std::cout << "Execution time GPU basic: " << time_hist_gpu_basic.count() << " ms" << std::endl;

        CUDA_CHECK(cudaMemcpy(histogram_dtohost_basic, histogram_d_basic, NUM_BINS * sizeof(int), 
                            cudaMemcpyDeviceToHost));
    }

    // GPU execution shared memory
    {
        const size_t threadsPerBlock = 256;
        const size_t blocksHist = (ARRAY_LEN + threadsPerBlock - 1) / threadsPerBlock;
        const size_t blocksSaturate = (NUM_BINS + threadsPerBlock - 1) / threadsPerBlock;
        constexpr size_t sharedMemSize = sizeof(histType) * NUM_BINS;
        auto start = clock::now();
        histogram1D_smem_kernel<<<blocksHist,threadsPerBlock,sharedMemSize>>>
                                (array_d,histogram_d_smem,ARRAY_LEN,NUM_BINS);
        saturate_histogram_kernel<<<blocksSaturate,threadsPerBlock>>>(histogram_d_smem,NUM_BINS,SATURATION);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = clock::now();
        const timingType time_hist_gpu_smem = end - start;
        std::cout << "Execution time GPU shared memory: " << time_hist_gpu_smem.count() << " ms" << std::endl;

        CUDA_CHECK(cudaMemcpy(histogram_dtohost_smem, histogram_d_smem, NUM_BINS * sizeof(int), 
                            cudaMemcpyDeviceToHost));
    }

    // CPU execution
    {
        auto start = clock::now();
        histogram_cpu_saturation(array_host,histogram_host,ARRAY_LEN,NUM_BINS,SATURATION);
        auto end = clock::now();
        const timingType time_hist_cpu = end - start;
        std::cout << "Execution time CPU: " << time_hist_cpu.count() << " ms" << std::endl;
    }
    
    // check result correctness and and write to disk
    bool error_flag1 = check_histogram(histogram_host, histogram_dtohost_basic, NUM_BINS);
    //bool error_flag2 = false;
    bool error_flag2 = check_histogram(histogram_host, histogram_dtohost_smem, NUM_BINS);
    if(error_flag1 || error_flag2){
        std::cout<< "Hostograms not equal"<<std::endl;
    }else{
        std::cout<< "Hostograms are equal, good!"<<std::endl;
        if (!write_to_disk(histogram_dtohost_smem, ARRAY_LEN, NUM_BINS, dist_name)) {
            std::cerr << "Failed to write array_host to file" << std::endl;
        }
    }

    // free memory
    delete[] array_host;
    delete[] histogram_host;
    delete[] histogram_dtohost_basic;
    delete[] histogram_dtohost_smem;
    CUDA_CHECK(cudaFree(array_d));
    CUDA_CHECK(cudaFree(histogram_d_basic));
    CUDA_CHECK(cudaFree(histogram_d_smem));

    return 0;
}
