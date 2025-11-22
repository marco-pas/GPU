#include <iostream>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>


constexpr int NUM_CUDASTREAMS = 4;
constexpr int REPETITIONS = 0;


#define RUN_TYPE 1

#if RUN_TYPE == 1
constexpr int numVectors = 1;
constexpr int Ns[numVectors] = { 
    2*524288,
};

constexpr int numSegSize = 1;
constexpr int segSize[numSegSize] = {
    160000
};
#elif RUN_TYPE == 2
constexpr int numVectors = 1;
constexpr int Ns[numVectors] = { 
    4*524288,
};

constexpr int numSegSize = 6;
constexpr int segSize[numSegSize] = {
    2048,
    8192,
    32768,
    131072,
    160000,
    524288
};

#else
constexpr int numVectors = 9;
constexpr int Ns[numVectors] = { 
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    2*524288,
    4*524288
};

constexpr int numSegSize = 4;
constexpr int segSize[numSegSize] = {
    8192,
    32768,
    65536,
    131072
};
#endif

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


__global__
void vectorAddKernel(const dataType* A, const dataType* B, dataType* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

void vectorAddCPU(const dataType* A, const dataType* B, dataType* C, int N) {
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

void fill_vectors_host(dataType* A, dataType* B, const size_t length)
{
    for (int i = 0; i < length; ++i) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }
}


bool check_result(dataType* gpu_result, dataType* cpu_ref, const size_t length, double* max_error)
{
    *max_error = 0.0;
    const double eps = 1e-5;
    bool check = true;
    for (int i = 0; i < length; ++i) {
        double diff = std::fabs(double(gpu_result[i]) - double(cpu_ref[i]));
        if (diff > *max_error) *max_error = diff;
        if (diff > eps) {
            std::cerr << "Index " << i
                        << ": GPU=" << gpu_result[i]
                        << " CPU=" << cpu_ref[i]
                        << " diff=" << diff << std::endl;
            check = false;
            break;
        }
    }
    return check;
}

int main(int argc, char** argv) {
    using clock = std::chrono::high_resolution_clock;

    const int blockSize = 256; 

    for (int idx = 0; idx < numVectors; ++idx) {
        const int vec_length = Ns[idx];
        const size_t bytes = static_cast<size_t>(vec_length) * sizeof(dataType);

        std::cout << "\n\n\n --- Run " << (idx + 1) << "/" << numVectors << " - vector length=" 
        << vec_length << " --- " << std::endl;

        // 1. Allocate host memory (pinned for H2D/D2H buffers)
        dataType *a_host   = nullptr;
        dataType *b_host   = nullptr;
        dataType *c_host   = nullptr;
        dataType *c_dtohost = nullptr;

        CUDA_CHECK(cudaMallocHost((void**)&a_host,   bytes));  // pinned
        CUDA_CHECK(cudaMallocHost((void**)&b_host,   bytes));  // pinned
        CUDA_CHECK(cudaMallocHost((void**)&c_dtohost,   bytes));  // pinned
        c_host = new dataType[vec_length];                            // normal host memory

        // 2. Allocate device memory
        dataType *a_d = nullptr, *b_d = nullptr, *c_d = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&a_d, bytes));
        CUDA_CHECK(cudaMalloc((void**)&b_d, bytes));
        CUDA_CHECK(cudaMalloc((void**)&c_d, bytes));

        // 3. Initialize host memory
        fill_vectors_host(a_host, b_host,vec_length);

        // CPU vector add 
    #if 1
    {   
        std::cout << "\nRun CPU" << std::endl;
        vectorAddCPU(a_host, b_host, c_host, vec_length);    
        auto start = clock::now();
        for(int rep=0; rep<REPETITIONS; ++rep){
            vectorAddCPU(a_host, b_host, c_host, vec_length);
        }
        auto end = clock::now();
        const timingType time_cpu = end - start;
        std::cout << "Execution time CPU: " << time_cpu.count()/REPETITIONS << " ms" << std::endl;
    }
    #endif


        // GPU vector add basic 
    #if 0   
    {   const int gridSize = (vec_length + blockSize - 1) / blockSize;
        std::cout << "\nRun GPU basic" << std::endl;
        std::cout << "GridSize = " << gridSize
                  << ", blockSize = " << blockSize
                  << ", total threads = " << gridSize * blockSize << std::endl;
        CUDA_CHECK(cudaMemcpy(a_d, a_host, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(b_d, b_host, bytes, cudaMemcpyHostToDevice));
        vectorAddKernel<<<gridSize, blockSize>>>(a_d, b_d, c_d, vec_length);
        CUDA_CHECK(cudaMemcpy(c_dtohost, c_d, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());    
        auto start = clock::now();
        for(int rep=0; rep<REPETITIONS; ++rep){
            CUDA_CHECK(cudaMemcpy(a_d, a_host, bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(b_d, b_host, bytes, cudaMemcpyHostToDevice));
            vectorAddKernel<<<gridSize, blockSize>>>(a_d, b_d, c_d, vec_length);
            CUDA_CHECK(cudaMemcpy(c_dtohost, c_d, bytes, cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        auto end = clock::now();
        const timingType time_gpu_basic = end - start;
        std::cout << "Execution time GPU basic: "
            << time_gpu_basic.count()/REPETITIONS << " ms" << std::endl;
        double max_error = 0.0;
        bool check = check_result(c_dtohost,c_host, vec_length, &max_error);
        if (check) {
           // std::cout << "Check PASSED max error " << max_error << "\n"<< std::endl;
        } else {
            std::cout << "Check FAILED max error " << max_error << "\n"<< std::endl;
        }
    }
    #endif


        // GPU vector add stream 
    #if 1
    {   
        std::cout << "\nRun GPU streams" << std::endl;
        cudaStream_t streams[NUM_CUDASTREAMS];
        for(int i=0; i<NUM_CUDASTREAMS;++i){
            CUDA_CHECK(cudaStreamCreate(streams+i));
        }
        for(int seg_size_idx = 0; seg_size_idx<numSegSize; ++seg_size_idx){
            const int seg_size = std::min(segSize[seg_size_idx], vec_length);
            const int num_segments = static_cast<int>(
                                    std::ceil(double(vec_length) / double(seg_size) ) ); 
            std::cout << "Segment size = " << seg_size << " - num segments = "<< num_segments <<std::endl;
            
            const int gridSize = (seg_size + blockSize - 1) / blockSize;
            std::cout << "GridSize = " << gridSize
                  << ", blockSize = " << blockSize
                  << ", total threads = " << gridSize * blockSize << std::endl;
            
            for(int idx=0; idx<num_segments; ++idx){
                const int stream_idx = idx % NUM_CUDASTREAMS;
                const int idx_first_element = idx * seg_size;
                const int seg_length = std::min(vec_length-idx_first_element,seg_size);
                const size_t bytes_seg = static_cast<size_t>(seg_length) * sizeof(dataType);
                CUDA_CHECK(cudaMemcpyAsync(a_d+idx_first_element, a_host+idx_first_element, 
                            bytes_seg, cudaMemcpyHostToDevice, streams[stream_idx]));
                CUDA_CHECK(cudaMemcpyAsync(b_d+idx_first_element, b_host+idx_first_element, 
                            bytes_seg, cudaMemcpyHostToDevice, streams[stream_idx]));
                vectorAddKernel<<<gridSize, blockSize, 0, streams[stream_idx]>>>
                                (a_d+idx_first_element, b_d+idx_first_element, 
                                c_d+idx_first_element, seg_length);
                CUDA_CHECK(cudaMemcpyAsync(c_dtohost+idx_first_element, c_d+idx_first_element, 
                            bytes_seg, cudaMemcpyDeviceToHost, streams[stream_idx]));
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());
            auto start = clock::now();
            for(int rep=0; rep<REPETITIONS; ++rep){
            for(int idx=0; idx<num_segments; ++idx){
                const int stream_idx = idx % NUM_CUDASTREAMS;
                const int idx_first_element = idx * seg_size;
                const int seg_length = std::min(vec_length-idx_first_element,seg_size);
                const size_t bytes_seg = static_cast<size_t>(seg_length) * sizeof(dataType);
                CUDA_CHECK(cudaMemcpyAsync(a_d+idx_first_element, a_host+idx_first_element, 
                            bytes_seg, cudaMemcpyHostToDevice, streams[stream_idx]));
                CUDA_CHECK(cudaMemcpyAsync(b_d+idx_first_element, b_host+idx_first_element, 
                            bytes_seg, cudaMemcpyHostToDevice, streams[stream_idx]));
                vectorAddKernel<<<gridSize, blockSize, 0, streams[stream_idx]>>>
                                (a_d+idx_first_element, b_d+idx_first_element, 
                                c_d+idx_first_element, seg_length);
                CUDA_CHECK(cudaMemcpyAsync(c_dtohost+idx_first_element, c_d+idx_first_element, 
                            bytes_seg, cudaMemcpyDeviceToHost, streams[stream_idx]));
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());
            }   
            auto end = clock::now();
            const timingType time_gpu_streams = end - start;
            std::cout << "Execution time GPU streams: "
            << time_gpu_streams.count()/REPETITIONS << " ms" << std::endl;
            double max_error = 0.0;
            bool check = check_result(c_dtohost,c_host, vec_length, &max_error);
            if (check) {
              //  std::cout << "Check PASSED max error " << max_error << "\n"<< std::endl;
            } else {
                std::cout << "Check FAILED max error " << max_error << "\n"<< std::endl;
            }
        }
        for(int i=0; i<NUM_CUDASTREAMS;++i){
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
        }
    }
    #endif

        

        // 9. Free host memory
        CUDA_CHECK(cudaFreeHost(a_host));
        CUDA_CHECK(cudaFreeHost(b_host));
        CUDA_CHECK(cudaFreeHost(c_dtohost));
        delete[] c_host;

        // 10. Free device memory
        CUDA_CHECK(cudaFree(a_d));
        CUDA_CHECK(cudaFree(b_d));
        CUDA_CHECK(cudaFree(c_d));
    }

    return 0;
}