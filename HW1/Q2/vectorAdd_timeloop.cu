#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define TPB 32

// check the erros
#define CHECK(call) do {                                 \
    cudaError_t err = (call);                            \
    if (err != cudaSuccess) {                            \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n",      \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1);                                         \
    }                                                    \
} while (0)


double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void vectorAdd(float *v_out, const float *v_in1, const float *v_in2, int len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        v_out[i] = v_in1[i] + v_in2[i];
        // printf("i = %2d: %f + %f = %f\n", i, v_in1[i], v_in2[i], v_out[i]);
    }
}

int main()
{
    double iStart, iElaps1, iElaps2, iElaps3;

    // 2^6 to 2^20
    for (int N = 64; N <= 1048576; N *= 2)
    {
        printf("\n========== N = %d ==========\n", N);

        // @@ 1. Allocate in host memory.
        float *h1, *h2, *h3, *h3_cpu;

        // regular malloc for host memory or cudaHostAlloc with proper flags for the pinned version
        // h1 = (float*)malloc(N * sizeof(float));
        // h2 = (float*)malloc(N * sizeof(float));
        // h3 = (float*)malloc(N * sizeof(float));
        // h3_cpu = (float*)malloc(N * sizeof(float));

        // pinned
        cudaHostAlloc(&h1, N * sizeof(float), cudaHostAllocDefault);
        cudaHostAlloc(&h2, N * sizeof(float), cudaHostAllocDefault);
        cudaHostAlloc(&h3, N * sizeof(float), cudaHostAllocDefault);
        cudaHostAlloc(&h3_cpu, N * sizeof(float), cudaHostAllocDefault);

        // @@ 2. Allocate in device memory.
        float *v1 = nullptr;
        float *v2 = nullptr;
        float *v3 = nullptr;
        cudaMalloc(&v1, N * sizeof(float));
        cudaMalloc(&v2, N * sizeof(float));
        cudaMalloc(&v3, N * sizeof(float));

        // @@ 3. Initialize host memory.
        for (int i = 0; i < N; i++) 
        {
            h1[i] = 10.0f;
            h2[i] = -2.0f;
        }

        // @@ 4. Copy from host memory to device memory.
        iStart = cpuSecond();
        cudaMemcpy(v1, h1, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(v2, h2, N * sizeof(float), cudaMemcpyHostToDevice);
        iElaps1 = cpuSecond() - iStart;

        // @@ 5. Initialize thread block and thread grid
        dim3 grid((N+TPB-1)/TPB, 1, 1);
        dim3 tpb(TPB, 1, 1);

        // @@ 6. Invoke the CUDA Kernel.
        iStart = cpuSecond();
        vectorAdd<<<grid,tpb>>>(v3, v1, v2, N); // tpb: threads per block, grid: blocks
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        iElaps2 = cpuSecond() - iStart;

        // @@ 7. Copy results from GPU to CPU 
        iStart = cpuSecond();
        cudaMemcpy(h3, v3, N * sizeof(float), cudaMemcpyDeviceToHost);
        iElaps3 = cpuSecond() - iStart;

        printf("\nTimes: %f %f %f", iElaps1, iElaps2, iElaps3);

        // @@ 8. Compare the results with the CPU reference result
        for (int i = 0; i < N; i++) 
        {
            h3_cpu[i] = h1[i] + h2[i];
        }
        float maxError = 0.0f;
        for (int i = 0; i < N; i++) 
        {
            float err = fabs(h3[i] - h3_cpu[i]);
            if (err > maxError) {
                maxError = err;
            }
        }
        printf("\nMax absolute error = %f\n", maxError);

        // @@ 9. Free host memory.
        // free(h1);
        // free(h2);
        // free(h3);
        // free(h3_cpu);

        // if using cudaHostAlloc
        cudaFreeHost(h1);
        cudaFreeHost(h2);
        cudaFreeHost(h3);
        cudaFreeHost(h3_cpu);

        // @@ 10. Free device memory.
        cudaFree(v1);
        cudaFree(v2);
        cudaFree(v3);
    }

    return 0;
}
