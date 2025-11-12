#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 64
#define TPB 32

// check errors
#define CHECK(call) do {                                 \
    cudaError_t err = (call);                            \
    if (err != cudaSuccess) {                            \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n",      \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1);                                         \
    }                                                    \
} while (0)

__global__ void vectorAdd(float *v_out, const float *v_in1, const float *v_in2, int len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        v_out[i] = v_in1[i] + v_in2[i];
        printf("i = %2d: %f + %f = %f\n", i, v_in1[i], v_in2[i], v_out[i]);
    }
}

int main()
{
    // @@ 1. Allocate in host memory.
    float *h1 = (float*)malloc(N * sizeof(float));
    float *h2 = (float*)malloc(N * sizeof(float));
    float *h3 = (float*)malloc(N * sizeof(float));
    float *h3_cpu = (float*)malloc(N * sizeof(float));

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
    cudaMemcpy(v1, h1, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(v2, h2, N * sizeof(float), cudaMemcpyHostToDevice);

    // @@ 5. Initialize thread block and thread grid
    dim3 grid((N+TPB-1)/TPB, 1, 1);
    dim3 tpb(TPB, 1, 1);

    // @@ 6. Invoke the CUDA Kernel.
    vectorAdd<<<grid,tpb>>>(v3, v1, v2, N); // tpb: threads per block, grid: blocks
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // @@ 7. Copy results from GPU to CPU 
    cudaMemcpy(h3, v3, N * sizeof(float), cudaMemcpyDeviceToHost);

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
    free(h1);
    free(h2);
    free(h3);

    // @@ 10. Free device memory.
    cudaFree(v1);
    cudaFree(v2);
    cudaFree(v3);

    return 0;
}