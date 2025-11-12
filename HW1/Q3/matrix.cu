// MATRIX MULTIPLICATION

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 128  // numARows = numCRows
#define M 256  // numAColumns = numBRows
#define P 32   // numBColumns = numCColumns
#define TPB 16 // Threads per block dimension (16x16 = 256 threads)

// check errors
#define CHECK(call) do {                                 \
    cudaError_t err = (call);                            \
    if (err != cudaSuccess) {                            \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n",      \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1);                                         \
    }                                                    \
} while (0)

__global__ void matrixMult(float *C, const float *A, const float *B, int numARows, int numAColumns, int numBColumns)
{
    // Each thread computes one element C[row][col]
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds
    if (row < numARows && col < numBColumns) {
        float sum = 0.0f;
        
        // Compute dot product of row from A and column from B
        for (int k = 0; k < numAColumns; k++) {
            sum += A[row * numAColumns + k] * B[k * numBColumns + col];
        }
        
        C[row * numBColumns + col] = sum;
    }
}

int main()
{
    // @@ 1. Allocate in host memory.
    float *A = (float*)malloc(N * M * sizeof(float));
    float *B = (float*)malloc(M * P * sizeof(float));
    float *C = (float*)malloc(N * P * sizeof(float));

    // @@ 2. Allocate in device memory.
    float *A_gpu = nullptr;
    float *B_gpu = nullptr;
    float *C_gpu = nullptr;
    CHECK(cudaMalloc(&A_gpu, N * M * sizeof(float)));
    CHECK(cudaMalloc(&B_gpu, M * P * sizeof(float)));
    CHECK(cudaMalloc(&C_gpu, N * P * sizeof(float)));

    // @@ 3. Initialize host memory.
    for (int i = 0; i < (N * M); i++) 
    {
        A[i] = 1.0f;
    }

    for (int i = 0; i < (M * P); i++) 
    {
        B[i] = 1.0f;
    }

    // @@ 4. Copy from host memory to device memory.
    CHECK(cudaMemcpy(A_gpu, A, N * M * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_gpu, B, M * P * sizeof(float), cudaMemcpyHostToDevice));

    // @@ 5. Initialize thread block and thread grid
    // 2D grid for output matrix C (N x P)
    dim3 tpb(TPB, TPB, 1);  // 16x16 threads per block
    dim3 grid((P + TPB - 1) / TPB, (N + TPB - 1) / TPB, 1);

    // @@ 6. Invoke the CUDA Kernel.
    matrixMult<<<grid, tpb>>>(C_gpu, A_gpu, B_gpu, N, M, P);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // @@ 7. Copy results from GPU to CPU 
    CHECK(cudaMemcpy(C, C_gpu, N * P * sizeof(float), cudaMemcpyDeviceToHost));

    // @@ 8. Compare the results with the CPU reference result
    // leave empty

    // @@ 9. Free host memory.
    free(A);
    free(B);
    free(C);

    // @@ 10. Free device memory.
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);

    printf(" ----- Completed successfully. ----- ");
    return 0;
}