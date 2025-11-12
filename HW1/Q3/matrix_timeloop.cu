#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

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

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void matrixMult(float *C, const float *A, const float *B, int numARows, int numAColumns, int numBColumns)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < numARows && col < numBColumns) {
        float sum = 0.0f;

        for (int k = 0; k < numAColumns; k++) {
            sum += A[row * numAColumns + k] * B[k * numBColumns + col];
        }
        
        C[row * numBColumns + col] = sum;
    }
}

int main()
{
    double iStart, iElaps1, iElaps2, iElaps3;

    // loop over matrix size
    for (int size_factor = 1; size_factor <= 15; size_factor++)
    {
        // increase dimensions
        int N = 32 * size_factor;  // numARows = numCRows
        int M = 64 * size_factor;  // numAColumns = numBRows  
        int P = 16 * size_factor;  // numBColumns = numCColumns
        
        printf("\n========== N = %d, M = %d, P = %d ==========\n", N, M, P);

        // @@ 1. Allocate in host memory.
        float *A = (float*)malloc(N * M * sizeof(float));
        float *B = (float*)malloc(M * P * sizeof(float));
        float *C = (float*)malloc(N * P * sizeof(float));
        float *C_cpu = (float*)malloc(N * P * sizeof(float)); // For verification

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
        iStart = cpuSecond();
        CHECK(cudaMemcpy(A_gpu, A, N * M * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(B_gpu, B, M * P * sizeof(float), cudaMemcpyHostToDevice));
        iElaps1 = cpuSecond() - iStart;

        // @@ 5. Initialize thread block and thread grid
        // 2D grid for output matrix C (N x P)
        dim3 tpb(TPB, TPB, 1);  // 16x16 threads per block
        dim3 grid((P + TPB - 1) / TPB, (N + TPB - 1) / TPB, 1);

        // @@ 6. Invoke the CUDA Kernel.
        iStart = cpuSecond();
        matrixMult<<<grid, tpb>>>(C_gpu, A_gpu, B_gpu, N, M, P);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        iElaps2 = cpuSecond() - iStart;

        // @@ 7. Copy results from GPU to CPU 
        iStart = cpuSecond();
        CHECK(cudaMemcpy(C, C_gpu, N * P * sizeof(float), cudaMemcpyDeviceToHost));
        iElaps3 = cpuSecond() - iStart;

        printf("Transfer to GPU: %f ms\n", iElaps1 * 1000);
        printf("Kernel execution: %f ms\n", iElaps2 * 1000);
        printf("Transfer from GPU: %f ms\n", iElaps3 * 1000);
        printf("Total GPU time: %f ms\n", (iElaps1 + iElaps2 + iElaps3) * 1000);

        // @@ 8. Compare the results with the CPU reference result
        // check the result to make sure
        int errors = 0;
        float expected = (float)M;
        for (int i = 0; i < N * P; i++) {
            if (fabs(C[i] - expected) > 1e-5) {
                errors++;
                if (errors <= 5) { // Print first 5 errors only
                    printf("Error at element %d: expected %f, got %f\n", i, expected, C[i]);
                }
            }
        }
        if (errors > 0) {
            printf("Total errors: %d\n", errors);
        } else {
            printf("Results verified successfully - all elements equal %d\n", M);
        }

        // @@ 9. Free host memory.
        free(A);
        free(B);
        free(C);
        free(C_cpu);

        // @@ 10. Free device memory.
        cudaFree(A_gpu);
        cudaFree(B_gpu);
        cudaFree(C_gpu);
    }

    printf("\n ----- Completed successfully. ----- \n");
    return 0;
}
