#include <stdio.h>
#include <stdlib.h>

#define X 600    // horizontal n
#define Y 899    // vertical m
#define TPBX 64
#define TPBY 16

// check errors
#define CHECK(call) do {                                 \
    cudaError_t err = (call);                            \
    if (err != cudaSuccess) {                            \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n",      \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1);                                         \
    }                                                    \
} while (0)

__global__ void divCounter(int n, int m, int* warpDiv) {

    int Row = blockIdx.y*blockDim.y + threadIdx.y;
    int Col = blockIdx.x*blockDim.x + threadIdx.x;

    bool pred = (Row < m) && (Col < n);

    unsigned mask = 0xFFFFFFFF;   // FULL WARP MASK

    unsigned ballot = __ballot_sync(mask, pred);

    // divergence: some threads true, some false
    if (ballot != 0 && ballot != 0xFFFFFFFF) {
        // only the lane 0 increments
        if ((threadIdx.x & 0x1F) == 0)
            atomicAdd(warpDiv, 1);
    }
}



__global__ void PictureKernel(float* d_Pin, float* d_Pout, int n, int m) { 

   // Calculate the row # of the d_Pin and d_Pout element to process

 int Row = blockIdx.y*blockDim.y + threadIdx.y;

 // Calculate the column # of the d_Pin and d_Pout element to process 

 int Col = blockIdx.x*blockDim.x + threadIdx.x;

 // each thread computes one element of d_Pout if in range 

 if ((Row < m) && (Col < n)) { // this is where the divetence gets created

    // the conditional depends on threadIdx so divergence will be created!
    d_Pout[Row*n+Col] = 2*d_Pin[Row*n+Col];
    }

}

int main()
{
    // CPU memory
    float *h_in  = (float*)malloc(X * Y * sizeof(float));
    float *h_out = (float*)malloc(X * Y * sizeof(float));

    int h_warpDiv = 0;        // host counter
    int* d_warpDiv = nullptr; // device counter
    cudaMalloc(&d_warpDiv, sizeof(int));
    cudaMemset(d_warpDiv, 0, sizeof(int));

    // GPU memory
    float *d_in = nullptr;
    float *d_out = nullptr;

    cudaMalloc(&d_in,  X * Y * sizeof(float));
    cudaMalloc(&d_out, X * Y * sizeof(float));

    // initialize input picture on CPU
    for (int i = 0; i < X*Y; i++) {
        h_in[i] = (float)i;
    }

    // copy input
    cudaMemcpy(d_in, h_in, X*Y*sizeof(float), cudaMemcpyHostToDevice);

    // 2D block and grid
    dim3 block(TPBX, TPBY);
    dim3 grid((X + TPBX - 1)/TPBX, (Y + TPBY - 1)/TPBY);

    // launch real kernel
    PictureKernel<<<grid, block>>>(d_in, d_out, X, Y);
    CHECK(cudaGetLastError());  
    CHECK(cudaDeviceSynchronize());

    // launch divergence counter
    divCounter<<<grid, block>>>(X, Y, d_warpDiv);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    cudaMemcpy(&h_warpDiv, d_warpDiv, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Divergence count is %d\n", h_warpDiv);   

    // copy back
    cudaMemcpy(h_out, d_out, X*Y*sizeof(float), cudaMemcpyDeviceToHost);

    printf("----- Done! -----\n");

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_warpDiv);

    return 0;
}
