#include <cuda_runtime_api.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>

#define gpuCheck(stmt)                                               \
  do {                                                               \
      cudaError_t err = stmt;                                        \
      if (err != cudaSuccess) {                                      \
          printf("ERROR. Failed to run stmt %s\n", #stmt);           \
          break;                                                     \
      }                                                              \
  } while (0)

struct timeval t_start, t_end;
void cputimer_start(){
  gettimeofday(&t_start, 0);
}

void cputimer_stop(const char* info){
  gettimeofday(&t_end, 0);
  double time = (1000000.0*(t_end.tv_sec-t_start.tv_sec) + t_end.tv_usec-t_start.tv_usec);
  printf("Timing - %s. \t\tElasped %.0f microseconds \n", info, time);
}

#define MASK_WIDTH 5
#define TILE_WIDTH 256 //@@ INSERT CODE HERE


__global__ void convolution_1D_basic(float *N, float *M, float *P, int array_len)
{
    //@@ INSERT CODE HERE
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < array_len) {
        float pValue = 0.0f;
        int start = idx - MASK_WIDTH / 2;
        
        for (int j = 0; j < MASK_WIDTH; j++) {
            int nIdx = start + j;
            // Handle boundary conditions (out-of-boundary elements are 0)
            if (nIdx >= 0 && nIdx < array_len) {
                pValue += N[nIdx] * M[j];
            }
        }
        P[idx] = pValue;
    }
}

__global__ void convolution_1D_tiled(float *N, float *M, float *P, int array_len)
{
  //@@ INSERT CODE HERE
  __shared__ float input_tile[TILE_WIDTH + MASK_WIDTH - 1];
  
  //@@ INSERT CODE HERE
  // Load mask into registers (only 5 reads per thread, cached after first warp)
  float m0 = M[0];
  float m1 = M[1];
  float m2 = M[2];
  float m3 = M[3];
  float m4 = M[4];
  
  int tx = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tx;
  int halo_radius = MASK_WIDTH / 2;
  
  // Calculate the starting index for this tile in global memory
  int halo_start = (blockIdx.x * blockDim.x) - halo_radius;
  
  // Load left halo elements (first halo_radius threads load left halo)
  if (tx < halo_radius) {
    int halo_idx = halo_start + tx;
    input_tile[tx] = (halo_idx >= 0) ? N[halo_idx] : 0.0f;
  }
  
  // Load main tile elements
  int main_idx = halo_start + halo_radius + tx;
  input_tile[halo_radius + tx] = (main_idx >= 0 && main_idx < array_len) ? N[main_idx] : 0.0f;
  
  // Load right halo elements (last halo_radius threads load right halo)
  if (tx < halo_radius) {
    int halo_idx = halo_start + halo_radius + TILE_WIDTH + tx;
    input_tile[halo_radius + TILE_WIDTH + tx] = (halo_idx < array_len) ? N[halo_idx] : 0.0f;
  }
  
  __syncthreads();
  
  // Compute convolution using registers for mask (no global memory reads in loop)
  if (idx < array_len) {
    float pValue = input_tile[tx]     * m0 +
                   input_tile[tx + 1] * m1 +
                   input_tile[tx + 2] * m2 +
                   input_tile[tx + 3] * m3 +
                   input_tile[tx + 4] * m4;
    P[idx] = pValue;
  }
}

int main(int argc, char *argv[]) {
  
  // Check for command line arguments
  if (argc < 2) {
    printf("Usage: %s <array_size>\n", argv[0]);
    printf("Example: %s 10000\n", argv[0]);
    return 1;
  }
  
  // Read the arguments from the command line
  int N = atoi(argv[1]);


  float *hostN; // The input array N of length N
  float *hostM; // The 1D mask M of length MASK_WIDTH
  float *hostP; // The output array P of length N

  cputimer_start();
  //@@ Allocate the host memory
  hostN = (float *)malloc(N * sizeof(float));
  hostM = (float *)malloc(MASK_WIDTH * sizeof(float));
  hostP = (float *)malloc(N * sizeof(float));
  cputimer_stop("Allocated host memory");


  float *deviceN;
  float *deviceM;
  float *deviceP;

  cputimer_start();
  //@@ Allocate the device memory
  gpuCheck(cudaMalloc((void **)&deviceN, N * sizeof(float)));
  gpuCheck(cudaMalloc((void **)&deviceM, MASK_WIDTH * sizeof(float)));
  gpuCheck(cudaMalloc((void **)&deviceP, N * sizeof(float)));
  cputimer_stop("Allocated device memory");

  
  cputimer_start();
  //@@ Initialize N with random values
  srand(42); // Seed for reproducibility
  for (int i = 0; i < N; i++) {
    hostN[i] = (float)(rand() % 100) / 100.0f;
  }
  //@@ Initialize M with [-0.25, 0.5, 1.0, 0.5, 0.25]
  hostM[0] = -0.25f;
  hostM[1] = 0.5f;
  hostM[2] = 1.0f;
  hostM[3] = 0.5f;
  hostM[4] = -0.25f;
  //@@ Initialize P with 0.0
  for (int i = 0; i < N; i++) {
    hostP[i] = 0.0f;
  }
  cputimer_stop("Initialized arrays");

  
  cputimer_start();
  //@@ INSERT CODE HERE
  gpuCheck(cudaMemcpy(deviceN, hostN, N * sizeof(float), cudaMemcpyHostToDevice));
  gpuCheck(cudaMemcpy(deviceM, hostM, MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice));
  cputimer_stop("Copying data to the GPU.");
  

  /* Call the basic kernel */
  cputimer_start();
  //@@  Define the execution configuration
  int blockSize = TILE_WIDTH;
  int gridSize = (N + TILE_WIDTH - 1) / TILE_WIDTH;
  //@@  Run the 1D convolution kernel (basic)
  convolution_1D_basic<<<gridSize, blockSize>>>(deviceN, deviceM, deviceP, N);
  gpuCheck(cudaDeviceSynchronize());
  cputimer_stop("Finished 1D convolution(basic)");
  
  // Store basic kernel results for validation
  float *hostP_basic = (float *)malloc(N * sizeof(float));
  
  cputimer_start();
  //@@ INSERT CODE HERE
  gpuCheck(cudaMemcpy(hostP_basic, deviceP, N * sizeof(float), cudaMemcpyDeviceToHost));
  cputimer_stop("Copying output P to the CPU and print out the results");
  
  // Print first 10 results from basic kernel
  /**
  printf("Basic kernel results (first 10 elements):\n");
  for (int i = 0; i < 10 && i < N; i++) {
    printf("P[%d] = %.4f\n", i, hostP_basic[i]);
  }
  */

  /* Call the tiled kernel */
  // Reset output array on device
  gpuCheck(cudaMemset(deviceP, 0, N * sizeof(float)));
  
  cputimer_start();
  //@@  Define the execution configuration
  //@@  Run the 1D convolution kernel (tiled)
  convolution_1D_tiled<<<gridSize, blockSize>>>(deviceN, deviceM, deviceP, N);
  gpuCheck(cudaDeviceSynchronize());
  cputimer_stop("Finished 1D convolution(tiled)");
  
  cputimer_start();
  //@@ INSERT CODE HERE
  gpuCheck(cudaMemcpy(hostP, deviceP, N * sizeof(float), cudaMemcpyDeviceToHost));
  cputimer_stop("Copying output P to the CPU and print out the results");
  
  // Print first 10 results from tiled kernel
  /**
  printf("\nTiled kernel results (first 10 elements):\n");
  for (int i = 0; i < 10 && i < N; i++) {
    printf("P[%d] = %.4f\n", i, hostP[i]);
  }
  */

  //@@ Validate the results from the two implementations
  printf("\nValidating results...\n");
  float maxError = 1e-10;
  int errorCount = 0;
  for (int i = 0; i < N; i++) {
    float error = fabs(hostP_basic[i] - hostP[i]);
    if (error > maxError) maxError = error;
    if (error > 1e-5) errorCount++;
  }
  printf("Max error between basic and tiled: %.6f\n", maxError);
  printf("Number of mismatches (error > 1e-5): %d\n", errorCount);
  if (errorCount == 0) {
    printf("VALIDATION PASSED!\n");
  } else {
    printf("VALIDATION FAILED!\n");
  }


  cputimer_start();
  //@@ INSERT CODE HERE
  gpuCheck(cudaFree(deviceN));
  gpuCheck(cudaFree(deviceM));
  gpuCheck(cudaFree(deviceP));
  free(hostN);
  free(hostM);
  free(hostP);
  free(hostP_basic);
  cputimer_stop("Free memory resources");

  return 0;
}
