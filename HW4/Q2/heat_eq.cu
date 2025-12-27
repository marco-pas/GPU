#include <cuda_runtime_api.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>

// Preprocessor macro to enable/disable prefetching
#define ENABLE_PREFETCH 1

#define gpuCheck(stmt)                                               \
  do {                                                               \
      cudaError_t err = stmt;                                        \
      if (err != cudaSuccess) {                                      \
          printf("ERROR. Failed to run stmt %s\n", #stmt);           \
          break;                                                     \
      }                                                              \
  } while (0)

// Macro to check the cuBLAS status
#define cublasCheck(stmt)                                            \
  do {                                                               \
      cublasStatus_t err = stmt;                                     \
      if (err != CUBLAS_STATUS_SUCCESS) {                            \
          printf("ERROR. Failed to run cuBLAS stmt %s\n", #stmt);    \
          break;                                                     \
      }                                                              \
  } while (0)

// Macro to check the cuSPARSE status
#define cusparseCheck(stmt)                                          \
  do {                                                               \
      cusparseStatus_t err = stmt;                                   \
      if (err != CUSPARSE_STATUS_SUCCESS) {                          \
          printf("ERROR. Failed to run cuSPARSE stmt %s\n", #stmt);  \
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

// Initialize the sparse matrix needed for the heat time step
void matrixInit(double* A, int* ArowPtr, int* AcolIndx, int dimX,
    double alpha) {
  // Stencil from the finete difference discretization of the equation
  double stencil[] = { 1, -2, 1 };
  // Variable holding the position to insert a new element
  size_t ptr = 0;
  // Insert a row of zeros at the beginning of the matrix
  ArowPtr[0] = 0;  // Initialize first row pointer (boundary row is empty)
  ArowPtr[1] = ptr;
  // Fill the non zero entries of the matrix
  for (int i = 1; i < (dimX - 1); ++i) {
    // Insert the elements: A[i][i-1], A[i][i], A[i][i+1]
    for (int k = 0; k < 3; ++k) {
      // Set the value for A[i][i+k-1]
      A[ptr] = stencil[k];
      // Set the column index for A[i][i+k-1]
      AcolIndx[ptr++] = i + k - 1;
    }
    // Set the number of newly added elements
    ArowPtr[i + 1] = ptr;
  }
  // Insert a row of zeros at the end of the matrix
  ArowPtr[dimX] = ptr;
}

int main(int argc, char **argv) {
  int device = 0;            // Device to be used
  int dimX;                  // Dimension of the metal rod
  int nsteps;                // Number of time steps to perform
  double alpha = 0.4;        // Diffusion coefficient
  double* temp;              // Array to store the final time step
  double* A;                 // Sparse matrix A values in the CSR format
  int* ARowPtr;              // Sparse matrix A row pointers in the CSR format
  int* AColIndx;             // Sparse matrix A col values in the CSR format
  int nzv;                   // Number of non zero values in the sparse matrix
  double* tmp;               // Temporal array of dimX for computations
  size_t bufferSize = 0;     // Buffer size needed by some routines
  void* buffer = nullptr;    // Buffer used by some routines in the libraries
  int concurrentAccessQ;     // Check if concurrent access flag is set
  double zero = 0;           // Zero constant
  double one = 1;            // One constant
  double norm;               // Variable for norm values
  double error;              // Variable for storing the relative error
  double tempLeft = 200.;    // Left heat source applied to the rod
  double tempRight = 300.;   // Right heat source applied to the rod
  cublasHandle_t cublasHandle;      // cuBLAS handle
  cusparseHandle_t cusparseHandle;  // cuSPARSE handle
  cusparseSpMatDescr_t matA;        // Sparse matrix descriptor
  cusparseDnVecDescr_t vecTemp;     // Dense vector descriptor for temp
  cusparseDnVecDescr_t vecTmp;      // Dense vector descriptor for tmp

  cudaMemLocation locHost{};
  locHost.type = cudaMemLocationTypeHost;
  cudaMemLocation locDev{};
  locDev.type = cudaMemLocationTypeDevice;
  locDev.id = device;

  // Read the arguments from the command line
  dimX = atoi(argv[1]);
  nsteps = atoi(argv[2]);

  // Print input arguments
  printf("The X dimension of the grid is %d \n", dimX);
  printf("The number of time steps to perform is %d \n", nsteps);

  // Get if the cudaDevAttrConcurrentManagedAccess flag is set
  gpuCheck(cudaDeviceGetAttribute(&concurrentAccessQ, cudaDevAttrConcurrentManagedAccess, device));

  // Calculate the number of non zero values in the sparse matrix. This number
  // is known from the structure of the sparse matrix
  nzv = 3 * dimX - 6;

  //@@ Insert the code to allocate the temp, tmp and the sparse matrix
  //@@ arrays using Unified Memory
  cputimer_start();
  gpuCheck(cudaMallocManaged(&temp, sizeof(double) * dimX));
  gpuCheck(cudaMallocManaged(&tmp, sizeof(double) * dimX));
  gpuCheck(cudaMallocManaged(&A, sizeof(double) * nzv));
  gpuCheck(cudaMallocManaged(&ARowPtr, sizeof(int) * (dimX + 1)));
  gpuCheck(cudaMallocManaged(&AColIndx, sizeof(int) * nzv));
  cputimer_stop("Allocating device memory");

  // Start total execution timer (after allocation, includes prefetch + init + loop)
  struct timeval t_total_start, t_total_end;
  gettimeofday(&t_total_start, 0);

  // Check if concurrentAccessQ is non zero in order to prefetch memory
#if ENABLE_PREFETCH
  if (concurrentAccessQ) {
    cputimer_start();
    //@@ Insert code to prefetch in Unified Memory asynchronously to CPU
    gpuCheck(cudaMemPrefetchAsync(temp, sizeof(double) * dimX, locHost, 0));
    gpuCheck(cudaMemPrefetchAsync(tmp, sizeof(double) * dimX, locHost, 0));
    gpuCheck(cudaMemPrefetchAsync(A, sizeof(double) * nzv, locHost, 0));
    gpuCheck(cudaMemPrefetchAsync(ARowPtr, sizeof(int) * (dimX + 1), locHost, 0));
    gpuCheck(cudaMemPrefetchAsync(AColIndx, sizeof(int) * nzv, locHost, 0));
    cputimer_stop("Prefetching GPU memory to the host");
  }
#else
  printf("Prefetching DISABLED\n");
#endif

  // Initialize the sparse matrix
  cputimer_start();
  matrixInit(A, ARowPtr, AColIndx, dimX, alpha);
  cputimer_stop("Initializing the sparse matrix on the host");

  //Initiliaze the boundary conditions for the heat equation
  cputimer_start();
  memset(temp, 0, sizeof(double) * dimX);
  temp[0] = tempLeft;
  temp[dimX - 1] = tempRight;
  cputimer_stop("Initializing memory on the host");

#if ENABLE_PREFETCH
  if (concurrentAccessQ) {
    cputimer_start();
    //@@ Insert code to prefetch in Unified Memory asynchronously to the GPU
    gpuCheck(cudaMemPrefetchAsync(temp, sizeof(double) * dimX, locDev, 0));
    gpuCheck(cudaMemPrefetchAsync(tmp, sizeof(double) * dimX, locDev, 0));
    gpuCheck(cudaMemPrefetchAsync(A, sizeof(double) * nzv, locDev, 0));
    gpuCheck(cudaMemPrefetchAsync(ARowPtr, sizeof(int) * (dimX + 1), locDev, 0));
    gpuCheck(cudaMemPrefetchAsync(AColIndx, sizeof(int) * nzv, locDev, 0));
    cputimer_stop("Prefetching GPU memory to the device");
  }
#endif

  cputimer_start();
  //@@ Insert code to create the cuBLAS handle
  cublasCheck(cublasCreate(&cublasHandle));

  //@@ Insert code to create the cuSPARSE handle
  cusparseCheck(cusparseCreate(&cusparseHandle));

  //@@ Insert code to set the cuBLAS pointer mode to CUSPARSE_POINTER_MODE_HOST
  cublasCheck(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST));

  //@@ Insert code to call cusparse api to create the mat descriptor used by cuSPARSE
  // Create sparse matrix A in CSR format (new generic API)
  cusparseCheck(cusparseCreateCsr(&matA, dimX, dimX, nzv,
      ARowPtr, AColIndx, A,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

  // Create dense vector descriptors
  cusparseCheck(cusparseCreateDnVec(&vecTemp, dimX, temp, CUDA_R_64F));
  cusparseCheck(cusparseCreateDnVec(&vecTmp, dimX, tmp, CUDA_R_64F));

  //@@ Insert code to call cusparse api to get the buffer size needed by the sparse matrix per
  //@@ vector (SMPV) CSR routine of cuSPARSE
  cusparseCheck(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &one, matA, vecTemp, &zero, vecTmp, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));

  //@@ Insert code to allocate the buffer needed by cuSPARSE
  gpuCheck(cudaMalloc(&buffer, bufferSize));
  gpuCheck(cudaDeviceSynchronize());
  cputimer_stop("Initializing cuBLAS and cuSPARSE handles and descriptors");

  // Perform the time step iterations
  int actualIterations = 0;
  double iterTime_us = 0.0;
  struct timeval t_loop_start, t_loop_end;
  gettimeofday(&t_loop_start, 0);  // Start full loop timer
  for (int it = 0; it < nsteps; ++it) {
    //@@ Insert code to call cusparse api to compute the SMPV (sparse matrix multiplication) for
    //@@ the CSR matrix using cuSPARSE. This calculation corresponds to:
    //@@ tmp = 1 * A * temp + 0 * tmp
    cputimer_start();
    cusparseCheck(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &one, matA, vecTemp, &zero, vecTmp, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));
    gpuCheck(cudaDeviceSynchronize());  // Wait for SpMV to complete before timing
    gettimeofday(&t_end, 0);
    iterTime_us += (1000000.0*(t_end.tv_sec-t_start.tv_sec) + t_end.tv_usec-t_start.tv_usec);
    
    //@@ Insert code to call cublas api to compute the axpy routine using cuBLAS.
    //@@ This calculation corresponds to: temp = alpha * tmp + temp
    cublasCheck(cublasDaxpy(cublasHandle, dimX, &alpha, tmp, 1, temp, 1));

    //@@ Insert code to call cublas api to compute the norm of the vector using cuBLAS
    //@@ This calculation corresponds to: ||temp||
    cublasCheck(cublasDnrm2(cublasHandle, dimX, tmp, 1, &norm));

    actualIterations++;
    // If the norm of A*temp is smaller than 10^-4 exit the loop
    if (norm < 1e-4)
      break;
  }
  gpuCheck(cudaDeviceSynchronize());
  gettimeofday(&t_loop_end, 0);  // End full loop timer
  gettimeofday(&t_total_end, 0);  // End total execution timer
  double totalTime_us = (1000000.0*(t_loop_end.tv_sec-t_loop_start.tv_sec) + t_loop_end.tv_usec-t_loop_start.tv_usec);
  double totalExecTime_us = (1000000.0*(t_total_end.tv_sec-t_total_start.tv_sec) + t_total_end.tv_usec-t_total_start.tv_usec);

  // FLOPS calculation for SpMV: 2 FLOPs per non-zero (1 multiply + 1 add)
  // nzv = 3 * dimX - 6 for tridiagonal matrix
  double flopsPerSpMV = 2.0 * nzv;
  double totalFlops = flopsPerSpMV * actualIterations;
  double gflops = totalFlops / (iterTime_us) / 1e3;  // GFLOP/s

  printf("Iterations: %d, SpMV Time: %.2f ms, Loop Time: %.2f ms, Total Exec Time: %.2f ms, SpMV GFLOP/s: %.4f\n",
         actualIterations, iterTime_us / 1000.0, totalTime_us / 1000.0, totalExecTime_us / 1000.0, gflops);

  // Calculate the exact solution using thrust
  thrust::device_ptr<double> thrustPtr(tmp);
  thrust::sequence(thrustPtr, thrustPtr + dimX, tempLeft,
      (tempRight - tempLeft) / (dimX - 1));

  // Calculate the relative approximation error:
  one = -1;
  //@@ Insert the code to call cublas api to compute the difference between the exact solution
  //@@ and the approximation
  //@@ This calculation corresponds to: tmp = -temp + tmp
  cublasCheck(cublasDaxpy(cublasHandle, dimX, &one, temp, 1, tmp, 1));

  //@@ Insert the code to call cublas api to compute the norm of the absolute error
  //@@ This calculation corresponds to: || tmp ||
  cublasCheck(cublasDnrm2(cublasHandle, dimX, tmp, 1, &norm));

  error = norm;
  //@@ Insert the code to call cublas api to compute the norm of temp
  //@@ This calculation corresponds to: || temp ||
  cublasCheck(cublasDnrm2(cublasHandle, dimX, temp, 1, &norm));

  // Calculate the relative error
  error = error / norm;
  printf("The relative error of the approximation is %f\n", error);

  //@@ Insert the code to destroy the mat descriptor
  cusparseCheck(cusparseDestroySpMat(matA));
  cusparseCheck(cusparseDestroyDnVec(vecTemp));
  cusparseCheck(cusparseDestroyDnVec(vecTmp));

  //@@ Insert the code to destroy the cuSPARSE handle
  cusparseCheck(cusparseDestroy(cusparseHandle));

  //@@ Insert the code to destroy the cuBLAS handle
  cublasCheck(cublasDestroy(cublasHandle));

  //@@ Insert the code for deallocating memory
  gpuCheck(cudaFree(temp));
  gpuCheck(cudaFree(tmp));
  gpuCheck(cudaFree(A));
  gpuCheck(cudaFree(ARowPtr));
  gpuCheck(cudaFree(AColIndx));
  gpuCheck(cudaFree(buffer));

  return 0;
}