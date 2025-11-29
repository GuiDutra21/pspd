#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess)
  {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__global__ void add(int N, float *a, float *b, float *c)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x; // handles the data at its thread id

  printf("gridDim.x=%d, blockDim.x=%d, blockIdx.x=%d, threadIdx.x=%d => %d\n", gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, tid);

  while (tid < N)
  {
    c[tid] = a[tid] + b[tid];
    printf("tid = %d\n", tid);
    tid += blockDim.x * gridDim.x;
  }
}

int main(int argc, char *argv[])
{
  if (argc != 5)
  {
    fprintf(stderr, "Syntax: %s <vector size N> <grid size> <block size> <device id>\n", argv[0]);
    return EXIT_FAILURE;
  }

  int N = atoi(argv[1]);
  int GridSize = atoi(argv[2]);
  int BlockSize = atoi(argv[3]);
  int devId = atoi(argv[4]);

  checkCuda(cudaSetDevice(devId));

  float *a, *b, *c;
  // allocate the memory on the CPU
  a = (float *)malloc(N * sizeof(float));
  b = (float *)malloc(N * sizeof(float));
  c = (float *)malloc(N * sizeof(float));

  float *dev_a, *dev_b, *dev_c;
  // allocate the memory on the GPU
  checkCuda(cudaMalloc((void **)&dev_a, N * sizeof(float)));
  checkCuda(cudaMalloc((void **)&dev_b, N * sizeof(float)));
  checkCuda(cudaMalloc((void **)&dev_c, N * sizeof(float)));

  // fill the arrays 'a' and 'b' on the CPU
  for (int i = 0; i < N; i++)
  {
    a[i] = -i;
    b[i] = i * i;
  }

  // copy the arrays 'a' and 'b' to the GPU
  checkCuda(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice));

  add<<<GridSize, BlockSize>>>(N, dev_a, dev_b, dev_c);

  // copy the array 'c' back from the GPU to the CPU
  checkCuda(cudaMemcpy(c, dev_c, N * sizeof(float), cudaMemcpyDeviceToHost));

  // display the results
  /*for (int i=0; i<N; i++) {
      printf( "%6.0f + %6.0f = %6.0f\n", a[i], b[i], c[i] );
  }*/

  cudaDeviceProp prop;
  checkCuda(cudaGetDeviceProperties(&prop, devId));
  printf("Device: %s\n", prop.name);

  // free the memory allocated on the CPU
  free(a);
  a = NULL;
  free(b);
  b = NULL;
  free(c);
  c = NULL;

  // free the memory allocated on the GPU
  checkCuda(cudaFree(dev_a));
  checkCuda(cudaFree(dev_b));
  checkCuda(cudaFree(dev_c));

  return 0;
}
