#include <stdio.h>

#define BLOCK_SIZE  8
#define N 20
__global__ void add(int *a, int *b, int *c)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < N)
    {
        c[index] = a[index] + b[index];
    } 
}

int main()
{
    int *a,*b,*c;
    int *dev_a,*dev_b,*dev_c;
    int size = N * sizeof(int);

    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    for(int i = 0; i < N;i++)
    {
        a[i] = i;
        b[i] = i + 1;
    }

    cudaMalloc((void**)&dev_a,size);
    cudaMalloc((void**)&dev_b,size);
    cudaMalloc((void**)&dev_c,size);

    cudaMemcpy(dev_a,a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b,b,size,cudaMemcpyHostToDevice);

    int numBlock = (N + BLOCK_SIZE -1) / BLOCK_SIZE;

    add<<<numBlock, BLOCK_SIZE>>>(dev_a,dev_b,dev_c);

    cudaDeviceSynchronize();

    cudaMemcpy(c,dev_c,size,cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; i++)
    {
       printf("c[%2d] = %d + %d = %d\n", i, a[i], b[i], c[i]);
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a);
    free(b);
    free(c);
}