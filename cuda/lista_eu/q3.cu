#include <stdlib.h>
#include <stdio.h>
#include <time.h>
// Somando 2 vetores usando n blocos e 1 thread em cada bloco

__global__ void add(int *a, int *b, int *c)
{
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

// Function to fill array with random integers
void random_ints(int* arr, int n)
{
    srand(time(NULL));
    for(int i = 0; i < n; i++)
    {
        arr[i] = rand() % 100;  // Random integers 0-99
    }
}

#define N 16
int main(void)
{
    int *a, *b, *c;       // host copies of a, b, c
    int *d_a, *d_b, *d_c; // device copies of a, b, c
    int size = N * sizeof(int);

    // Alloc space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Alloc space for host copies of a, b, c and setup input values
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);
    random_ints(a, N);
    random_ints(b, N);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
    // Launch add() kernel on GPU with N blocks
    add<<<N, 1>>>(d_a, d_b, d_c);
    
    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N; i++)
    {
        printf("%d\n",c[i]);
    }
    
    // Cleanup
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}