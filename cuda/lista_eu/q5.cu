#include <stdio.h>

#define N (2048 * 2048)
#define THREADS_PER_BLOCK 512

// Somando 2 vetores usando threads e blocos

__global__ void add(int *a, int *b, int *c)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    c[index] = a[index] + b[index];
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
    random_ints(a, N);
    b = (int *)malloc(size);
    random_ints(b, N);
    c = (int *)malloc(size);
    
    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
    // Launch add() kernel on GPU
    add<<<N / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
    
    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    
    // display the results
    for (int i=0; i<N; i++) {
        printf( "%d + %d = %d\n", a[i], b[i], c[i] );
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
