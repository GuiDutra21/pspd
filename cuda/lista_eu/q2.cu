// Somando dois elementos na gpu e imprimido na cpu

#include<stdio.h>

__global__ void add(int *a, int *b, int *c)
{
    *c = *a + *b;
}

int main()
{
    int a,b,c;
    int *d_a,*d_b,*d_c;
    int size = sizeof(int);
    
    // Alocando espaço para as cópias do device: a, b, c
    cudaMalloc((void**)&d_a,size);
    cudaMalloc((void**)&d_b,size);
    cudaMalloc((void**)&d_c,size);

    // Valores para o input
    a = 2;
    b = 7;

    // Copia os inputs para o device
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
    
    // Lança o kernel add() na GPU
    add<<<1,1>>>(d_a, d_b, d_c);
    
    // Copia os resultados de volta para a CPU host
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

    printf("%d\n",c);
    // Limpeza
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}