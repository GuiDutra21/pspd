#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 4
#define RADIUS 2
#define N 16

__global__ void stencil_1d(int *in, int *out)
{
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + RADIUS;

    // Read input elements into shared memory
    temp[lindex] = in[gindex];
    if (threadIdx.x < RADIUS)
    {
        temp[lindex - RADIUS] = in[gindex - RADIUS];
        temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
    }
    // Synchronize (ensure all the data is available)
    __syncthreads();

    // Apply the stencil
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++)
        result += temp[lindex + offset];
    
    // Store the result
    out[gindex] = result;
}

__global__ void st_treino(int*in, int *out)
{
    int temp[BLOCK_SIZE + 2 * RADIUS];
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + RADIUS;

    temp[lindex] = in[gindex];
    if(threadIdx.x < RADIUS)
    {
        temp[lindex - RADIUS] = in[gindex - RADIUS];
        temp[lindex + BLOCK_SIZE] = in[gindex +  BLOCK_SIZE]; 
    }

    __syncthreads();

    int result = 0;
    for(int offset = -RADIUS; offset <= RADIUS; offset++)
    {
        result += temp[lindex + offset];
    }

    out[gindex] = result;
}

int main(void)
{
    int *h_in, *h_out;      // host arrays
    int *d_in, *d_out;      // device arrays
    int size = N * sizeof(int);
    
    // Alocar memória na CPU
    h_in = (int *)malloc(size);
    h_out = (int *)malloc(size);
    
    // Inicializar vetor de entrada com valores simples
    printf("Vetor de entrada:\n");
    for (int i = 0; i < N; i++) {
        h_in[i] = i;  // Valores: 0, 1, 2, 3, 4, ..., 16
        printf("%d ", h_in[i]);
    }
    printf("\n\n");
    
    // Alocar memória na GPU
    cudaMalloc((void **)&d_in, size);
    cudaMalloc((void **)&d_out, size);
    
    // Copiar dados para GPU
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    
    // Calcular configuração do kernel
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    printf("Configuração:\n");
    printf("- N = %d elementos\n", N);
    printf("- BLOCK_SIZE = %d threads/bloco\n", BLOCK_SIZE);
    printf("- RADIUS = %d\n", RADIUS);
    printf("- Número de blocos = %d\n\n", numBlocks);
    
    // Lançar kernel
    stencil_1d<<<numBlocks, BLOCK_SIZE>>>(d_in, d_out);
    
    // Sincronizar
    cudaDeviceSynchronize();
    
    // Copiar resultado de volta para CPU
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    
    // Exibir todos os resultados (N é pequeno, então mostra tudo)
    printf("Resultados (soma de %d vizinhos):\n", 2*RADIUS + 1);
    printf("Índice | Input | Output | Vizinhos somados\n");
    printf("-------|-------|--------|-----------------\n");
    for (int i = 0; i < N; i++) {
        printf("  %2d   |  %2d   |  %3d   | ", i, h_in[i], h_out[i]);
        
        // Mostrar quais vizinhos foram somados (para entendimento)
        printf("[");
        for (int offset = -RADIUS; offset <= RADIUS; offset++) {
            int idx = i + offset;
            if (idx >= 0 && idx < N) {
                printf("%d", h_in[idx]);
            } else {
                printf("?");  // Fora dos limites (comportamento indefinido)
            }
            if (offset < RADIUS) printf("+");
        }
        printf("]\n");
    }
    
    // Liberar memória
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    
    printf("\nPrograma concluído!\n");
    
    return 0;
}