#include <stdio.h>

__global__ void hello_cuda() {
    printf("Olá, mundo, da GPU!\n");
}

int main() {
    hello_cuda<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
