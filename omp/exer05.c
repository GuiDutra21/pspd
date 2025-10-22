#include <stdio.h>
#include <omp.h>
#define MAX 500000

int main()
{
    int thid,nthreads;
    int A[MAX], B[MAX], C[MAX];
    int soma = 0;
    for (int i = 0; i < MAX; i++)
    {
        A[i] = 1;
        B[i] = 2;

    }
    
    for(int i = 0; i < MAX; i++)
    {
        C[i] = A[i] + B[i];
        
        #pragma omp atomic
        soma += C[i];
    }
    printf("Resultado final: %d\n",soma);

    return 0;
}