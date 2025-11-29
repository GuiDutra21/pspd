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
    
    #pragma omp parallel private(thid)
    {
        thid = omp_get_thread_num();
        
        #pragma single
        nthreads = omp_get_num_threads();
        
        #pragma omp for
        for(int i = 0; i < MAX; i++)
        {
            C[i] = A[i] + B[i];
            
            #pragma omp atomic // ou #pragma omp critical
            soma += C[i];
        }
    }
    printf("Resultado final: %d\n",soma);

    return 0;
}