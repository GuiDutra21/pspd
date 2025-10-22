#include <stdio.h>
#include <omp.h>
#define MAX 8

int main()
{
    int thid,nthreads;
    int A[MAX], B[MAX], C[MAX];
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

        int chunk = MAX/nthreads;
        int ini = thid * chunk;
        int fim = ini + chunk;

        if(thid == nthreads - 1)
            fim = MAX;
        
        for(int i = ini; i < fim; i++)
        {
            C[i] = A[i] + B[i];
            printf("%d/%d --> %d\n",thid, nthreads, C[i]);
        }
    }

    return 0;
}