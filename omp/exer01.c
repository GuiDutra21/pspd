#include <stdio.h>
#include <omp.h>

int main()
{
    int thid, nthreads;

    omp_set_num_threads(5);

// Apenas a thread mais rapida executa o omp_get_num_threads()
// e como ela eh private apenas ela recebe o valor correto do numero de threads
#pragma omp parallel private(thid, nthreads) num_threads(3)
    {
        thid = omp_get_thread_num();
#pragma omp single
        nthreads = omp_get_num_threads();
        printf("%d/%d --> Regiao paralela\n", thid, nthreads);
    }

    printf("%d/%d Estou na area sequencial\n", thid, nthreads);

#pragma omp parallel private(thid) shared(nthreads) num_threads(5)
    {
        thid = omp_get_thread_num();
        nthreads = omp_get_num_threads();
#pragma omp single
        printf("%d/%d --> Outra regiao paralela\n", thid, nthreads);
    }

    printf("%d/%d Estou na area sequencial novamemte\n", thid, nthreads);
    return 0;
}

// #include <stdio.h>
// #include <omp.h>
// int main()
// {
//     int myid, nthreads;
//     omp_set_num_threads(5);
//     nthreads = omp_get_num_threads(); // Se executada fora do pragma so vai retornar 1
//     printf("teste %d\n",nthreads);
//     #pragma omp parallel private(myid) shared(nthreads)
//     {
//         myid = omp_get_thread_num();
//         printf("%d of %d – hello world!\n", myid, nthreads);
//     }
//     return 0;
// }