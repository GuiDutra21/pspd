#include <stdio.h>
#include <mpi.h>
#define MASTER 0
#define SLAVE 1
#define TAG 0
#define TAM 8 

// Imprime o vetor de forma igualitaria entre os processos
int main()
{
    int rank, nprocs, vet[TAM];
    int ini, fim;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    for(int i = 0; i < TAM; i++)
    {
        vet[i] = (i + 1) * 10;
    }
    
    int chunk = (int) TAM/nprocs;
    ini = rank * chunk;
    fim = ini + chunk;
    
    if(rank == nprocs - 1)
    {
        fim = TAM;
    }

    printf("Processo %d/%d:",rank, nprocs - 1);
    for (int i = ini; i < fim; i++)
    {
        printf(" %d",vet[i]);
    }
    printf("\n");
    
    MPI_Finalize();

    return 0;
}