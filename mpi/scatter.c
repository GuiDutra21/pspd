// Uso do Scatter

#define MASTER 0
#include <stdio.h>
#include <mpi.h>

int main()
{
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int vet[4];
    int dadoRecebido;

    if (rank == MASTER)
    {
        vet[0] = 10;
        vet[1] = 20;
        vet[2] = 30;
        vet[3] = 40;
    }

    MPI_Scatter(vet, 1, MPI_INT, &dadoRecebido, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    printf("Processo %d recebeu %d\n", rank, dadoRecebido);

    MPI_Finalize();
    return 0;
}