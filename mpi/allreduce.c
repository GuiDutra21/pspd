#define MASTER 0
#include <stdio.h>
#include <mpi.h>

int main()
{
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int dado = rank * 10;
    int resposta;

    MPI_Allreduce(&dado, &resposta, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    // if(rank == MASTER)
        printf("Processo %d recebeu %d\n", rank, resposta);

    MPI_Finalize();
    return 0;
}