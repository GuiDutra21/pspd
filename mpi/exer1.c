#include <stdio.h>
#include <mpi.h>
#define MASTER 0
#define SLAVE 1
#define TAG 0

int main()
{
    int rank, nprocs;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    
    if( rank == MASTER)
    {
        int k;
        printf("Informe um int: ");
        fflush(stdout);   // força a impressão antes do scanf
        scanf("%d",&k);
        MPI_Send(&k, 1, MPI_INT, SLAVE, TAG, MPI_COMM_WORLD);
    }
    else
    {
        int k_rec;
        MPI_Recv(&k_rec, 1, MPI_INT, MASTER, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Valor recebido: %d\n", k_rec);
    }
    MPI_Finalize();
    return 0;
}