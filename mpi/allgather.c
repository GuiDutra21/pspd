#define MASTER 0
#include <stdio.h>
#include <mpi.h>

// 
int main()
{
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int send = rank + 1;
    int recv[4];

    MPI_Allgather(&send, 1, MPI_INT, recv, 1, MPI_INT, MPI_COMM_WORLD);

    printf("Vetor do rank %d : ", rank);
    for (int j = 0; j < 4; j++)
    {
        printf("%d ", recv[j]);
    }
    printf("\n");

    MPI_Finalize();
    return 0;
}
