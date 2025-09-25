#define MASTER 0
#include <stdio.h>
#include <mpi.h>

int main()
{
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int vetSend[4];
    int vetRecv[4];

    for (int i = 0; i < 4; i++)
    {
        vetSend[i] = rank * 10 + i;
    }

    MPI_Alltoall(vetSend, 1, MPI_INT, vetRecv, 1, MPI_INT, MPI_COMM_WORLD);

    printf("Rank %d: ",rank);
    for (int i = 0; i < 4; i++)
    {
        printf("%d ",vetRecv[i]);
    }
    printf("\n");

    MPI_Finalize();
    return 0;
}