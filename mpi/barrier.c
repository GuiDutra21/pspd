// Elaborar um programa MPI para imprimir o rank do processo em ordem crescente, usando MPI_Barrier

#include <mpi.h>
#include <stdio.h>

int main()
{
    int rank, nprocess;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocess);

    for (int i = 0; i < nprocess; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == i)
        {
            printf("Meu rank %d\n", rank);
            fflush(stdout);
        }
    }

    MPI_Finalize();
    return 0;
}