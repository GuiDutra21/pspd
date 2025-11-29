#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#define MASTER 0

int main()
{
    MPI_Init(NULL, NULL);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand((unsigned)time(NULL) + rank);

    int vet[100];

    for (int i = 0; i < 100; i++)
    {
        vet[i] = i;
    }

    int idx = 0;
    if (rank == MASTER)
    {
        int slave = 1;
        while (idx < 100)
        {
            MPI_Send(&idx, 1, MPI_INT, slave, 0, MPI_COMM_WORLD);
            MPI_Recv(&idx, 1, MPI_INT, slave, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            slave++;
            if (slave == size)
            {
                slave = 1;
            }
        }

        // Envia sinal de parada para todos os slaves
        int para = -1;
        for (int slave = 1; slave < size; slave++)
        {
            MPI_Send(&para, 1, MPI_INT, slave, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        while (1)
        {

            MPI_Recv(&idx, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (idx == -1)
                break;

            int qtdPrint = (rand() % 15) + 1;
            int limite;

            if (idx + qtdPrint > 99)
                limite = 100;
            else
                limite = idx + qtdPrint;

            printf("slave %d => ", rank);
            for (int i = idx; i < limite; i++)
            {
                printf("%d ", vet[i]);
            }
            printf("\n");
            MPI_Send(&limite, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
