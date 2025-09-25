#define MASTER 0
#include <stdio.h>
#include <mpi.h>

int main()
{
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int vet[4];          // vetor original (usado só no root)
    int resultado[4];    // vetor para coletar resultados (usado só no root)
    int dadoRecebido;

    if (rank == MASTER)
    {
        vet[0] = 10;
        vet[1] = 20;
        vet[2] = 30;
        vet[3] = 40;
    }

    // Distribui 1 número para cada processo
    MPI_Scatter(vet, 1, MPI_INT, &dadoRecebido, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    // Cada processo multiplica o valor por 2
    dadoRecebido *= 2;

    // Coleta de volta os valores processados
    MPI_Gather(&dadoRecebido, 1, MPI_INT, resultado, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    // Só o root imprime o vetor final
    if (rank == MASTER)
    {
        printf("Vetor final: ");
        for (int i = 0; i < size; i++)
        {
            printf("%d ", resultado[i]);
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
