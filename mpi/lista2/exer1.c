/* Elaborar um programa MPI de
modo que os processos calculem,
colaborativamente a média dos valores
de um vetor de 8 posições. Nesse caso,
dividir uma quantidade de posições para
cada processo usando MPI_Scatter e
recolha as parciais usando MPI_Gather */

#define MASTER 0

#include <stdio.h>
#include <mpi.h>

int main()
{
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int vet1[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    int tmp[2];       // cada processo recebe 2 valores
    int vet2[4];      // resultados parciais (soma de cada processo)

    // distribui 2 valores do vetor original para cada processo
    MPI_Scatter(vet1, 2, MPI_INT, tmp, 2, MPI_INT, MASTER, MPI_COMM_WORLD);

    // cada processo soma seus dois valores
    int retorno = tmp[0] + tmp[1];

    // coleta todas as somas parciais no processo MASTER
    MPI_Gather(&retorno, 1, MPI_INT, vet2, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    if (rank == MASTER)
    {
        int resultado = 0;
        for (int i = 0; i < size; i++)
        {
            resultado += vet2[i];  // soma de todas as parciais
        }
        double media = (double)resultado / 8.0;
        printf("Média: %.2f\n", media);
    }

    MPI_Finalize();
    return 0;
}