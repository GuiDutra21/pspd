// #include <stdio.h>
// #include <mpi.h>

// #define MASTER 0
// #define TAG 0

// int main(int argc, char **argv)
// {
//     int rank, nprocs;
//     MPI_Init(&argc, &argv);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

//     int token;

//     if (rank == MASTER)
//     {
//         printf("Informe um numero inteiro: ");
//         fflush(stdout);
//         scanf("%d", &token);

//         printf("Processo %d inicia com %d\n", rank, token);

//         MPI_Send(&token, 1, MPI_INT, (rank + 1) % nprocs, TAG, MPI_COMM_WORLD);
//     }

//     while (1)
//     {
//         MPI_Recv(&token, 1, MPI_INT, (rank - 1 + nprocs) % nprocs, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

//         if (token < 0)
//         {
//             // repassa o "sinal de parada" e sai
//             MPI_Send(&token, 1, MPI_INT, (rank + 1) % nprocs, TAG, MPI_COMM_WORLD);
//             break;
//         }

//         token--;
//         if (token >= 0)
//         {
//             printf("Processo %d recebeu token = %d\n", rank, token);
//         }

//         if (token == 0)
//         {
//             // ao chegar em zero, manda o -1 para avisar todo mundo
//             int stop = -1;
//             MPI_Send(&stop, 1, MPI_INT, (rank + 1) % nprocs, TAG, MPI_COMM_WORLD);
//             break;
//         }
//         else
//         {
//             MPI_Send(&token, 1, MPI_INT, (rank + 1) % nprocs, TAG, MPI_COMM_WORLD);
//         }
//     }

//     MPI_Finalize();
//     return 0;
// }

// Carrosel que executa token vezes
#include <stdio.h>
#include <mpi.h>

#define MASTER 0
#define TAG 0

int main(int argc, char **argv)
{
    int rank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int token;

    if (rank == MASTER)
    {
        printf("Informe um numero inteiro: ");
        fflush(stdout);
        scanf("%d", &token);
        
        printf("Processo %d inicia com %d\n", rank, token);
        
        MPI_Send(&token, 1, MPI_INT, (rank + 1) % nprocs, TAG, MPI_COMM_WORLD);
    }

    while (1)
    {
        MPI_Recv(&token, 1, MPI_INT, (rank - 1 + nprocs) % nprocs, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        token--;

        if (token >= 0)
        {
            printf("Processo %d recebeu token = %d\n", rank, token);
        }

        MPI_Send(&token, 1, MPI_INT, (rank + 1) % nprocs, TAG, MPI_COMM_WORLD);

        // Sai do laco pois ja nao vai mais receber 
        if (token - nprocs <= 0)
        {
            break;
        }
    }

    MPI_Finalize();
    return 0;
}