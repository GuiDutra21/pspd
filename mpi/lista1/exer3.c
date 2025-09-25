#include <stdio.h>
#include <mpi.h>
#define MASTER 0
#define SLAVE 1
#define TAG 0

// O master manda algo para todos os processos e depois recebe os dados de volta alterado
int main()
{
    int rank, nprocs;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    
    if( rank == MASTER)
    {
        int k[3] = {10, 20, 30};
        for(int i = 1; i < nprocs; i++)
        {
            MPI_Send(&k, 3, MPI_INT, i, TAG, MPI_COMM_WORLD);

        }
        for(int i = 1; i < nprocs; i++)
        {
            MPI_Recv(&k, 3, MPI_INT, i, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // No lugar do i eu posso colocar MPI_ANY_SOURCE para ele receber de qualquer slave

            printf("Valrores recebidos: ");
            for(int j = 0; j < 3; j++)
            {   
                printf(" %d",k[j]);
            }
            printf("\n");
        }
    }
    else
    {
        int k_rec[3];
        MPI_Recv(&k_rec, 3, MPI_INT, MASTER, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for(int i = 0; i < 3; i++)
        {   
            k_rec[i] = k_rec[i] * 2 * rank;
        }
        MPI_Send(&k_rec,3,  MPI_INT, MASTER, TAG, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();

    return 0;
}