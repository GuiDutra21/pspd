#include <stdio.h>
#include <mpi.h>
#define MASTER 0
#define SLAVE 1
#define TAG 0
#define TAM 8 

int main()
{
    int rank, nprocs;
    int token;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
        
    if (rank == MASTER)
    {
        token = 0;
        printf("Estou no MASTER com o  token = %d\n",token);
        MPI_Send(&token, 1, MPI_INT, 1, TAG, MPI_COMM_WORLD);
        MPI_Recv(&token, 1, MPI_INT, (nprocs - 1), TAG, MPI_COMM_WORLD ,MPI_STATUS_IGNORE);
        printf("Estou no MASTER com o token = %d\n",token);

    }
    else
    {   
        MPI_Recv(&token, 1, MPI_INT, rank - 1, TAG, MPI_COMM_WORLD ,MPI_STATUS_IGNORE);
        token++;
        printf("Estou no processo %d/%d com token = %d\n",rank, nprocs - 1, token);
        MPI_Send(&token, 1, MPI_INT, ((rank + 1)%nprocs), TAG, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();

    return 0;
}