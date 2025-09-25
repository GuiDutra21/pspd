Uso do Bcast

#define MASTER 0
#include <stdio.h>
#include <mpi.h>

int main()
{   
    MPI_Init(NULL,NULL);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int dado;
    if(rank == MASTER)
    {   
        dado = 10;
    }

    MPI_Bcast(&dado,1,MPI_INT,MASTER,MPI_COMM_WORLD);

    if(rank != MASTER)
        printf("Sou o slave %d e recebi o dado: %d\n",rank, dado);

    MPI_Finalize();
    return 0;
}

