#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define MASTER 0 

int main()
{   
    MPI_Init(NULL,NULL);

    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // int n;
    // scanf("%d",&n);

    // int vet[n];
    // for(int i = 0; i < n; i++)
    // {
    //     scanf("%d",&vet[i]);
    // }
     srand((unsigned)time(NULL) + rank);
    printf("%d\n",rand() % 100);
    // if(rank == MASTER)
    // {

    // }
    // else
    // {

    // }

    
    MPI_Finalize();

    return 0;
}