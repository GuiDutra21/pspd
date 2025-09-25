// Apenas para imprimir o hostname de cada processo
#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <unistd.h>

#define MASTER 0
#define TAG 0

int main(int argc, char **argv)
{
    int rank, nprocs;
    char msg[100];
    char host[100];
    MPI_Status st;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    int token;
    
    if (rank != MASTER)
    {   
        gethostname(host, sizeof(host) - 1);
        sprintf(msg,"Hi, I am alive and I am running on %s!\n", host);
        MPI_Send(msg, strlen(msg) + 1, MPI_CHAR, MASTER, TAG, MPI_COMM_WORLD);
    }
    else
    {      
        for(int i = 1; i < nprocs; i++)
        {
            MPI_Recv(msg,100, MPI_CHAR, MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &st);
            printf("Processo: %d/%d enviou: %s\n", st.MPI_SOURCE, nprocs, msg);
        }
    }

    MPI_Finalize();
    return 0;
}