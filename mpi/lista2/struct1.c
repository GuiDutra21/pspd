// Elabore um programa para fazer uso de MPI_Type_contiguous

#define MASTER 0

#include <stdio.h>
#include <mpi.h>

struct token
{
    int contador;
    int indice;
};

typedef struct token token;

int main()
{
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Datatype tipo_token;
    MPI_Type_contiguous(2, MPI_INT, &tipo_token);
    MPI_Type_commit(&tipo_token);
    
    if (rank == MASTER)
    {
        token tk;
        tk.contador = 10;
        tk.indice = 5;
        MPI_Send(&tk, 1, tipo_token, 1, 0, MPI_COMM_WORLD);
    }
    else
    {
        token tk;
        MPI_Recv(&tk,1, tipo_token, 0,0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Dados recebidos %d %d\n",tk.contador, tk.indice);
    }
    MPI_Type_free(&tipo_token);
    MPI_Finalize();
 
    return 0;
}
