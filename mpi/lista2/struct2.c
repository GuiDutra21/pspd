// Elabore um programa para fazer uso de MPI_Type_contiguous

#define MASTER 0

#include <stdio.h>
#include <string.h>
#include <mpi.h>

#define STR_LEN 20
#define QTD 3

struct pessoa
{
    int idade;
    char nome[STR_LEN];
    float peso;
};

typedef struct pessoa pessoa;


MPI_Datatype createNewType()
{
    int qtdCampos = QTD;
    int lenghts[QTD] = {1,STR_LEN,1};
    MPI_Aint offsets[QTD] = {0, sizeof(int), sizeof(int) + STR_LEN};
    MPI_Datatype types[QTD] = {MPI_INT, MPI_CHAR, MPI_FLOAT};
    MPI_Datatype tipo_pessoa;
    MPI_Type_create_struct(qtdCampos, lenghts, offsets, types, &tipo_pessoa);
    MPI_Type_commit(&tipo_pessoa);
    return tipo_pessoa;
}

int main()
{
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Datatype tipo_pessoa = createNewType();
    
    if (rank == MASTER)
    {
        pessoa p;
        p.idade = 21;
        strcpy(p.nome, "Gui");
        p.peso = 79.10;
        MPI_Send(&p, 1, tipo_pessoa, 1, 0, MPI_COMM_WORLD);
    }
    else
    {
        pessoa p;
        MPI_Recv(&p,1, tipo_pessoa, 0,0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Dados recebidos %d %s %f\n",p.idade, p.nome, p.peso);
    }
    MPI_Type_free(&tipo_pessoa);
    MPI_Finalize();
 
    return 0;
}