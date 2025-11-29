#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define MASTER 0 
#define TAG 0 

int main()
{   
    MPI_Init(NULL,NULL);

    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    srand((unsigned)time(NULL) + rank); // Para ter a aleatoridade

    // So para leitura dos dados e compartilhamento com os slaves
    int n;
    if(rank == MASTER)
    {
        scanf("%d",&n);
        for (int i = 1; i < size; i++)
        MPI_Send(&n, 1, MPI_INT, i, TAG, MPI_COMM_WORLD);
        
    }
    else
    {
        MPI_Recv(&n,1 ,MPI_INT, MASTER, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // So para leitura dos dados e compartilhamento com os slaves
    int vet[n];
    if(rank == MASTER)
    {
        for(int i = 0; i < n; i++)
            scanf("%d",&vet[i]);
        
        for(int i = 1; i < size; i++)
            MPI_Send(&vet, n, MPI_INT, i, TAG, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Recv(&vet,n ,MPI_INT, MASTER, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    if(rank == MASTER)
    {   
        int indice = 0;
        while (indice < n)
        {
            int filho = rand() % (size - 1) + 1;
            MPI_Send(&indice, 1, MPI_INT, filho, TAG, MPI_COMM_WORLD);
            MPI_Recv(&indice, 1, MPI_INT, filho, TAG,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Envia sinal de parada para todos
        indice = -1;
        for(int i = 1; i < size; i++)
        {
            MPI_Send(&indice, 1, MPI_INT, i, TAG, MPI_COMM_WORLD);
        }
    }
    else
    {
        int qtdChamado = 0;
        int indice;
        while(1)
        {
            MPI_Recv(&indice, 1 ,MPI_INT, MASTER, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            if(indice == -1)
            break;
            
            qtdChamado++;
            int limite = indice + (qtdChamado * rank);
            if(limite > n) limite = n;
            
            for(int i = indice; i < limite; i++)
            {
                printf("%d ",vet[i] * (i+1));
                fflush(stdout);
            }
            indice = limite;
            MPI_Send(&indice, 1, MPI_INT, MASTER, TAG, MPI_COMM_WORLD);
        }
    }
    
    
    MPI_Finalize();
    if(rank == MASTER)
    {
        printf("\n");
    }
    return 0;
}