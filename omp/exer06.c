#include <stdio.h>
#include <omp.h>
#define MAX 8

//  Cada thread le um numero no arquivo arqteste

int main()
{
    int lido;
    FILE *fd;

    #pragma omp parallel private(lido)
    {
        
        fd = fopen("arqteste.txt", "r");
        int thid = omp_get_thread_num();
        int largura = 9;
        int offset = thid * largura;
        fseek(fd, offset, SEEK_SET);
        fscanf(fd, "%d", &lido);
        printf("%d --> Valor lido %d\n",thid, lido);
    }
    return 0;
}

// int main()
// {
//     int lido;
//     FILE fd;
    
//     #pragma omp parallel private(lido)
//     {
        
//         fd = fopen("arqteste.txt", "r");
//         int thid = omp_get_thread_num();
//         int largura = 9;
//         int offset = thid * largura;
//         fseek(fd,offset,SEEK_SET);
//         fscanf(fd,"%d", &lido);
//         printf("%d --> Valor lido %d\n",thid, lido);
//     }
// }