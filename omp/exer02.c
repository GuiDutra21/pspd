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
        int nthreads = omp_get_num_threads();
        int chunk = MAX/nthreads;
        int largura = 9;
        int offset = thid * largura *  chunk;
        fseek(fd, offset, SEEK_SET);
        for(int i = 0; i<chunk; i++)
        {
            fscanf(fd, "%d", &lido);
            printf("%d/%d --> Valor lido %d\n",thid,nthreads, lido);
        }
        fclose(fd);
    }
    return 0;
}