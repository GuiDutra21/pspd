#include <stdio.h>
#include <omp.h>
#define MAX 1000000

//  Cada thread le um numero no arquivo arqteste

int main()
{
    int lido;
    int n = 3;
    int soma = 0;
    #pragma omp parallel private(lido) reduction(+:soma)
    {
        
        FILE *fd; // Tem que ser local
        fd = fopen("arqnums.txt", "r");
        int thid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int chunk = MAX/nthreads;
        int largura = 9;
        int offset = thid * largura * chunk;
        fseek(fd, offset, SEEK_SET);
        for(int i = 0; i < chunk; i++)
        {
            fscanf(fd, "%d", &lido);
            if(lido == n)
                soma++;
        }
    }
    printf("%d ocorreu %d no arquivo\n",n,soma);

    return 0;
}