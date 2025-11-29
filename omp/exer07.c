#include <stdio.h>
#include <omp.h>
#define MAX 8

// Cada thread le chunk numeros do arquivo arqteste
int main()
{
    int lido;
    
    
    #pragma omp parallel private(lido)
    {
        FILE *fd;
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

// #include <stdio.h>
// #include <omp.h>
// #define MAX 8

// int main() {
//     char linha[16];
//     int lido;
//     FILE *fd;

//     #pragma omp parallel private(fd, linha, lido)
//     {
//         fd = fopen("arqteste.txt", "r");
//         int thid = omp_get_thread_num();
//         int nthreads = omp_get_num_threads();
//         int chunk = MAX / nthreads;
//         int largura = 9;
//         int offset = thid * largura * chunk;
//         fseek(fd, offset, SEEK_SET);

//         for (int i = 0; i < chunk; i++) {
//             if (fgets(linha, sizeof(linha), fd)) {
//                 lido = atoi(linha);
//                 printf("%d/%d --> Valor lido %d\n", thid, nthreads, lido);
//             }
//         }

//         fclose(fd);
//     }
//     return 0;
// }
