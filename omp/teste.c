#include <stdio.h>
#include <omp.h>

int main()
{
    int i = 0;

#pragma omp parallel
    {
        if (omp_get_thread_num() == 1)
        {
            // #pragma omp atomic
            i = i + 10;
        }
    }

    printf("i=%d\n", i);
    return 0;
}

#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define MAX 1000
int *indice;
char texto_base[] = "abcdefghijklmnopqrstuvwxyz 1234567890 ABCDEFGHIJKLMNOPQRSTUVWXYZ";
void imprimeTexto(void)
{
    int tmp_index, i;
    struct timeval tv;
    int number;
    gettimeofday(&tv, NULL);
    number = ((tv.tv_usec / 47) % 3) + 1;
    tmp_index = *indice;
    for (i = 0; i < number; i++)
        if (!(tmp_index + i > sizeof(texto_base)))
        {
            fprintf(stderr, "%c", texto_base[tmp_index + i]);
            usleep(1);
        } /* fim-if */
    *indice = tmp_index + i;
    if (tmp_index + i > sizeof(texto_base))
    {
        fprintf(stderr, "\n");
        *indice = 0;
    } /* fim-if */
} /* fim-imprimeTexto */
int main()
{
    indice = (int *)malloc(sizeof(int));
    *indice = 0;
    int cont = 0;
#pragma omp parallel num_threads(3)
    {
        printf("Thread %d iniciada...\n", omp_get_thread_num());
        sleep(1);
        /* Entrando no loop principal */
        while (cont < MAX)
        {
#pragma omp critical
            imprimeTexto();
#pragma omp atomic
            cont++;
        } /* fim-while */
    } /* fim-pragma */
    printf("\n");
    return 0;
}
/* fim-main *