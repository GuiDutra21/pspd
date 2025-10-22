#include <stdio.h>
#include <omp.h>
#include <unistd.h>

#define MAX 14
// Com o schedule(runtime) podemos alterar o tipo do schedule, e fazemos isso assim:
// export OMP_SCHEDULE="static,2"
int main(int argc, char *argv[])
{
    long int sum = 0;
    #pragma omp parallel for reduction (+:sum) schedule(runtime)
    
        for (int i = 1; i <= MAX; i++)
        {
            printf("Iteracoes %2d na thread %d\n", i, omp_get_thread_num());
            sleep(i < 4 ? i + 1:1);
            sum += 1;
        }
    

    printf("%ld\n",sum);

    return 0;
}