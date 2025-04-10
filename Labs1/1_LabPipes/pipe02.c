#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>

int main()
{
    int fd[2];
    char entrada[50], readbuffer[80];
    int nbytes;
    pid_t   pid;
    pipe(fd);

    if((pid = fork()) == -1)
        return 0;
    else if (pid == 0)
    {
        close(fd[0]);
        printf("Digite uma mensagem: ");
        scanf(" %[^\n]", entrada); 
        write(fd[1], entrada, strlen(entrada) + 1);
        close(fd[1]);
        exit(0);
    }
    else
    {
        close(fd[1]);
        nbytes = read(fd[0], readbuffer, sizeof(readbuffer));
        printf("A mensagem passada foi: %s e foi lido %d bytes\n", readbuffer, nbytes);
        close(fd[0]);
    }
    
    return 0;
}