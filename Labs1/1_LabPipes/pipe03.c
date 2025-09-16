#include <stdio.h>
#include <sys/types.h>
#include <stdlib.h>
#include <unistd.h>

int fd[2];

int main()
{
    pid_t pid;
    pipe(fd);

    pid = fork();

    if(pid == 0)
    {
        close(fd[0]);
        printf("Escreva uma emnsagem de ate 10 caracteres: ");
        char buffer[5];
        fgets(buffer,5,stdin);
        write(fd[1],buffer,sizeof(buffer));
        exit(0);
    }
    else
    {
        close(fd[1]);
        char buffer[5];
        read(fd[0], buffer, sizeof(buffer));
        printf("mensagem recebida: %s",buffer);
    }
    return 0;
}