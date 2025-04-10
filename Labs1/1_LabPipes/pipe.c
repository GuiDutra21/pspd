#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>

int main(void) {

        int     fd[2]; 
        // fd = file descriptor (descritor de arquivo).
        // É um número inteiro que o sistema operacional usa para identificar um arquivo aberto, socket, pipe, etc.
        // fd[0] le dados do pipe
        // fd[1] escreve dados do pipe

        int nbytes;
        pid_t   pid;
        char    string[] = "Olá, mundo!";
        char    readbuffer[80];

        pipe(fd);
        if((pid = fork()) == -1) {
                perror("fork");
                exit(1);
        } /* fim-if */
        if(pid == 0) {
                /* Child process closes up input side of pipe */
                close(fd[0]);
                /* Send "string" through the output side of pipe */
		printf("Processo [%d] --> Enviando string pelo pipe...\n", getpid());
                write(fd[1], string, (strlen(string)+1));
                exit(0);
        } else {
                /* Parent process closes up output side of pipe */
                close(fd[1]);
                /* Read in a string from the pipe */
                nbytes = read(fd[0], readbuffer, sizeof(readbuffer));
                printf("Processo [%d] --> Recebendo string do pipe ...: %s\n", getpid(), readbuffer);
        } /* fim-if */

        printf("fd[0] = %d, fd[1] = %d\n", fd[0], fd[1]);

        return(0);
} /* fim-main */
