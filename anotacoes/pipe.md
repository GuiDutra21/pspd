# Passo a passo para a criação do Pipe unidirecional

> OBS: Para realizar o pipe utilizei de um processo pai que recebe os dados e de um processo filho que manda os dados através do pipe

## 1. Declare as variáveis e importe as bibliotecas 

>   
    #include <stdio.h>
    #include <stdlib.h>
    #include <unistd.h>
    #include <string.h>
    #include <sys/types.h>
    
    int fd[2];
    char entrada[50], readbuffer[80];
    int nbytes;
    pid_t pid;

## 2. Chame a função **pipe(fd)** para realmente criar o pipe

> 
    pipe(fd);

Assim o sistema cria um canal de comunicação unidirecional (um pipe), e preenche fd (file descriptor) com dois descritores:  
- fd[0] que lê dados do pipe  
- fd[1] que escreve dados no pipe

> OBS: O fd[0] sempre é de leitura e o fd[1] de escrita, não tem como inverter

##  3. Faça o fork

>
    if((pid = fork()) == -1)
        return 0;

## 4. Para o processo filho

  4.1. Feche o fd[0]  
  4.2. Leia algo do terminal  
  4.3. Mande através do pipe com a função write  
  4.4. Feche o fd[1] e finalize o processo filho



>   
    else if (pid == 0)
    {
        printf("Digite uma mensagem: ");  
        scanf(" %[^\n]", entrada);  
        write(fd[1], entrada, strlen(entrada) + 1);
        close(fd[1]);  
        exit(0);
    }


##  5. Para o processo pai 

5.1 Feche o fd[1]  
5.2 Leia a mensagem do pipe através da função read  
5.3 Mostre a mensagem lida  
5.4 Feche o fd[0]

>
    else
    {
        close(fd[1]);
        nbytes = read(fd[0], readbuffer, sizeof(readbuffer));
        printf("A mensagem passada foi: %s e foi lido %d bytes\n", readbuffer, nbytes);
        close(fd[0]);
    }

## Código completo

>
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