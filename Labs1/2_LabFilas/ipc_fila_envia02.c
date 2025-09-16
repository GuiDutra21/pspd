#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/msg.h>

struct my_msgbuf{
    long mytype;
    char text[12];
};

int main()
{
    int msqid;
    key_t key;
    struct my_msgbuf buffer;

    if ((key = ftok("ipc_fila_envia02.c", 'B') ) == -1 )
    {
        fprintf(stderr, "Erro na geracao do ID da fila.\n");
        exit(1);
    }

    if((msqid = msgget(key, 0666 | IPC_CREAT)) == -1 )
    {
        fprintf( stderr, "Erro na criacao da fila.\n" );
        exit(1);
    }

    printf( "Digite mensagens de até 10 caracteres (Ctrl+D para sair):\n" );
    buffer.mytype = 1; // tipo da mensagem

    while(fgets(buffer.text, sizeof(buffer.text), stdin))
    {
        // OBS: Faz um cast da struct my_msgbuf para a struct msgbuf
        if(msgsnd(msqid, (struct msgbuf *) &buffer,sizeof(buffer.text), 0))
        {
            fprintf( stderr, "Erro no envio da mensagem.\n");
        }
    }
    return 0;
}