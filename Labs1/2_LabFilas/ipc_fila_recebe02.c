#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/msg.h>

struct my_msgbuf
{
    long mtype;
    char text[12];
};

int main()
{
    int msqid;
    key_t key;
    struct my_msgbuf buffer;

    if ((key = ftok("ipc_fila_envia02.c", 'B')) == -1)
    {
        fprintf(stderr, "Erro na geracao do ID da fila.\n");
        exit(1);
    }

    if((msqid = msgget(key,0666)) == -1)
    {
        fprintf( stderr, "Erro ao conectar a fila.\n" );
        exit(1);
    }

    printf( "Recebendo mensagens...\n" );
    buffer.mtype = 1;
    
    while(1)
    {
        // OBS: Faz um cast da struct my_msgbuf para a struct msgbuf
        if(msgrcv(msqid, (struct msgbuf *) &buffer, sizeof(buffer.text), 0, 0 ) == -1 )
        {
            fprintf( stderr, "Erro ao receber mensagem.\n" );
            exit(1);
        }
        printf( "MENSAGEM: %s\n", buffer.text );
    }
    return 0;
}