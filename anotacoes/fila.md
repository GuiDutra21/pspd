# Passo a passo para a criação de uma fila de mensagens

Para criar a fila utilizamos de 2 arquivos, um que envia a mensagem e outro que recebe

## No arquivo que envia a mensagem faça:

1. Gere o identificador único da fila atráves da função ftok
2. Crie a fila de mensagem atráves da função msgget (usando o identificador gerado anteriormente)
3. Passe a mensagem que pode ser alguma string ou alguma outra estrutura (struct) através da função msgsnd

## No arquivo que recebe a mensagem faça:

1. Gere o identificador único da fila atráves da função ftok
2. Se conecte a fila de mensagem criada pelo o outro arquivo através da função msgget (usando o identificador gerado anteriormente)
3. Faça um laço infinito e dentro dele coloque a função msgrcv para receber a mensagem -> vai ficar "escutando" se alguma mensagem chegou

## Para executar a lista de mensagens basta compilar e executar os dois arquivos

# Arquivos:

- Envia:

> 
    #include <stdio.h>
    #include <stdlib.h>
    #include <sys/types.h>
    #include <sys/ipc.h>
    #include <sys/msg.h>

    /* Estrutura a ser enviada a fila de mensagens */
    struct my_msgbuf {
        long mtype;
        char mtext[10];
    };

    int main(int argc, char **argv) {
        struct my_msgbuf buf;

        /* ID mantido pelo kernel */
        int msqid;

        /* Identificador unico da fila de mensagens */
        key_t key;
        
        /* Gerando o identificador unico da fila... */
        if( ( key = ftok( "ipc_fila_envia.c", 'B' ) ) == -1 ) {
            fprintf( stderr, "Erro na geracao do ID da fila.\n" );
        exit(1);
        }

        /* Criando a fila de mensagens */
        if( ( msqid = msgget( key, 0666 | IPC_CREAT ) ) == -1 ) {
            fprintf( stderr, "Erro na criacao da fila.\n" );
            exit(1);
        }

        printf("Arquivo que envia: msqid-%d e key-%d\n",msqid, key);

        /* Enviando mensagens */
        printf( "Digite mensagens de até 10 caracteres (Ctrl+D para sair):\n" );
        buf.mtype = 1;
        while( fgets( buf.mtext, 10, stdin ) /*&& !feof(stdin)*/ )  {
            // OBS: Faz um cast da struct my_msgbuf para a struct msgbuf
            if( msgsnd( msqid, (struct msgbuf *) &buf, sizeof(buf.mtext), 0 ) == -1 )
            fprintf( stderr, "Erro no envio da mensagem.\n" );
        }

    //    if( msgctl( msqid, IPC_RMID, NULL ) == -1 ) {
    //        fprintf( stderr, "Erro na remocao da fila.\n" );
    //        exit(1);
    //    }
        return 0;
    }

- Recebe:
> 
    #include <stdio.h>
    #include <stdlib.h>
    #include <sys/types.h>
    #include <sys/ipc.h>
    #include <sys/msg.h>

    /* Estrutura a ser enviada a fila de mensagens */
    struct my_msgbuf {
        long mtype;
        char mtext[10];
    };

    int main(int argc, char **argv) {

        struct my_msgbuf buf;
        
        /* ID mantido pelo kernel */
        int msqid;
        
        /* Identificador unico da fila de mensagens */
        key_t key;
        
        /* Gerando o identificador unico da fila... */
        if( ( key = ftok( "ipc_fila_envia.c", 'B' ) ) == -1 ) {
            fprintf( stderr, "Erro na geracao do ID da fila.\n" );
            exit(1);
        }
        
        /* Conectando-se a fila de mensagens */
        if( ( msqid = msgget( key, 0666 ) ) == -1 ) {
            fprintf( stderr, "Erro ao conectar a fila.\n" );
            exit(1);
        }
        
        printf("Arquivo que recebe: msqid-%d e key-%d\n",msqid, key);

        printf( "Recebendo mensagens...\n" );
        buf.mtype = 1;
        
        for(;;) {
            // OBS: Faz um cast da struct my_msgbuf para a struct msgbuf
            if( msgrcv( msqid, (struct msgbuf *) &buf, sizeof(buf.mtext), 0, 0 ) == -1 ) {
                fprintf( stderr, "Erro ao receber mensagem.\n" );
                exit(1);
            }
            printf( "MENSAGEM: %s\n", buf.mtext );
        }
        return 0;
    }