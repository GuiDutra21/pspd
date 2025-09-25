struct lista {
    int num;
    struct lista *prox;
};

program SOMA_LISTA {
    version SOMA_LISTA_VERSION{
        int add(lista) = 1;
    } = 55;
} = 0x13;