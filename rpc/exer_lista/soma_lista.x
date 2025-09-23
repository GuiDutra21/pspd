#define VERSION_NUMBER 1
struct lista{
    int num;
    struct lista *prox;
};

typedef struct lista lista;


program PROG{
    version VER {
        int soma(lista) = 1;
    } = 1;
} = 0x13;