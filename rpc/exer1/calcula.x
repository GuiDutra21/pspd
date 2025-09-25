struct operandos{
    int x;
    int y;
}

program PROG{
    version VERSION{
        int add(operandos) = 1;
        int sub(operandos) = 1;
    } = 100;
} = 5555;