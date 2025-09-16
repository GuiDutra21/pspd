#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>

int main()
{
    pid_t pid;
    
    for (int i = 0; i < 3; i++)
    {
        pid = fork();
        printf("Jesus from PID %d\n", getpid());
        fflush(stdout); // força a saída no terminal
    }
    
    return 0;
}

// 1ª iteração → 2 prints
// 2ª iteração → 4 prints
// 3ª iteração → 8 prints

// Então: 2 + 4 + 8 = 14 prints no total