// #include <iostream>
// #include <stdio.h>
// #include <sys/ipc.h>
// #include <sys/shm.h>
// using namespace std;

// int main() {
// 	// ftok to generate unique key
// 	key_t key = ftok("shmfile", 65);

// 	// shmget returns an identifier in shmid
// 	int shmid = shmget(key, 1024, 0666 | IPC_CREAT);

// 	// shmat to attach to shared memory
// 	// (void*)0 → o sistema escolhe o endereço onde a memória será mapeada
// 	char* str = (char*)shmat(shmid, (void*)0, 0);

// 	cout << "Write Data : ";
// 	cin.getline(str, 1024);

// 	cout << "Data written in memory: " << str << endl;

// 	// detach from shared memory
// 	// shmdt(str);

// 	return 0;
// }

#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <string.h>

int main() {
    // ftok para gerar uma chave única
    key_t key = ftok("shmfile", 65);

    // shmget retorna um identificador na variável shmid
    int shmid = shmget(key, 1024, 0666 | IPC_CREAT);

    // shmat para anexar a memória compartilhada
    char* str = (char*) shmat(shmid, (void*)0, 0);

    printf("Digite os dados: ");
    fgets(str, 1024, stdin); // lê a string com espaços

    // Remove o \n final que fgets adiciona
    str[strcspn(str, "\n")] = '\0';

    printf("Dados escritos na memória: %s\n", str);

    // Desanexa a memória compartilhada (opcional)
    shmdt(str);

    return 0;
}
