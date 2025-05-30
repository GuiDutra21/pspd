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
// 	char* str = (char*)shmat(shmid, (void*)0, 0);

// 	cout << "Data read from memory:" << str;

// 	// detach from shared memory
// 	shmdt(str);

// 	// destroy the shared memory
// //	shmctl(shmid, IPC_RMID, NULL);

// 	return 0;
// }

#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>

int main() {
    // ftok para gerar uma chave única
    key_t key = ftok("shmfile", 65);

    // shmget retorna um identificador na variável shmid
    int shmid = shmget(key, 1024, 0666 | IPC_CREAT);

    // shmat para anexar a memória compartilhada
    char* str = (char*) shmat(shmid, (void*)0, 0);

    printf("Dados lidos da memória: %s\n", str);

    // Desanexa a memória compartilhada
    shmdt(str);

    // Destroi a memória compartilhada (descomentar se desejar remover)
    shmctl(shmid, IPC_RMID, NULL);

    return 0;
}
