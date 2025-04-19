# Passo a passo para a criação de uma memória compartilahda entre arquivos

Para criar essa "comunicação" utilizamos de 2 arquivos, um que escreve na memória e outro que lê dessa memória

## No arquivo que escreve na memória faça:
1. ftok
2. shmget
3. shmat 
4. Escreva algo na string retornada na função anterior
5. shmdt

## No arquivo que lê da memória faça:
1. ftok
2. shmget
3. shmat
4. Lê da string retornada na função anterior
5. shmdt
6. (Opcional) destroi a meória usando a função shmctl