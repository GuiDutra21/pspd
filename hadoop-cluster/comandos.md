# Montagem de um cluster Hadoop com Docker Compose

## Criação do cluster Hadoop com Docker Compose

- Criei uma pasta, por exemplo `hadoop-cluster`, e dentro dela criei um arquivo chamado `docker-compose.yml` com o seguinte conteúdo:

```yaml

services:
  hadoop-master:
    image: ubuntu:24.04
    container_name: hadoop-master
    hostname: hadoop-master
    privileged: true
    tty: true
    stdin_open: true
    networks:
      hadoop-network:
        ipv4_address: 172.20.0.10
    volumes:
      - hadoop_master_data:/hadoop-data
      - ./shared:/shared
    command: /bin/bash -c "sleep infinity"

  hadoop-slave1:
    image: ubuntu:24.04
    container_name: hadoop-slave1
    hostname: hadoop-slave1
    privileged: true
    tty: true
    stdin_open: true
    networks:
      hadoop-network:
        ipv4_address: 172.20.0.11
    volumes:
      - hadoop_slave1_data:/hadoop-data
      - ./shared:/shared
    command: /bin/bash -c "sleep infinity"

  hadoop-slave2:
    image: ubuntu:24.04
    container_name: hadoop-slave2
    hostname: hadoop-slave2
    privileged: true
    tty: true
    stdin_open: true
    networks:
      hadoop-network:
        ipv4_address: 172.20.0.12
    volumes:
      - hadoop_slave2_data:/hadoop-data
      - ./shared:/shared
    command: /bin/bash -c "sleep infinity"

networks:
  hadoop-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

volumes:
  hadoop_master_data:
  hadoop_slave1_data:
  hadoop_slave2_data:
```

## Iniciar o cluster
- Dentro da pasta inicie o cluster executando o comando:

```bash
docker-compose up -d
```

## Acessar os containers

- Antes de prosseguir rode o comando `docker ps` para verificar os nomes dos containers.

- Ex na coluna: `NAMES`

-  hadoop-master

-  hadoop-slave1

-  hadoop-slave2

OBS: Os nomes dos containers podem variar, utilize os nomes corretos conforme o retorno do comando `docker ps`.

- Abra 3 terminais diferentes e acesse cada container com os comandos, neste exemplo estou utilizando os nomes dos containers conforme o exemplo acima, mas utilize os nomes corretos conforme o retorno do comando `docker ps`:

```bash
docker exec -it hadoop-master bash
docker exec -it hadoop-slave1 bash
docker exec -it hadoop-slave2 bash
```
Obs: Rode cada comando em um terminal diferente.

## Baixando e configurando o Hadoop

- Dentro de cada container, ou seja, em cada terminal, execute os seguintes comandos para baixar e configurar o Hadoop:

### Instalando o Java e o OpenSSH

```bash
apt-get update
apt install openjdk-21-jdk openssh-server
```
- Agora precisamos localizar o JAVA_HOME. Execute, em apenas 1 terminal, o seguinte comando abaixo para encontrar o caminho do Java instalado:

```bash
whereis java
```
- O comando retornará algo como `/usr/bin/java /usr/share/java /usr/share/man/man1/java.1.gz`.
- Agora precisamos descobrir o caminho completo do Java. Execute o comando abaixo:

```bash
readlink -f /usr/bin/java ( ou o primeiro caminho retornado pelo comando anterior)
```

- O comando retornará algo como `/usr/lib/jvm/java-21-openjdk-amd64/bin/java`.
- Copiei o caminho até a parte `java-21-openjdk-amd64`, ou seja, o caminho completo do JAVA_HOME é `/usr/lib/jvm/java-21-openjdk-amd64`.
- Antes de prosseguir, instale o nano ou vim
```bash
apt install nano
```

- Agora, em cada terminal, abra o arquivo `~/.bashrc` para adicionar a variável de ambiente JAVA_HOME

```bash
nano ~/.bashrc
```

- No final do arquivo, adicione a seguinte linha (substitua o caminho pelo que você encontrou):

```bash
JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
PATH=$PATH:$JAVA_HOME
```
- Salve e feche o arquivo. Em seguida, execute o comando abaixo para aplicar as alterações:

```bash
source ~/.bashrc
```

- Para testar se a variável foi configurada corretamente, execute:

```bash
echo $JAVA_HOME
```
- Deve retornar o caminho que você adicionou. Ex: `/usr/lib/jvm/java-21-openjdk-amd64`

### Configurando o SSH sem senha

- Agora, em cada terminal, execute o seguinte comando para iniciar o ssh:

```bash
service ssh start
```
- Agora, em cada terminal, execute o seguinte comando e pressione Enter para todas as opções:

OBS: Após rodar o comando a primeira opção que irá aparecer é o local para salvar a chave, copiei o caminho que é mostrado entre parenteses, Ex: `/root/.ssh/id_rsa`

```bash
ssh-keygen -t rsa
```
- Em seguida, execute o comando abaixo, em cada terminal, para adicionar a chave pública ao arquivo `authorized_keys`:

OBS: No caminho copiado anteriormente, acrescente `.pub` no final.
```bash
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
```

- Por fim, em cada terminal, execute o comando abaixo para copiar a chave pública para o localhost:

```bash
ssh-copy-id -i /root/.ssh/id_rsa.pub root@localhost
```
- Quando solicitado, digite `yes` para continuar e pressione Enter quando pedir a senha (não há senha definida, então apenas pressione Enter).

- Agora, em cada terminal, teste a conexão SSH com o comando:

```bash
ssh localhost
```

- OBS: Toda vez que você subir novamente os containers, você precisará iniciar o serviço SSH novamente com o comando `service ssh start`.

- Se tudo estiver configurado corretamente, você deverá conseguir se conectar sem precisar digitar uma senha.

- Agora, precisamos configurar o SSH sem senha entre o master e os slaves.

### Configurando o SSH sem senha entre o master e os slaves


- No terminal do hadoop-master, execute os seguintes comandos para copiar a chave pública para os slaves:

```bash
ssh-copy-id -i /root/.ssh/id_rsa.pub root@hadoop-slave1
ssh-copy-id -i /root/.ssh/id_rsa.pub root@hadoop-slave2
```

- No hadoop-master, rode o seguinte comando para exibir a chave pública gerada anteriormente e **copie** o conteúdo exibido:
```bash
cat ~/.ssh/id_rsa.pub
```

- No terminal de cada slave, adicione a chave copiada manualmente:

```bash
mkdir -p ~/.ssh
echo "COLE_A_CHAVE_PUBLICA_DO_MASTER_AQUI" >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

- Agora teste a conexão SSH sem senha do master para os slaves:
- OBS: Substitua `hadoop-slave1` e `hadoop-slave2` pelos endereços IP ou nomes corretos dos slaves, se necessário.


- Agora faça o mesmo processo para os slaves se conectarem ao master e entre si.

- No hadoop-slave1, rode o seguinte comando para exibir a chave pública gerada anteriormente e **copie** o conteúdo exibido:



### Baixando e configurando o Hadoop

----
- Depois de entrar no localhost, entre na pasta opt em cada terminal:

```bash
cd ..
cd opt
```

- Agora, em cada terminal, execute os seguintes comandos para baixar e extrair o Hadoop:

```bash
wget https://dlcdn.apache.org/hadoop/common/hadoop-3.4.2/hadoop-3.4.2.tar.gz
tar -xvzf hadoop-3.4.2.tar.gz
```
- Agora, em cada terminal, execute o comando abaixo para atribuir a porta e o hostname de cada nó:

```bash
nano etc/hadoop/core-site.xml
<configuration>
      <property><name>fs.default.name</name><value>hdfs://0.0.0.0:9000</value></property>
</configuration>

```

- Agora, abra o arquivo hadoop-env.sh em cada terminal:

```bash
nano etc/hadoop/hadoop-env.sh
```

- Encontre a linha que contém:

```bash 
# The java implementation to use. By default, this environment
# variable is REQUIRED on ALL platforms except OS X!
export JAVA_HOME= 
```

- Tire o comentário da linha que contem o `export JAVA_HOME=` e modifique-a para incluir o caminho do JAVA_HOME que você configurou anteriormente:

```bash
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
```
- No final desse mesmo arquivo, tambem adicione as seguintes linhas em cada terminal:

? Retirar o que esta embaixo ?
```bash
export HDFS_NAMENODE_USER="root"
export HDFS_DATANODE_USER="root"
export HDFS_SECONDARYNAMENODE_USER="root"
export YARN_RESOURCEMANAGER_USER="root"
export YARN_NODEMANAGER_USER="root"
```

- Agora, em cada termina, execute o comando abaixo para formartar o NameNode:

```bash
bin/hadoop namenode -format
```

- Agora, em cada terminal, execute os seguintes comandos para instalar o sudo:

```bash
apt update
apt install -y sudo
echo "root ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
```
- Para iniciar o HDFS, execute o comando abaixo:

```bash
sbin/start-dfs.sh
```

jps

http://172.20.0.11:9870/dfshealth.html#tab-overview


bin/hadoop dfs -mkdir /user

bin/hadoop dfs -mkdir /user/root

nano ~/.bashrc

export HADOOP_HOME=/opt/hadoop-3.4.2

export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin

source ~/.bashrc

----

# Multi-node

ssh-keygen -t rsa (opcional)



entre em cada terminal no opt/hadoop-3.4.2 e abra o arquivo `etc/hadoop/core-site.xml`, nele no lugar do 0.0.0.0:9000 coloque o nome hadoop-master:9000
<configuration>
        <property><name>fs.default.name</name><value>hdfs://hadoop-master:9000</value></property>
</configuration>

agora apenas no master abra o arquivo etc/hadoop/workers e adicione os nomes dos slaves:
hadoop-slave1
hadoop-slave2

- Agora, em cada terminal, abra o arquivo `etc/hadoop/hdfs-site.xml` e adicione o seguinte conteúdo:
<configuration>

    <!-- Define o diretorio onde o NameNode armazena seus metadados -->
    <property>
        <name>dfs.namenode.name.dir</name>
        <value>file:/opt/hadoop-3.4.2/hdfs/namenode</value>
    </property>

    <!-- Define o diretorio onde o DataNode armazena os blocos de dados -->
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>file:/opt/hadoop-3.4.2/hdfs/datanode</value>
    </property>


</configuration>

