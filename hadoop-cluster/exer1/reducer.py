#!/usr/bin/python3
"""reducer.py"""
from operator import itemgetter
import sys

palavra_att = None
count_att = 0
palavra = None

for linha in sys.stdin:
    linha = linha.strip() # Remove os espacos em branco
    palavra, count = linha.split(' ', 1) # Pega o que foi apssado pelo mapper.py
    
    try:
        count = int(count)
    except ValueError: # Se o valor de count nnnao for int ignora e continua
        continue
    # Este if so funcionna porque o Hadoop ordena a saida do map
    # por uma chave (no caso palavra) antes de ser passado para o reducer
    if palavra_att == palavra:
        count_att += count
    else:
        if palavra_att:
            print (f"{palavra_att}  {count_att}") # Escreve o resultado na saida
        count_att = count
        palavra_att = palavra
        
if palavra_att == palavra:
    print(f"{palavra_att} {count_att}")