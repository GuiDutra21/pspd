#!/usr/bin/python3
"""mapper.py"""
import sys
# Os inputs vem da entrada padra, STDIN
for linha in sys.stdin:
    linha = linha.strip() # Remove os espacos em branco
    palavras = linha.split() # Divide a linha em duas palavras
    for count in palavras:
        print(f"{count} 1") # a saida desse apsso servira de entrada para po reduce.py