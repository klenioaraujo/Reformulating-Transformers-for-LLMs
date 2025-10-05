#!/usr/bin/env python3
"""
Criar mapeamento char → token_id do GPT-2
Baseado no vocabulário do GPT-2 (aproximação sem transformers)
"""

import json

print('=== CRIANDO MAPEAMENTO CHAR → GPT-2 TOKEN ===')

# Mapeamento básico baseado em caracteres ASCII
char_to_gpt2_token = {}

# Caracteres comuns e seus tokens aproximados
# Estes são tokens que provavelmente existem no vocabulário do GPT-2
char_mapping = {
    ' ': 220,    # Espaço
    '!': 0,      # Exclamação
    '"': 1,      # Aspas
    '#': 2,      # Hash
    '$': 3,      # Dólar
    '%': 4,      # Percentual
    '&': 5,      # E comercial
    "'": 6,      # Apóstrofo
    '(': 7,      # Parêntese esquerdo
    ')': 8,      # Parêntese direito
    '*': 9,      # Asterisco
    '+': 10,     # Mais
    ',': 11,     # Vírgula
    '-': 12,     # Hífen
    '.': 13,     # Ponto
    '/': 14,     # Barra
    '0': 15,     # Zero
    '1': 16,     # Um
    '2': 17,     # Dois
    '3': 18,     # Três
    '4': 19,     # Quatro
    '5': 20,     # Cinco
    '6': 21,     # Seis
    '7': 22,     # Sete
    '8': 23,     # Oito
    '9': 24,     # Nove
    ':': 25,     # Dois pontos
    ';': 26,     # Ponto e vírgula
    '<': 27,     # Menor que
    '=': 28,     # Igual
    '>': 29,     # Maior que
    '?': 30,     # Interrogação
    '@': 31,     # Arroba
    'A': 32,     # A maiúsculo
    'B': 33,     # B maiúsculo
    'C': 34,     # C maiúsculo
    'D': 35,     # D maiúsculo
    'E': 36,     # E maiúsculo
    'F': 37,     # F maiúsculo
    'G': 38,     # G maiúsculo
    'H': 39,     # H maiúsculo
    'I': 40,     # I maiúsculo
    'J': 41,     # J maiúsculo
    'K': 42,     # K maiúsculo
    'L': 43,     # L maiúsculo
    'M': 44,     # M maiúsculo
    'N': 45,     # N maiúsculo
    'O': 46,     # O maiúsculo
    'P': 47,     # P maiúsculo
    'Q': 48,     # Q maiúsculo
    'R': 49,     # R maiúsculo
    'S': 50,     # S maiúsculo
    'T': 51,     # T maiúsculo
    'U': 52,     # U maiúsculo
    'V': 53,     # V maiúsculo
    'W': 54,     # W maiúsculo
    'X': 55,     # X maiúsculo
    'Y': 56,     # Y maiúsculo
    'Z': 57,     # Z maiúsculo
    '[': 58,     # Colchete esquerdo
    '\\': 59,    # Barra invertida
    ']': 60,     # Colchete direito
    '^': 61,     # Circunflexo
    '_': 62,     # Sublinhado
    '`': 63,     # Crase
    'a': 64,     # a minúsculo
    'b': 65,     # b minúsculo
    'c': 66,     # c minúsculo
    'd': 67,     # d minúsculo
    'e': 68,     # e minúsculo
    'f': 69,     # f minúsculo
    'g': 70,     # g minúsculo
    'h': 71,     # h minúsculo
    'i': 72,     # i minúsculo
    'j': 73,     # j minúsculo
    'k': 74,     # k minúsculo
    'l': 75,     # l minúsculo
    'm': 76,     # m minúsculo
    'n': 77,     # n minúsculo
    'o': 78,     # o minúsculo
    'p': 79,     # p minúsculo
    'q': 80,     # q minúsculo
    'r': 81,     # r minúsculo
    's': 82,     # s minúsculo
    't': 83,     # t minúsculo
    'u': 84,     # u minúsculo
    'v': 85,     # v minúsculo
    'w': 86,     # w minúsculo
    'x': 87,     # x minúsculo
    'y': 88,     # y minúsculo
    'z': 89,     # z minúsculo
    '{': 90,     # Chave esquerda
    '|': 91,     # Barra vertical
    '}': 92,     # Chave direita
    '~': 93,     # Tilde
}

# Adicionar ao mapeamento
for char, token_id in char_mapping.items():
    char_to_gpt2_token[char] = token_id

# Salvar arquivo
output_path = 'models/gpt2_full_spectral_embeddings/char_to_gpt2_token.json'
with open(output_path, 'w') as f:
    json.dump(char_to_gpt2_token, f, indent=2)

print(f'✅ Mapeamento salvo: {output_path}')
print(f'• {len(char_to_gpt2_token)} caracteres mapeados')
print(f'• Exemplo: "H" → {char_to_gpt2_token["H"]}')
print(f'• Exemplo: "e" → {char_to_gpt2_token["e"]}')
print(f'• Exemplo: " " → {char_to_gpt2_token[" "]}')