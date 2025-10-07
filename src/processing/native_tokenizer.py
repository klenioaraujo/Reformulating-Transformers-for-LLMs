import json
from typing import List

class NativeTokenizer:
    """
    Um decodificador minimalista que carrega um vocab.json local para
    converter IDs de token em texto, sem depender da biblioteca transformers.
    """
    def __init__(self, vocab_path="data/gpt2_vocab.json"):
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                # O vocab.json mapeia token -> id
                encoder = json.load(f)
            # Nós precisamos do inverso: id -> token
            self.decoder = {int(id): token for token, id in encoder.items()}
            self.vocab_size = len(encoder)
            print(f"✅ NativeTokenizer carregado com sucesso: {self.vocab_size} tokens.")
        except FileNotFoundError:
            print(f"❌ ERRO: Arquivo de vocabulário não encontrado em {vocab_path}")
            self.decoder = {}
            self.vocab_size = 0

    def decode(self, token_ids: List[int]) -> str:
        """
        Converte uma lista de IDs de token em uma string de texto.
        Lida com a decodificação do vocabulário GPT-2.
        """
        if not self.decoder:
            return "[ERRO: Vocabulário não carregado]"

        text_parts = []
        for token_id in token_ids:
            token_str = self.decoder.get(token_id, '')

            # GPT-2 usa 'Ġ' para representar espaço
            if token_str.startswith('Ġ'):
                # Ġ representa espaço no GPT-2
                text_parts.append(' ' + token_str[1:])
            elif token_str == '<|endoftext|>':
                # Token especial - podemos ignorar
                continue
            else:
                text_parts.append(token_str)

        # Junta todas as partes
        result = ''.join(text_parts)

        # Remove espaços extras no início se houver
        return result.lstrip()