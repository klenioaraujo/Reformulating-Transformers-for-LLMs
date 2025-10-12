import json
from typing import List

class NativeTokenizer:
    """
    Um decodificador minimalista que carrega um vocabulário nativo local para
    converter IDs de token em texto, sem depender da biblioteca transformers.
    Implementa autonomia de vocabulário usando vocabulário emergente.
    """
    def __init__(self, vocab_path="data/native_vocab.json"):
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)

            # Handle native vocabulary format with metadata
            if isinstance(vocab_data, dict) and 'id_to_token' in vocab_data:
                # Native vocabulary format
                self.decoder = {int(id): token for id, token in vocab_data['id_to_token'].items()}
                self.vocab_size = vocab_data.get('vocab_size', len(self.decoder))
                self.special_tokens = vocab_data.get('metadata', {}).get('special_tokens', [])
                print(f"✅ NativeTokenizer carregado com sucesso: {self.vocab_size} tokens nativos.")
            else:
                # Fallback to simple token->id format
                self.decoder = {int(id): token for token, id in vocab_data.items()}
                self.vocab_size = len(vocab_data)
                self.special_tokens = []
                print(f"✅ NativeTokenizer carregado (formato simples): {self.vocab_size} tokens.")

        except FileNotFoundError:
            print(f"❌ ERRO: Arquivo de vocabulário não encontrado em {vocab_path}")
            self.decoder = {}
            self.vocab_size = 0
            self.special_tokens = []

    def decode(self, token_ids: List[int]) -> str:
        """
        Converte uma lista de IDs de token em uma string de texto.
        Usa vocabulário nativo emergente para autonomia completa.
        """
        if not self.decoder:
            return "[ERRO: Vocabulário não carregado]"

        text_parts = []
        for token_id in token_ids:
            token_str = self.decoder.get(token_id, '')

            # Handle special tokens from native vocabulary
            if token_str in ['<pad>', '<unk>', '<bos>', '<eos>', '<mask>']:
                # Special tokens - skip or handle appropriately
                if token_str == '<eos>':
                    break  # End of sequence
                continue  # Skip other special tokens
            else:
                text_parts.append(token_str)

        # Join all parts
        result = ''.join(text_parts)

        # Clean up extra spaces
        return result.strip()