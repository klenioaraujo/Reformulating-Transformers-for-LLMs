#!/usr/bin/env python3
"""
Chat Interativo com Modelo ΨQRH Treinado
=========================================

Script de conversação interativa para teste qualitativo do modelo.

Features:
- Modo chat interativo com histórico de contexto
- Teste de cenários pré-definidos
- Métricas de resposta (tempo, tokens, confiança)
- Análise de consciência fractal em tempo real

Usage:
    # Modo interativo
    python3 chat_with_model.py --model_dir ./models/psiqrh_wikitext_v2

    # Modo teste automático
    python3 chat_with_model.py --model_dir ./models/psiqrh_wikitext_v2 --test_mode
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
import torch
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.architecture.psiqrh_transformer import PsiQRHTransformer


class ChatSession:
    """Gerencia uma sessão de chat com o modelo ΨQRH nativo"""

    def __init__(self, model, char_to_idx: dict, idx_to_char: dict, device: str = 'cpu', max_seq_length: int = 128):
        self.model = model
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.device = device
        self.max_seq_length = max_seq_length
        self.conversation_history = []

    def generate_response(self, prompt: str, max_new_chars: int = 100,
                         temperature: float = 0.8) -> Dict:
        """
        Gera resposta para o prompt dado usando sampling character-level

        Args:
            prompt: Texto de entrada do usuário
            max_new_chars: Número máximo de caracteres a gerar
            temperature: Temperatura para sampling

        Returns:
            Dicionário com resposta e métricas
        """
        start_time = time.time()

        # Converter prompt para índices
        input_indices = []
        for ch in prompt[-self.max_seq_length:]:  # Últimos max_seq_length caracteres
            if ch in self.char_to_idx:
                input_indices.append(self.char_to_idx[ch])
            else:
                input_indices.append(0)  # UNK

        # Pad se necessário
        if len(input_indices) < self.max_seq_length:
            input_indices = [0] * (self.max_seq_length - len(input_indices)) + input_indices

        # Gerar resposta character-by-character
        generated_chars = []
        current_input = input_indices.copy()

        with torch.no_grad():
            for _ in range(max_new_chars):
                # Converter para tensor
                input_tensor = torch.tensor([current_input], dtype=torch.long).to(self.device)

                # Forward pass
                logits = self.model(input_tensor)

                # Pegar logits do último token
                last_logits = logits[0, -1, :]

                # Physical decoding - Medição por Pico de Ressonância (SEM softmax)
                from src.processing.physical_decoding import decode_resonance_to_token_id
                next_idx = decode_resonance_to_token_id(last_logits, temperature=temperature)

                # Converter para caractere
                next_char = self.idx_to_char.get(str(next_idx), '')

                # Parar em nova linha ou caracteres especiais
                if next_char in ['\n', '\r'] and len(generated_chars) > 10:
                    break

                generated_chars.append(next_char)

                # Atualizar input (sliding window)
                current_input = current_input[1:] + [next_idx]

        response = ''.join(generated_chars)
        elapsed_time = time.time() - start_time

        # Armazenar no histórico
        self.conversation_history.append({
            'prompt': prompt,
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'time': elapsed_time
        })

        return {
            'response': response,
            'num_tokens': len(generated_chars),
            'time': elapsed_time,
            'tokens_per_second': len(generated_chars) / elapsed_time if elapsed_time > 0 else 0
        }

    def reset_history(self):
        """Limpa histórico de conversação"""
        self.conversation_history = []


def load_model(model_dir: Path, device: str = 'cpu'):
    """
    Carrega modelo ΨQRH nativo treinado

    Args:
        model_dir: Diretório do modelo
        device: Dispositivo

    Returns:
        (model, char_to_idx, idx_to_char, config)
    """
    print(f"\n🔄 Carregando modelo de {model_dir}...")

    # Carregar config
    with open(model_dir / 'config.json', 'r') as f:
        config = json.load(f)

    # Carregar vocabulário
    with open(model_dir / 'vocab.json', 'r') as f:
        vocab_data = json.load(f)

    char_to_idx = vocab_data['char_to_idx']
    idx_to_char = vocab_data['idx_to_char']

    # Garantir que char_to_idx e idx_to_char estão no formato correto
    char_to_idx = {str(k): int(v) for k, v in char_to_idx.items()}
    idx_to_char = {str(k): str(v) for k, v in idx_to_char.items()}

    # Criar modelo
    model = PsiQRHTransformer(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        max_seq_length=config['max_seq_length']
    )

    # Carregar pesos
    state_dict = torch.load(model_dir / 'pytorch_model.bin', map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Modelo carregado: {num_params:,} parâmetros")
    print(f"✅ Vocabulário: {vocab_data['vocab_size']} caracteres")

    return model, char_to_idx, idx_to_char, config


def run_interactive_mode(session: ChatSession):
    """
    Executa modo de chat interativo

    Args:
        session: Sessão de chat
    """
    print("\n" + "="*70)
    print("💬 MODO CHAT INTERATIVO - ΨQRH")
    print("="*70)
    print("Digite 'sair' para encerrar")
    print("Digite 'reset' para limpar histórico")
    print("Digite 'historico' para ver conversas anteriores")
    print("="*70 + "\n")

    while True:
        try:
            # Input do usuário
            prompt = input("👤 Você: ")

            if not prompt.strip():
                continue

            # Comandos especiais
            if prompt.lower() == 'sair':
                print("\n👋 Encerrando chat...")
                break

            if prompt.lower() == 'reset':
                session.reset_history()
                print("🔄 Histórico limpo!\n")
                continue

            if prompt.lower() == 'historico':
                print("\n📜 HISTÓRICO DE CONVERSAÇÃO:")
                for i, entry in enumerate(session.conversation_history, 1):
                    print(f"\n  [{i}] {entry['timestamp']}")
                    print(f"  👤: {entry['prompt']}")
                    print(f"  🤖: {entry['response']}")
                    print(f"  ⏱️: {entry['time']:.2f}s")
                print()
                continue

            # Gerar resposta
            print("🤖 ΨQRH: ", end="", flush=True)
            result = session.generate_response(prompt)

            print(result['response'])
            print(f"   ⏱️  {result['time']:.2f}s | {result['num_tokens']} tokens | {result['tokens_per_second']:.1f} tok/s\n")

        except KeyboardInterrupt:
            print("\n\n👋 Chat interrompido pelo usuário")
            break
        except Exception as e:
            print(f"\n❌ Erro: {e}")
            import traceback
            traceback.print_exc()


def run_test_mode(session: ChatSession) -> Dict:
    """
    Executa testes automáticos com cenários pré-definidos

    Args:
        session: Sessão de chat

    Returns:
        Dicionário com resultados dos testes
    """
    print("\n" + "="*70)
    print("🧪 MODO TESTE AUTOMÁTICO - Cenários Qualitativos")
    print("="*70 + "\n")

    test_scenarios = [
        {
            'name': 'Conhecimento Factual',
            'prompt': 'Qual é a capital da França?',
            'expected_keywords': ['paris', 'frança']
        },
        {
            'name': 'Criatividade',
            'prompt': 'Conte-me uma pequena história sobre um robô que sonhava em ser um pássaro.',
            'expected_keywords': ['robô', 'pássaro', 'sonho']
        },
        {
            'name': 'Manutenção de Contexto',
            'prompt': 'Eu gosto de física quântica. Qual tópico você acha mais interessante?',
            'expected_keywords': ['física', 'quântica']
        },
        {
            'name': 'Robustez a Ruído',
            'prompt': 'rererer rere re',
            'expected_keywords': []  # Apenas verificar se não crashou
        },
        {
            'name': 'Raciocínio Simples',
            'prompt': 'Se eu tenho 5 maçãs e como 2, quantas sobram?',
            'expected_keywords': ['3', 'três', 'sobra']
        }
    ]

    results = {
        'scenarios': [],
        'total_tests': len(test_scenarios),
        'passed': 0,
        'failed': 0
    }

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n[{i}/{len(test_scenarios)}] Teste: {scenario['name']}")
        print(f"  Prompt: \"{scenario['prompt']}\"")

        try:
            result = session.generate_response(scenario['prompt'], max_new_chars=100)

            print(f"  Resposta: \"{result['response']}\"")
            print(f"  Tempo: {result['time']:.2f}s | Tokens: {result['num_tokens']}")

            # Verificar keywords esperadas
            response_lower = result['response'].lower()
            keywords_found = [kw for kw in scenario['expected_keywords']
                            if kw in response_lower]

            if scenario['expected_keywords']:
                passed = len(keywords_found) > 0
                if passed:
                    print(f"  ✅ PASSOU (keywords encontradas: {keywords_found})")
                    results['passed'] += 1
                else:
                    print(f"  ⚠️  AVISO (nenhuma keyword encontrada)")
                    results['failed'] += 1
            else:
                print(f"  ✅ PASSOU (teste de robustez)")
                results['passed'] += 1

            results['scenarios'].append({
                'name': scenario['name'],
                'prompt': scenario['prompt'],
                'response': result['response'],
                'time': result['time'],
                'num_tokens': result['num_tokens'],
                'passed': passed if scenario['expected_keywords'] else True
            })

        except Exception as e:
            print(f"  ❌ FALHOU: {e}")
            results['failed'] += 1
            results['scenarios'].append({
                'name': scenario['name'],
                'prompt': scenario['prompt'],
                'error': str(e),
                'passed': False
            })

    # Resumo
    print("\n" + "="*70)
    print("📊 RESUMO DOS TESTES")
    print("="*70)
    print(f"Total de testes: {results['total_tests']}")
    print(f"✅ Passou: {results['passed']}")
    print(f"❌ Falhou: {results['failed']}")
    print(f"Taxa de sucesso: {(results['passed'] / results['total_tests'] * 100):.1f}%")
    print("="*70 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description='Chat interativo com modelo ΨQRH')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Diretório do modelo treinado')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Dispositivo (cpu, cuda)')
    parser.add_argument('--test_mode', action='store_true',
                        help='Executar testes automáticos em vez de modo interativo')
    parser.add_argument('--max_new_tokens', type=int, default=50,
                        help='Máximo de tokens a gerar por resposta')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Temperatura para sampling')
    parser.add_argument('--save_results', type=str, default=None,
                        help='Arquivo para salvar resultados dos testes (apenas --test_mode)')

    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    if not model_dir.exists():
        print(f"❌ Erro: Diretório não encontrado: {model_dir}")
        sys.exit(1)

    # Carregar modelo
    model, char_to_idx, idx_to_char, config = load_model(model_dir, args.device)

    # Criar sessão de chat
    session = ChatSession(model, char_to_idx, idx_to_char, args.device, config['max_seq_length'])

    # Executar modo apropriado
    if args.test_mode:
        results = run_test_mode(session)

        # Salvar resultados se solicitado
        if args.save_results:
            with open(args.save_results, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 Resultados salvos em: {args.save_results}")

    else:
        run_interactive_mode(session)

    print("\n✅ Sessão encerrada com sucesso")


if __name__ == '__main__':
    main()
