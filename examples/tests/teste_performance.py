# teste_performance.py
import time
import torch
from src.core.dynamic_quantum_matrix import DynamicQuantumCharacterMatrix

print('🚀 TESTE DE PERFORMANCE - MATRIZ QUÂNTICA OTIMIZADA')
print('=' * 60)

# Testar com diferentes tamanhos
vocab_sizes = [1000, 5000, 10000]

for vocab_size in vocab_sizes:
    print(f'\n📊 Testando com vocab_size = {vocab_size}')

    start_time = time.time()

    # Inicializar matriz
    matrix = DynamicQuantumCharacterMatrix(vocab_size=vocab_size, hidden_size=256)
    init_time = time.time() - start_time

    # Testar adaptação
    adapt_start = time.time()
    matrix.adapt_to_model('gpt2')
    adapt_time = time.time() - adapt_start

    # Testar codificação
    encode_start = time.time()
    test_text = 'Hello quantum world with optimized matrix'
    encoded = matrix.encode_text(test_text)
    encode_time = time.time() - encode_start

    print(f'   ⏱️  Inicialização: {init_time:.3f}s')
    print(f'   ⏱️  Adaptação: {adapt_time:.3f}s')
    print(f'   ⏱️  Codificação: {encode_time:.3f}s')
    print(f'   📐 Shape final: {encoded.shape}')

    # Verificar estabilidade numérica
    finite_check = torch.isfinite(encoded).all().item()
    print(f'   🔍 Valores finitos: {"✅" if finite_check else "❌"}')

print('\n🎉 TESTE DE PERFORMANCE CONCLUÍDO!')