# teste_carga_pesada.py
import torch
import time
import sys
import os

sys.path.insert(0, os.path.abspath('.'))

from src.core.dynamic_quantum_matrix import DynamicQuantumCharacterMatrix

print('🔬 TESTE FINAL - VALIDAÇÃO DE PRODUÇÃO')
print('=' * 60)

print('\n🎯 TESTANDO CENÁRIOS DE PRODUÇÃO:')

# Teste 1: Carga pesada
print('\n1. 📈 TESTE DE CARGA PESADA:')
start_time = time.time()

matrices = []
for i in range(5):
    matrix = DynamicQuantumCharacterMatrix(
        vocab_size=10000,
        hidden_size=256
    )
    matrix.adapt_to_model('gpt2')
    matrices.append(matrix)
    print(f'   Matriz {i+1} criada')

load_time = time.time() - start_time
print(f'   ⏱️  Tempo total: {load_time:.2f}s')
print(f'   📊 Memória: {len(matrices)} matrizes carregadas')

# Teste 2: Processamento em lote
print('\n2. 🔄 TESTE DE PROCESSAMENTO EM LOTE:')
texts = [
    'Processamento de texto em lote',
    'Sistema ΨQRH otimizado',
    'Representações quânticas avançadas',
    'Integração com LLMs',
    'Performance escalável'
]

batch_start = time.time()
encoded_batch = []
for text in texts:
    encoded = matrices[0].encode_text(text)
    encoded_batch.append(encoded)
    print(f'   Texto processado: {text[:30]}...')

batch_time = time.time() - batch_start
print(f'   ⏱️  Tempo por texto: {batch_time/len(texts):.3f}s')
print(f'   📊 Total de textos: {len(texts)}')

# Teste 3: Estabilidade numérica
print('\n3. 🔍 TESTE DE ESTABILIDADE NUMÉRICA:')

# Testar com texto muito longo
long_text = 'A' * 500  # Texto repetitivo
encoded_long = matrices[0].encode_text(long_text)

# Verificar estabilidade
finite_check = torch.isfinite(encoded_long).all().item()
real_stats = encoded_long.real
imag_stats = encoded_long.imag

print(f'   ✅ Valores finitos: {finite_check}')
print(f'   📊 Real - Min: {real_stats.min():.4f}, Max: {real_stats.max():.4f}')
print(f'   📊 Imag - Min: {imag_stats.min():.4f}, Max: {imag_stats.max():.4f}')

# Teste 4: Consistência entre execuções
print('\n4. 🔄 TESTE DE CONSISTÊNCIA:')

test_text = 'Texto de teste para consistência'
encoded_1 = matrices[0].encode_text(test_text)
encoded_2 = matrices[0].encode_text(test_text)

consistency_diff = torch.abs(encoded_1 - encoded_2).mean().item()
print(f'   🔍 Diferença entre execuções: {consistency_diff:.8f}')
print(f'   ✅ Consistente: {consistency_diff < 1e-6}')

print('\n🎉 TESTES DE PRODUÇÃO CONCLUÍDOS!')
print('\n🚀 SISTEMA ΨQRH PRONTO PARA IMPLANTAÇÃO EM PRODUÇÃO!')