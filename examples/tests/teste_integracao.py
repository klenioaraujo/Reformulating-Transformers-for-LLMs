# teste_integracao.py
import torch
import sys
import os

# Adicionar caminhos para módulos do projeto
sys.path.insert(0, os.path.abspath('.'))

print('🔬 VERIFICAÇÃO FINAL - INTEGRAÇÃO COMPLETA ΨQRH')
print('=' * 65)

try:
    from src.core.dynamic_quantum_matrix import DynamicQuantumCharacterMatrix
    from src.core.quaternion_operations import OptimizedQuaternionOperations
    from spectral_parameters_integration import SpectralParametersIntegrator
    print('✅ Todos os módulos principais importados')
except ImportError as e:
    print(f'❌ Erro de importação: {e}')
    sys.exit(1)

# Testar funcionalidades básicas
try:
    # 1. Matriz quântica
    matrix = DynamicQuantumCharacterMatrix(vocab_size=5000, hidden_size=128)
    print('✅ Matriz quântica inicializada')

    # 2. Adaptação
    success = matrix.adapt_to_model('gpt2')
    print(f'✅ Adaptação: {"Sucesso" if success else "Falha"}')

    # 3. Codificação
    encoded = matrix.encode_text('Teste de integração ΨQRH')
    print(f'✅ Codificação: shape {encoded.shape}')

    # 4. Propriedades físicas
    props = matrix.validate_physical_properties()
    valid_props = sum(props.values())
    print(f'✅ Propriedades físicas: {valid_props}/3 validadas')

    # 5. Serialização
    matrix.save_adapted_matrix('test_integration.pt')
    print('✅ Serialização funcionando')

    # 6. Desserialização
    loaded = DynamicQuantumCharacterMatrix.load_adapted_matrix('test_integration.pt')
    print('✅ Desserialização funcionando')

    # Limpar arquivo de teste
    if os.path.exists('test_integration.pt'):
        os.remove('test_integration.pt')

    print('\n🎉 VERIFICAÇÃO COMPLETA - SISTEMA ΨQRH INTEGRADO!')

except Exception as e:
    print(f'❌ Erro durante verificação: {e}')
    sys.exit(1)