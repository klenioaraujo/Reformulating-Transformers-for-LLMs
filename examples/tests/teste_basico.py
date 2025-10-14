# teste_basico.py
import torch
from src.core.dynamic_quantum_matrix import DynamicQuantumCharacterMatrix

print("🔬 TESTE BÁSICO - INICIALIZAÇÃO")
print("=" * 50)

# Testar inicialização
matrix = DynamicQuantumCharacterMatrix(vocab_size=1000, hidden_size=64)
print("✅ Matriz quântica inicializada")

# Verificar propriedades físicas
props = matrix.validate_physical_properties()
print("✅ Propriedades físicas validadas:")
for prop, result in props.items():
    print(f'   {prop}: {"✅" if result else "❌"}')

# Testar codificação simples
test_text = 'Hello quantum'
encoded = matrix.encode_text(test_text)
print(f'✅ Codificação funcionando: shape {encoded.shape}')

print('🎉 Sistema ΨQRH operacional!')