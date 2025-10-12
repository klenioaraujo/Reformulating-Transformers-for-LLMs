import pytest
import torch
from src.core.harmonic_orchestrator import HarmonicOrchestrator

@pytest.fixture
def orchestrator():
    """Provides a default HarmonicOrchestrator instance for tests."""
    return HarmonicOrchestrator(device='cpu')

def test_orchestrator_initialization(orchestrator):
    """Testa se o HarmonicOrchestrator é inicializado corretamente."""
    assert orchestrator is not None
    assert hasattr(orchestrator, 'signature_analyzer')
    print("✅ Teste de Inicialização do Orquestrador Harmônico passou.")

def test_orchestration_of_a_base_function(orchestrator):
    """
    Testa a principal funcionalidade do orquestrador: modificar o comportamento
    de uma função base com base na assinatura harmônica de um sinal.
    """
    # 1. Sinal de entrada simples
    mock_signal = torch.sin(torch.linspace(0, 4 * torch.pi, 128))

    # 2. Função base a ser orquestrada (simula uma rotação SO(4) ou filtro)
    # Esta função simplesmente retorna a soma dos parâmetros que recebe.
    def base_mock_function(input_tensor, param1=0, param2=0):
        return input_tensor + param1 + param2

    # 3. Orquestrar a transformação
    # O orquestrador deve analisar o mock_signal e gerar novos parâmetros para a função base.
    result = orchestrator.orchestrate_transformation(
        signal=mock_signal,
        transformation_type='test_transformation',
        base_function=base_mock_function,
        input_tensor=torch.tensor(100.0) # Um valor inicial para a função base
    )

    # 4. Verificações
    # O resultado não deve ser o valor inicial (100.0), pois o orquestrador
    # deve ter adicionado os parâmetros dinâmicos (param1, param2).
    assert result.item() != 100.0
    # O resultado deve ser um tensor escalar
    assert result.dim() == 0
    # O resultado deve ser maior que o valor inicial, assumindo que os params são positivos
    assert result.item() > 100.0
    
    print("✅ Teste de Orquestração de Função Base passou.")
