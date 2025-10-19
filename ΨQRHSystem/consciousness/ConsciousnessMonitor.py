#!/usr/bin/env python3
"""
ConsciousnessMonitor - Processa o estado de consciência e aplica bootstrap.
"""

import torch
from typing import Dict, Any, Tuple, Optional

# Importar componentes necessários
try:
    from .fractal_consciousness_processor import create_consciousness_processor
    from .consciousness_bootstrapper import create_consciousness_bootstrapper
except ImportError as e:
    print(f"Falha ao importar componentes de consciência: {e}")
    # Stubs para permitir que o arquivo seja criado mesmo sem as dependências completas
    def create_consciousness_processor(**kwargs):
        print("⚠️  Usando stub para create_consciousness_processor")
        return None
    def create_consciousness_bootstrapper(**kwargs):
        print("⚠️  Usando stub para create_consciousness_bootstrapper")
        return None

class ConsciousnessMonitor:
    """
    Responsável pela quarta etapa: análise e modulação da consciência.
    """
    def __init__(self, embedding_dim: int, device: str = 'cpu'):
        """
        Inicializa o monitor de consciência.

        Args:
            embedding_dim: A dimensão do embedding usada pelo processador.
            device: O dispositivo computacional.
        """
        self.device = device
        self.consciousness_processor = create_consciousness_processor(
            embedding_dim=embedding_dim, device=self.device
        )
        self.bootstrapper = create_consciousness_bootstrapper(
            chaos_strength=0.1,
            logistic_r=3.99,
            min_fci_threshold=0.15,
            max_boost_iterations=5
        )

        if self.consciousness_processor and self.bootstrapper:
            print("✅ ConsciousnessMonitor inicializado com sucesso.")
        else:
            print("❌ Falha ao inicializar componentes do ConsciousnessMonitor.")

    def process_consciousness(self, psi: torch.Tensor, fractal_dim: float) -> Dict:
        """
        Processa a consciência e calcula o FCI (Fractal Consciousness Index).
        (Lógica migrada de psiqrh.py: _process_consciousness)
        """
        if not self.consciousness_processor:
            print("⚠️  Processador de consciência não disponível.")
            return {'FCI': 0.0, 'D_fractal': fractal_dim, 'state': 'UNAVAILABLE', 'CLZ': 0.0}

        # O processador espera um tensor 3D [batch, seq, embed]. O psi é 4D [batch, seq, embed, 4].
        # Calculamos a norma (magnitude) de cada quatérnio para reduzir a dimensionalidade.
        input_for_consciousness = torch.norm(psi, p=2, dim=-1)

        # Calcular os dados de acoplamento obrigatórios (spectral_energy e quaternion_phase)
        # A energia é a norma L2 dos quaternions, com média na dimensão da sequência.
        spectral_energy = torch.norm(psi, p=2, dim=-1).mean(dim=1)

        # A fase é o ângulo entre a parte real e a magnitude da parte vetorial, com média na sequência.
        real_part = psi[..., 0]
        vector_part_norm = torch.norm(psi[..., 1:], p=2, dim=-1)
        quaternion_phase = torch.atan2(vector_part_norm, real_part).mean(dim=1)

        results = self.consciousness_processor.forward(
            input_for_consciousness,
            spectral_energy=spectral_energy,
            quaternion_phase=quaternion_phase
        )

        return {
            'FCI': results.get('fci', 0.0),
            'D_fractal': fractal_dim,
            'state': getattr(results.get('final_consciousness_state', None), 'name', 'UNKNOWN'),
            'CLZ': results.get('clz', 0.5)
        }

    def apply_bootstrap(self, psi: torch.Tensor, consciousness_results: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Aplica bootstrap cognitivo para estados de baixa consciência (FCI < 0.3).
        (Lógica migrada de psiqrh.py: _apply_consciousness_bootstrap)
        """
        if not self.bootstrapper or not self.consciousness_processor:
            print("⚠️  Bootstrapper ou processador não disponível. Retornando estado original.")
            return psi, consciousness_results

        # O bootstrapper espera o tensor sem a dimensão de batch
        psi_bootstrapped, new_consciousness = self.bootstrapper.apply_bootstrap(
            psi.squeeze(0),
            consciousness_results,
            self.consciousness_processor
        )

        # Adiciona a dimensão de batch de volta
        return psi_bootstrapped.unsqueeze(0), new_consciousness

    def process(self, psi: torch.Tensor, fractal_dim: float) -> Tuple[torch.Tensor, Dict]:
        """
        Executa o pipeline completo de monitoramento de consciência.
        """
        consciousness_results = self.process_consciousness(psi, fractal_dim)
        
        # Aplica bootstrap se o FCI for baixo
        if consciousness_results.get('FCI', 1.0) < 0.3:
            print("🧠 FCI baixo detectado. Aplicando bootstrap cognitivo...")
            psi, consciousness_results = self.apply_bootstrap(psi, consciousness_results)
        
        return psi, consciousness_results

# Exemplo de uso
if __name__ == '__main__':
    device = 'cpu'
    embedding_dim = 64

    # 1. Criar um estado quântico de exemplo
    psi_initial = torch.randn(1, 100, embedding_dim, 4, device=device)
    fractal_dimension_example = 1.75

    # 2. Inicializar e usar o monitor
    monitor = ConsciousnessMonitor(embedding_dim=embedding_dim, device=device)
    if monitor.consciousness_processor:
        psi_final, results = monitor.process(psi_initial, fractal_dimension_example)

        print(f"\nResultados da Consciência:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        print(f"Shape do Psi inicial: {psi_initial.shape}")
        print(f"Shape do Psi final: {psi_final.shape}")
    else:
        print("\nNão foi possível executar o exemplo pois os componentes de consciência não foram carregados.")
