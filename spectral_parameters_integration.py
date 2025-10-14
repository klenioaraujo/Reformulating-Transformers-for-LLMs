#!/usr/bin/env python3
"""
Sistema de Integração de Parâmetros Espectrais
==============================================

Extrai e integra parâmetros espectrais (α, β, D) dos modelos semânticos convertidos
para uso na matriz quântica de conversão aprimorada.

Princípios Físicos Integrados:
- Equação de Padilha: f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))
- Dimensão Fractal: D = (3 - β) / 2
- Filtragem Espectral: F(k) = exp(i α · arctan(ln(|k| + ε)))

Uso:
    from spectral_parameters_integration import SpectralParametersIntegrator
    integrator = SpectralParametersIntegrator()
    params = integrator.extract_spectral_parameters('gpt2')
"""

import torch
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class SpectralParametersIntegrator:
    """
    Integra parâmetros espectrais dos modelos semânticos convertidos
    com a matriz quântica de conversão
    """

    def __init__(self, models_base_path: str = "models/semantic"):
        self.models_base_path = Path(models_base_path)
        self.semantic_models = self._discover_semantic_models()
        self.spectral_parameters = {}

    def _discover_semantic_models(self) -> List[str]:
        """Descobre modelos semânticos disponíveis"""
        models = []
        for model_file in self.models_base_path.glob("psiqrh_semantic_*.pt"):
            model_name = model_file.stem.replace("psiqrh_semantic_", "")
            models.append(model_name)
        return models

    def extract_spectral_parameters(self, model_name: str) -> Dict:
        """
        Extrai parâmetros espectrais α, β, D dos modelos convertidos
        usando análise espectral avançada
        """
        model_path = self.models_base_path / f"psiqrh_semantic_{model_name}.pt"

        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            model_state = checkpoint.get('model_state_dict', {})

            # Análise espectral dos embeddings
            spectral_params = self._analyze_model_spectrum(model_state, model_name)

            # Integração com Equação de Padilha
            padilha_params = self._compute_padilha_parameters(spectral_params)

            self.spectral_parameters[model_name] = {
                **spectral_params,
                **padilha_params,
                'model_size_mb': os.path.getsize(model_path) / (1024 * 1024),
                'extraction_time': str(torch.tensor(1.0))  # Placeholder para timestamp
            }

            return self.spectral_parameters[model_name]

        except Exception as e:
            print(f"❌ Erro extraindo parâmetros de {model_name}: {e}")
            return {}

    def _analyze_model_spectrum(self, model_state: Dict, model_name: str) -> Dict:
        """Análise espectral avançada do estado do modelo"""

        # Extrair embeddings principais
        embedding_keys = [k for k in model_state.keys() if 'embedding' in k.lower()]
        spectral_analysis = {}

        for key in embedding_keys[:3]:  # Analisar primeiros 3 embeddings
            tensor = model_state[key]

            if tensor.dim() >= 2:
                # Análise de valores singulares (SVD)
                try:
                    U, S, Vt = torch.linalg.svd(tensor.to(torch.complex64))

                    # Parâmetro α - escala espectral
                    alpha = torch.log(torch.mean(S) + 1e-8).item()

                    # Parâmetro β - decaimento espectral
                    spectral_decay = self._compute_spectral_decay(S)
                    beta = spectral_decay.item()

                    # Dimensão fractal D = (3 - β) / 2
                    fractal_dim = (3 - beta) / 2

                    spectral_analysis[f"{key}_spectral"] = {
                        'alpha': alpha,
                        'beta': beta,
                        'fractal_dim': fractal_dim,
                        'singular_values': S[:10].tolist(),  # Top 10 valores
                        'condition_number': (S[0] / S[-1]).item() if S[-1] > 0 else float('inf')
                    }
                except Exception as e:
                    print(f"⚠️  Erro na análise SVD de {key}: {e}")
                    continue

        return self._aggregate_spectral_parameters(spectral_analysis, model_name)

    def _compute_spectral_decay(self, singular_values: torch.Tensor) -> torch.Tensor:
        """Computa parâmetro β de decaimento espectral"""
        # Ajuste exponencial para decaimento espectral
        log_s = torch.log(singular_values[:20] + 1e-8)  # Primeiros 20 valores
        indices = torch.arange(len(log_s), dtype=torch.float32)

        # Regressão linear para estimar β
        if len(log_s) > 1:
            # Aproximação simples de decaimento
            beta = -torch.mean(torch.diff(log_s) / torch.diff(indices))
            return torch.clamp(beta, 0.1, 2.9)  # β entre 0.1 e 2.9
        return torch.tensor(1.0)

    def _compute_padilha_parameters(self, spectral_params: Dict) -> Dict:
        """Computa parâmetros da Equação de Padilha baseados na análise espectral"""

        # Extrair α e β médios
        alpha_values = [v.get('alpha', 1.0) for v in spectral_params.values()
                       if isinstance(v, dict) and 'alpha' in v]
        beta_values = [v.get('beta', 1.0) for v in spectral_params.values()
                      if isinstance(v, dict) and 'beta' in v]

        alpha_avg = np.mean(alpha_values) if alpha_values else 1.0
        beta_avg = np.mean(beta_values) if beta_values else 1.0

        # Equação de Padilha: f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))
        padilha_params = {
            'I0': 1.0,  # Intensidade base
            'omega': alpha_avg,  # Frequência angular relacionada a α
            'k': beta_avg,  # Número de onda relacionado a β
            'alpha_padilha': alpha_avg,
            'beta_padilha': beta_avg,
            'base_equation': f"I₀ sin(ωt + {alpha_avg:.3f}λ) e^(i(ωt - {beta_avg:.3f}λ + {beta_avg:.3f}λ²))"
        }

        return padilha_params

    def _aggregate_spectral_parameters(self, spectral_analysis: Dict, model_name: str) -> Dict:
        """Agrega parâmetros espectrais de diferentes componentes"""

        alphas = [v['alpha'] for v in spectral_analysis.values() if isinstance(v, dict)]
        betas = [v['beta'] for v in spectral_analysis.values() if isinstance(v, dict)]
        fractal_dims = [v['fractal_dim'] for v in spectral_analysis.values() if isinstance(v, dict)]

        return {
            'alpha_final': np.mean(alphas) if alphas else 1.0,
            'beta_final': np.mean(betas) if betas else 1.0,
            'fractal_dim_final': np.mean(fractal_dims) if fractal_dims else 1.0,
            'alpha_std': np.std(alphas) if alphas else 0.0,
            'beta_std': np.std(betas) if betas else 0.0,
            'component_analysis': spectral_analysis,
            'model_name': model_name
        }

    def get_available_models(self) -> List[str]:
        """Retorna lista de modelos semânticos disponíveis"""
        return self.semantic_models.copy()

    def get_model_parameters(self, model_name: str) -> Optional[Dict]:
        """Retorna parâmetros espectrais de um modelo específico"""
        if model_name not in self.spectral_parameters:
            self.extract_spectral_parameters(model_name)
        return self.spectral_parameters.get(model_name)


# Função de teste
if __name__ == "__main__":
    integrator = SpectralParametersIntegrator()

    print("🔍 Modelos semânticos disponíveis:")
    for model in integrator.get_available_models():
        print(f"   📁 {model}")

    # Testar extração de parâmetros
    if integrator.semantic_models:
        test_model = integrator.semantic_models[0]
        print(f"\n🔬 Extraindo parâmetros de: {test_model}")
        params = integrator.extract_spectral_parameters(test_model)

        if params:
            print("✅ Parâmetros extraídos com sucesso:")
            print(f"   α (filtragem): {params.get('alpha_final', 'N/A'):.3f}")
            print(f"   β (decaimento): {params.get('beta_final', 'N/A'):.3f}")
            print(f"   D (fractal): {params.get('fractal_dim_final', 'N/A'):.3f}")
            print(f"   Equação: {params.get('base_equation', 'N/A')}")
        else:
            print("❌ Falha na extração de parâmetros")
    else:
        print("⚠️  Nenhum modelo semântico encontrado")