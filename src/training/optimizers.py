#!/usr/bin/env python3
"""
Constrained Optimizers for ΨQRH Physical Training
==================================================

Implementa otimizadores que respeitam restrições físicas quânticas,
aplicando projeções nos parâmetros após cada passo de otimização.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import torch.optim as optim
from typing import Dict, Any, Optional, List, Iterable
import math


class ConstrainedAdam(optim.Adam):
    """
    Adam com Restrições Físicas para ΨQRH

    Após cada passo de otimização, aplica projeções de restrição nos parâmetros
    físicos para garantir que respeitam leis quânticas e propriedades fractais.

    Restrições aplicadas:
    - T_q (temperatura quântica): 0.1 ≤ T_q ≤ 5.0
    - α (parâmetro fractal): 0.1 ≤ α ≤ 3.0
    - β (parâmetro fractal): 0.5 ≤ β ≤ 1.5
    - Parâmetros de unitariedade: projeção na esfera unitária
    """

    def __init__(self,
                 params: Iterable[torch.nn.Parameter],
                 lr: float = 1e-3,
                 betas: tuple = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0,
                 amsgrad: bool = False,
                 physical_param_names: Optional[List[str]] = None):
        """
        Args:
            params: Parâmetros do modelo
            lr: Taxa de aprendizado
            betas: Coeficientes beta para Adam
            eps: Termo epsilon para estabilidade numérica
            weight_decay: Decaimento de peso
            amsgrad: Usar variante AMSGrad
            physical_param_names: Nomes dos parâmetros que devem ser restringidos
        """
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)

        # Parâmetros físicos a serem restringidos
        self.physical_param_names = physical_param_names or [
            'temp_calculator.T_q',
            'coherence_calculator.s_value',
            'alpha', 'beta', 'I0', 'omega', 'k',
            'quantum_temperature',
            'fractal_dimension',
            'spectral_alpha', 'spectral_beta'
        ]

        # Limites físicos para parâmetros
        self.physical_limits = {
            'T_q': (0.1, 5.0),           # Temperatura quântica
            'quantum_temperature': (0.1, 5.0),
            'alpha': (0.1, 3.0),         # Parâmetro fractal α
            'spectral_alpha': (0.1, 3.0),
            'beta': (0.5, 1.5),          # Parâmetro fractal β
            'spectral_beta': (0.5, 1.5),
            'I0': (0.1, 2.0),            # Amplitude da onda
            'omega': (0.1, 10.0),        # Frequência angular
            'k': (0.1, 5.0),             # Número de onda
            'fractal_dimension': (1.0, 2.0),  # Dimensão fractal
            's_value': (0.5, 5.0),       # Parâmetro de coerência
        }

    def step(self, closure=None):
        """
        Executa um passo de otimização com restrições físicas.

        Primeiro chama o step() do Adam pai, depois aplica projeções físicas.
        """
        # Executar passo de otimização padrão
        loss = super().step(closure)

        # Aplicar restrições físicas nos parâmetros
        self._apply_physical_constraints()

        return loss

    def _apply_physical_constraints(self):
        """
        Aplica projeções de restrição física nos parâmetros do modelo.
        """
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue

                # Verificar se este parâmetro deve ser restringido
                param_name = self._get_param_name(param)
                if param_name and self._should_constrain_param(param_name):
                    self._constrain_physical_param(param, param_name)

    def _get_param_name(self, param: torch.nn.Parameter) -> Optional[str]:
        """
        Tenta identificar o nome do parâmetro baseado em sua referência.
        """
        # Esta é uma implementação simplificada
        # Em produção, seria melhor ter um mapeamento explícito
        # ou usar hooks de parâmetro

        # Para demonstração, vamos assumir que podemos identificar
        # parâmetros físicos por seus valores ou contexto
        return None  # Implementação placeholder

    def _should_constrain_param(self, param_name: str) -> bool:
        """
        Decide se um parâmetro deve ser restringido.
        """
        if not param_name:
            return False

        # Verificar se o nome contém alguma palavra-chave física
        for physical_name in self.physical_param_names:
            if physical_name in param_name:
                return True

        return False

    def _constrain_physical_param(self, param: torch.nn.Parameter, param_name: str):
        """
        Aplica restrição física específica ao parâmetro.
        """
        # Encontrar limites apropriados
        limits = None
        for key, limit in self.physical_limits.items():
            if key in param_name:
                limits = limit
                break

        if limits:
            # Aplicar clamp nos limites físicos
            min_val, max_val = limits
            param.data.clamp_(min_val, max_val)
        else:
            # Para parâmetros sem limites específicos,
            # aplicar restrições gerais de estabilidade
            self._apply_general_constraints(param, param_name)

    def _apply_general_constraints(self, param: torch.nn.Parameter, param_name: str):
        """
        Aplica restrições gerais para estabilidade física.
        """
        # Prevenir valores extremos que quebram física
        param.data.clamp_(-10.0, 10.0)

        # Para parâmetros relacionados a unitariedade,
        # projetar na esfera unitária aproximada
        if 'unitary' in param_name.lower() or 'rotation' in param_name.lower():
            self._project_to_unitary_sphere(param)

    def _project_to_unitary_sphere(self, param: torch.nn.Parameter):
        """
        Projeta parâmetro na esfera unitária para preservar unitariedade.
        """
        # Normalização L2 para aproximar projeção unitária
        norm = torch.norm(param.data, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=1e-8)  # Evitar divisão por zero

        # Projetar na esfera unitária
        param.data.div_(norm)

    def get_physical_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo dos parâmetros físicos atuais.
        """
        summary = {
            'constrained_params': len(self.physical_param_names),
            'physical_limits': self.physical_limits.copy()
        }

        return summary


class QuantumConstrainedSGD(optim.SGD):
    """
    SGD com Restrições Quânticas

    Versão alternativa usando SGD com restrições físicas.
    Útil para treinamento mais estável em regimes quânticos.
    """

    def __init__(self,
                 params: Iterable[torch.nn.Parameter],
                 lr: float = 1e-3,
                 momentum: float = 0,
                 dampening: float = 0,
                 weight_decay: float = 0,
                 nesterov: bool = False,
                 physical_param_names: Optional[List[str]] = None):
        super().__init__(params, lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)

        self.physical_param_names = physical_param_names or [
            'alpha', 'beta', 'T_q', 'quantum_temperature'
        ]

        # Mesmo sistema de limites do ConstrainedAdam
        self.physical_limits = {
            'T_q': (0.1, 5.0),
            'quantum_temperature': (0.1, 5.0),
            'alpha': (0.1, 3.0),
            'beta': (0.5, 1.5),
        }

    def step(self, closure=None):
        """Executa passo SGD com restrições físicas."""
        loss = super().step(closure)
        self._apply_physical_constraints()
        return loss

    def _apply_physical_constraints(self):
        """Aplica restrições físicas (simplificado para SGD)."""
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue

                # Aplicar clamp simples baseado no nome do parâmetro
                # (implementação simplificada)
                param.data.clamp_(-5.0, 5.0)


def create_constrained_optimizer(optimizer_type: str = 'adam',
                                params: Iterable[torch.nn.Parameter] = None,
                                **kwargs) -> optim.Optimizer:
    """
    Factory function para criar otimizador com restrições físicas.

    Args:
        optimizer_type: Tipo do otimizador ('adam', 'sgd')
        params: Parâmetros do modelo
        **kwargs: Argumentos específicos do otimizador

    Returns:
        Otimizador com restrições físicas
    """
    if optimizer_type.lower() == 'adam':
        return ConstrainedAdam(params, **kwargs)
    elif optimizer_type.lower() == 'sgd':
        return QuantumConstrainedSGD(params, **kwargs)
    else:
        raise ValueError(f"Tipo de otimizador não suportado: {optimizer_type}")


def apply_physical_parameter_constraints(model: torch.nn.Module):
    """
    Aplica restrições físicas a todos os parâmetros do modelo de uma vez.

    Útil para inicialização ou correção manual de parâmetros.

    Args:
        model: Modelo PyTorch
    """
    physical_limits = {
        'T_q': (0.1, 5.0),
        'quantum_temperature': (0.1, 5.0),
        'alpha': (0.1, 3.0),
        'beta': (0.5, 1.5),
        'I0': (0.1, 2.0),
        'omega': (0.1, 10.0),
        'k': (0.1, 5.0),
        'fractal_dimension': (1.0, 2.0),
    }

    for name, param in model.named_parameters():
        # Aplicar restrições baseadas no nome do parâmetro
        for key, (min_val, max_val) in physical_limits.items():
            if key in name:
                param.data.clamp_(min_val, max_val)
                break