#!/usr/bin/env python3
"""
Validator - Funções de validação para o pipeline ΨQRH.
"""

import torch
from typing import Dict, Any

class Validator:
    """
    Agrupa métodos para validar a consistência matemática e a qualidade da saída.
    """
    def __init__(self):
        """
        Inicializa o validador.
        """
        print("✅ Validator inicializado.")

    def validate_mathematical_consistency(self, 
                                          psi_quaternions: torch.Tensor, 
                                          psi_filtered: torch.Tensor, 
                                          psi_rotated: torch.Tensor) -> Dict[str, Any]:
        """
        Valida a consistência matemática do pipeline quântico.
        (Lógica migrada de psiqrh.py: _validate_mathematical_consistency)
        """
        E_quaternions = torch.sum(psi_quaternions.abs() ** 2).item()
        E_filtered = torch.sum(psi_filtered.abs() ** 2).item()
        E_rotated = torch.sum(psi_rotated.abs() ** 2).item()

        # Evitar divisão por zero
        filtering_conservation = E_filtered / (E_quaternions + 1e-10)
        rotation_conservation = E_rotated / (E_filtered + 1e-10)

        energy_conservation_ratio = (filtering_conservation + rotation_conservation) / 2.0
        unitarity_score = 1.0 - abs(energy_conservation_ratio - 1.0)

        finite_values = torch.isfinite(psi_rotated).all().item()

        norm_initial = torch.norm(psi_quaternions).item()
        norm_final = torch.norm(psi_rotated).item()
        norm_preservation = min(norm_final / (norm_initial + 1e-10), norm_initial / (norm_final + 1e-10))

        validation_passed = (
            unitarity_score > 0.95 and
            finite_values and
            norm_preservation > 0.95
        )

        return {
            'energy_conservation_ratio': energy_conservation_ratio,
            'unitarity_score': unitarity_score,
            'numerical_stability': finite_values,
            'norm_preservation': norm_preservation,
            'validation_passed': validation_passed
        }

    def validate_generated_text(self, 
                                generated_text: str, 
                                input_text: str, 
                                psi_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida a qualidade e a sanidade do texto gerado.
        (Lógica migrada de psiqrh.py: _validate_generated_text)
        """
        is_valid = True
        validation_details = []

        if not generated_text or generated_text.isspace():
            is_valid = False
            validation_details.append("Texto gerado é vazio ou contém apenas espaços.")

        # Verificar repetição excessiva (ex: "aaaaa" ou "ababab")
        if len(generated_text) > 10:
            # Repetição de um único caractere
            if len(set(generated_text)) == 1:
                is_valid = False
                validation_details.append("Repetição excessiva de um único caractere.")
            # Repetição de um padrão de 2 caracteres
            if len(set(generated_text)) == 2 and generated_text.startswith(generated_text[:2] * 3):
                 is_valid = False
                 validation_details.append("Repetição excessiva de um padrão de 2 caracteres.")

        if not psi_stats.get('finite', False):
            is_valid = False
            validation_details.append("Estado quântico continha valores não finitos (NaN ou Inf).")

        return {
            'is_valid': is_valid,
            'validation_details': validation_details if validation_details else ["Nenhum problema detectado."]
        }

# Exemplo de uso
if __name__ == '__main__':
    validator = Validator()

    # 1. Testar validação matemática
    print("\n--- Testando Validação Matemática ---")
    psi_q = torch.randn(1, 10, 64, 4)
    psi_f = psi_q * 0.98 # Simula pequena perda de energia
    psi_r = psi_f * 1.01 # Simula pequeno ganho de energia
    math_results = validator.validate_mathematical_consistency(psi_q, psi_f, psi_r)
    for key, value in math_results.items():
        print(f"  {key}: {value}")

    # 2. Testar validação de texto
    print("\n--- Testando Validação de Texto ---")
    text_val_results1 = validator.validate_generated_text("Este é um texto válido.", "in", {'finite': True})
    print(f"Texto 1: {text_val_results1}")
    
    text_val_results2 = validator.validate_generated_text("aaaaaa", "in", {'finite': True})
    print(f"Texto 2: {text_val_results2}")

    text_val_results3 = validator.validate_generated_text("", "in", {'finite': True})
    print(f"Texto 3: {text_val_results3}")

    text_val_results4 = validator.validate_generated_text("ok", "in", {'finite': False})
    print(f"Texto 4: {text_val_results4}")
